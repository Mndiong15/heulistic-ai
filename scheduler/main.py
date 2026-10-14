"""
Fine-Tuning Platform — Job Scheduler

Responsibilities:
  - Poll the DB every POLL_INTERVAL_SECONDS for pending jobs
  - Respect per-session (max 3) and global (max 20) concurrency limits
  - Launch EC2 GPU instances for eligible jobs
  - Resume-flagged jobs (queue_position=0) are picked before regular FIFO jobs
  - Update queue_position for waiting jobs after each dispatch round
  - Run the watchdog every WATCHDOG_INTERVAL_SECONDS to kill orphaned instances
"""

import logging
import signal
import sys
import time

import psycopg2
from psycopg2.extras import RealDictCursor

import botocore.exceptions

import provisioner
import watchdog
from dispatch import CapacityUnavailable, instance_vcpus, plan_dispatch, queue_positions
from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("scheduler")

# ── Graceful shutdown ─────────────────────────────────────────────────────────

_running = True


def _handle_signal(sig: int, _frame: object) -> None:
    global _running
    logger.info("Received signal %d, shutting down after current tick", sig)
    _running = False


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)

# ── Queries ───────────────────────────────────────────────────────────────────

_ACTIVE_JOBS = """
    SELECT id, session_id, instance_type, kind
    FROM jobs
    WHERE status IN ('provisioning', 'running', 'checkpointing')
"""

# Oldest first, resumes (queue_position=0) ahead; dispatch.plan_dispatch then
# applies the capacity and fairness rules. Bounded so a flood of submissions
# cannot make one tick slow.
_PENDING_JOBS = """
    SELECT j.*, src.model_s3_path AS source_model_s3_path
    FROM jobs j
    LEFT JOIN jobs src ON src.id = j.source_job_id
    WHERE j.status = 'pending'
    ORDER BY
        COALESCE(j.queue_position, 999999) ASC,
        j.created_at ASC
    LIMIT 1000
"""

# Atomic claim: transitions to 'provisioning' atomically to prevent double-pick.
_CLAIM_JOB = """
    UPDATE jobs
    SET status         = 'provisioning',
        instance_id    = 'provisioning',
        provisioned_at = NOW()
    WHERE id = %(job_id)s AND status = 'pending'
"""

_SET_INSTANCE_AND_TOKEN = """
    UPDATE jobs
    SET instance_id    = %(instance_id)s,
        internal_token = %(token_hash)s
    WHERE id = %(job_id)s
"""

_FAIL_JOB = """
    UPDATE jobs
    SET status        = 'failed',
        error_message = %(msg)s,
        completed_at  = NOW()
    WHERE id = %(job_id)s
"""

_REQUEUE_JOB = """
    UPDATE jobs
    SET status         = 'pending',
        instance_id    = NULL,
        provisioned_at = NULL
    WHERE id = %(job_id)s
"""

# AWS error codes that mean "no capacity right now, try again later"
_TRANSIENT_EC2_ERRORS = frozenset({
    "VcpuLimitExceeded",
    "InsufficientInstanceCapacity",
    "MaxSpotInstanceCountExceeded",
    "RequestLimitExceeded",
    "ServiceUnavailable",
})

_HEARTBEAT_UPSERT = """
    INSERT INTO scheduler_heartbeat (id, ticked_at, pending_count, active_count, dispatched_count)
    VALUES (1, NOW(), %(pending)s, %(active)s, %(dispatched)s)
    ON CONFLICT (id) DO UPDATE SET
        ticked_at        = EXCLUDED.ticked_at,
        pending_count    = EXCLUDED.pending_count,
        active_count     = EXCLUDED.active_count,
        dispatched_count = EXCLUDED.dispatched_count
"""

# Positions come from dispatch.queue_positions, so they match the order jobs
# are actually started in. The status guard skips a job that started or was
# cancelled since it was read.
_SET_QUEUE_POSITION = """
    UPDATE jobs SET queue_position = %(pos)s
    WHERE id = %(id)s AND status = 'pending'
"""

# ── Core logic ────────────────────────────────────────────────────────────────


def _connect() -> "psycopg2.connection":  # type: ignore[name-defined]
    return psycopg2.connect(settings.db_url, cursor_factory=RealDictCursor)


def _active_jobs(conn: "psycopg2.connection") -> list:  # type: ignore[name-defined]
    with conn.cursor() as cur:
        cur.execute(_ACTIVE_JOBS)
        return cur.fetchall()


def _pending_jobs(conn: "psycopg2.connection") -> list:  # type: ignore[name-defined]
    with conn.cursor() as cur:
        cur.execute(_PENDING_JOBS)
        return cur.fetchall()


def _choose_jobs(pending: list, active: list) -> list:
    """The pending jobs to start this tick.

    In the cloud capacity is the EC2 GPU quota in vCPUs; locally it is a job
    count, since every local job is handed all of the machine's GPUs.
    """
    if settings.local_mode:
        # Exports run on the CPU (see local_provisioner), so they take no slot.
        capacity, cost = settings.max_concurrent_jobs, (lambda job: 0 if job.get("kind") == "export" else 1)
    else:
        capacity, cost = settings.vcpu_quota, (lambda job: instance_vcpus(job["instance_type"]))
    return plan_dispatch(
        pending, active, capacity=capacity, cost=cost, user_share=settings.user_vcpu_share
    )


def _update_queue_positions(conn: "psycopg2.connection", pending: list, active: list) -> None:  # type: ignore[name-defined]
    """Renumber waiting jobs in fair order, writing only the ones that moved."""
    positions = queue_positions(pending, active)
    changed = [
        {"id": job["id"], "pos": positions[job["id"]]}
        for job in pending
        if positions[job["id"]] != job.get("queue_position")
    ]
    if not changed:
        return
    with conn.cursor() as cur:
        cur.executemany(_SET_QUEUE_POSITION, changed)
    conn.commit()


def _write_heartbeat(conn: "psycopg2.connection", pending: int, active: int, dispatched: int) -> None:  # type: ignore[name-defined]
    with conn.cursor() as cur:
        cur.execute(_HEARTBEAT_UPSERT, {"pending": pending, "active": active, "dispatched": dispatched})
    conn.commit()


def _dispatch(conn: "psycopg2.connection", job: dict) -> None:  # type: ignore[name-defined]
    job_id = job["id"]

    # Atomically claim — if another scheduler process beat us, rowcount == 0
    with conn.cursor() as cur:
        cur.execute(_CLAIM_JOB, {"job_id": job_id})
        claimed = cur.rowcount
    if not claimed:
        logger.warning("Job %s already claimed, skipping", job_id)
        conn.rollback()
        return
    conn.commit()

    logger.info("Dispatching job %s on %s", job_id, job.get("instance_type"))

    try:
        instance_id, token_hash = provisioner.provision(job)
        with conn.cursor() as cur:
            cur.execute(
                _SET_INSTANCE_AND_TOKEN,
                {"instance_id": instance_id, "token_hash": token_hash, "job_id": job_id},
            )
        conn.commit()
    except botocore.exceptions.ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in _TRANSIENT_EC2_ERRORS:
            logger.warning("Job %s: capacity error (%s), requeueing", job_id, code)
            with conn.cursor() as cur:
                cur.execute(_REQUEUE_JOB, {"job_id": job_id})
            conn.commit()
        else:
            logger.exception("Job %s: provisioning failed — %s", job_id, exc)
            with conn.cursor() as cur:
                cur.execute(_FAIL_JOB, {"msg": f"Provisioning failed: {exc}", "job_id": job_id})
            conn.commit()
    except CapacityUnavailable as exc:
        logger.info("Job %s: %s — requeueing", job_id, exc)
        with conn.cursor() as cur:
            cur.execute(_REQUEUE_JOB, {"job_id": job_id})
        conn.commit()
    except Exception as exc:
        logger.exception("Job %s: provisioning failed — %s", job_id, exc)
        with conn.cursor() as cur:
            cur.execute(_FAIL_JOB, {"msg": f"Provisioning failed: {exc}", "job_id": job_id})
        conn.commit()


# ── Main loop ─────────────────────────────────────────────────────────────────


def main() -> None:
    logger.info(
        "Scheduler starting (poll=%ds, watchdog=%ds, heartbeat_timeout=%dm)",
        settings.poll_interval_seconds,
        settings.watchdog_interval_seconds,
        settings.heartbeat_timeout_minutes,
    )

    conn: "psycopg2.connection | None" = None  # type: ignore[name-defined]
    watchdog_ticks = settings.watchdog_interval_seconds // settings.poll_interval_seconds
    tick = 0

    while _running:
        try:
            if conn is None or conn.closed:
                logger.info("Connecting to database…")
                conn = _connect()

            # ── Dispatch pending jobs ──────────────────────────────────────
            active = _active_jobs(conn)
            pending_jobs = _pending_jobs(conn)
            to_start = _choose_jobs(pending_jobs, active) if pending_jobs else []
            if to_start:
                logger.info(
                    "Starting %d of %d pending job(s); %d active",
                    len(to_start), len(pending_jobs), len(active),
                )
            for job in to_start:
                _dispatch(conn, job)
            dispatched = len(to_start)
            started = {job["id"] for job in to_start}
            _update_queue_positions(
                conn,
                [job for job in pending_jobs if job["id"] not in started],
                active + to_start,
            )

            _write_heartbeat(conn, len(pending_jobs) - dispatched, len(active), dispatched)

            # ── Watchdog (every N ticks) ───────────────────────────────────
            if tick % watchdog_ticks == 0:
                watchdog.run(conn)

            tick += 1

        except psycopg2.OperationalError as exc:
            logger.error("DB connection error: %s — reconnecting in 5s", exc)
            try:
                if conn:
                    conn.close()
            except Exception:
                pass
            conn = None
            time.sleep(5)
            continue

        except Exception as exc:
            logger.exception("Unexpected error: %s", exc)
            try:
                if conn and not conn.closed:
                    conn.rollback()
            except Exception:
                conn = None

        time.sleep(settings.poll_interval_seconds)

    logger.info("Scheduler stopped")
    if conn and not conn.closed:
        conn.close()


if __name__ == "__main__":
    main()
