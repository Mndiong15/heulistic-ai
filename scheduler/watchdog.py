"""
Detects and cleans up orphaned training instances.

A job is considered orphaned when it has been running but sent no heartbeat
for HEARTBEAT_TIMEOUT_MINUTES. We also catch jobs that were launched but never
sent their first heartbeat (instance failed to boot), using created_at + a
generous 15-minute window to allow for cold-start.
"""

import logging

import psycopg2

from config import settings
from provisioner import terminate

logger = logging.getLogger(__name__)

_ORPHAN_QUERY = """
    SELECT id, instance_id
    FROM jobs
    WHERE status IN ('running', 'checkpointing')
      AND (
          -- Sent at least one heartbeat but then went silent
          (last_heartbeat IS NOT NULL
           AND last_heartbeat < NOW() - INTERVAL '%s minutes')
          OR
          -- Never sent a heartbeat; allow 15 min for instance cold-start
          (last_heartbeat IS NULL
           AND created_at < NOW() - INTERVAL '15 minutes')
      )
      AND instance_id IS NOT NULL
      AND instance_id != 'provisioning'
    UNION
    -- Provisioning jobs that never became running after 20 min (boot failure)
    SELECT id, instance_id
    FROM jobs
    WHERE status = 'provisioning'
      AND provisioned_at < NOW() - INTERVAL '20 minutes'
      AND instance_id IS NOT NULL
      AND instance_id != 'provisioning'
"""

_CANCELLED_WITH_INSTANCE_QUERY = """
    SELECT id, instance_id
    FROM jobs
    WHERE status = 'cancelled'
      AND instance_id IS NOT NULL
      AND instance_id != 'provisioning'
      AND completed_at > NOW() - INTERVAL '1 hour'
"""

_FAIL_JOB = """
    UPDATE jobs
    SET status        = 'failed',
        error_message = 'Instance heartbeat timeout — job may have crashed or failed to start',
        completed_at  = NOW()
    WHERE id = %s
"""

_CLEAR_INSTANCE = """
    UPDATE jobs SET instance_id = NULL WHERE id = %s
"""


def run(conn: "psycopg2.connection") -> None:  # type: ignore[name-defined]
    """Check for orphaned jobs and terminate their instances."""
    with conn.cursor() as cur:
        cur.execute(_ORPHAN_QUERY, (settings.heartbeat_timeout_minutes,))
        orphans = cur.fetchall()

    if orphans:
        logger.warning("Watchdog found %d orphaned job(s)", len(orphans))

    for job in orphans:
        job_id = job["id"]
        instance_id = job["instance_id"]

        logger.warning("Job %s: orphaned (instance %s), terminating", job_id, instance_id)

        # `terminate` dispatches to the Docker provisioner in local mode, so
        # this runs in both — a hung local container must be stopped too.
        try:
            terminate(instance_id)
        except Exception as exc:
            logger.error("Failed to terminate %s: %s", instance_id, exc)

        with conn.cursor() as cur:
            cur.execute(_FAIL_JOB, (job_id,))
        conn.commit()

        logger.info("Job %s marked as failed", job_id)

    with conn.cursor() as cur:
        cur.execute(_CANCELLED_WITH_INSTANCE_QUERY)
        cancelled = cur.fetchall()

    for job in cancelled:
        job_id = job["id"]
        instance_id = job["instance_id"]

        logger.info("Job %s: cancelled — terminating instance %s", job_id, instance_id)

        try:
            terminate(instance_id)
        except Exception as exc:
            logger.error("Failed to terminate %s: %s", instance_id, exc)

        with conn.cursor() as cur:
            cur.execute(_CLEAR_INSTANCE, (job_id,))
        conn.commit()
