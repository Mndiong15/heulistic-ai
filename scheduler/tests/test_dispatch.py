"""Choosing which pending jobs to start.

The scheduler used to start jobs by count: up to 20 at once, 3 per user. The
real limit is the EC2 GPU quota, counted in vCPUs — 64 of them, which is
sixteen 4-vCPU g5.xlarge jobs or a single 48-vCPU g6e.12xlarge. Counting jobs
meant either launching what AWS would refuse or idling capacity it would allow.
"""

from datetime import datetime, timedelta

import pytest

from dispatch import instance_vcpus, plan_dispatch, queue_order, queue_positions

T0 = datetime(2026, 1, 1)


def job(n, user, itype="g5.xlarge", minutes=0, resume=False):
    return {
        "id": n,
        "session_id": user,
        "instance_type": itype,
        "queue_position": 0 if resume else n,
        "created_at": T0 + timedelta(minutes=minutes if minutes else n),
    }


def ids(jobs):
    return [j["id"] for j in jobs]


def cloud(pending, active=(), quota=64, share=0.5):
    return plan_dispatch(pending, list(active), capacity=quota,
                         cost=lambda j: instance_vcpus(j["instance_type"]), user_share=share)


# ── vCPUs ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("itype,vcpus", [
    ("g4dn.xlarge", 4), ("g5.2xlarge", 8), ("g6.4xlarge", 16),
    ("g6e.8xlarge", 32), ("g5.12xlarge", 48), ("g6.16xlarge", 64),
])
def test_vcpus_follow_the_size_suffix(itype, vcpus):
    assert instance_vcpus(itype) == vcpus


def test_an_unknown_size_is_counted_large_not_free():
    # Underestimating would launch past the quota; AWS then refuses and the
    # job is requeued, which is recoverable but wasteful.
    assert instance_vcpus("g9.mystery") >= 48


# ── Capacity ──────────────────────────────────────────────────────────────────

def test_fills_the_quota_and_no_further():
    pending = [job(n, f"u{n}") for n in range(1, 21)]  # 20 users x 4 vCPUs
    assert len(cloud(pending)) == 16


def test_running_jobs_use_up_the_quota():
    active = [job(100 + n, f"r{n}", "g5.4xlarge") for n in range(3)]  # 48 vCPUs
    pending = [job(n, f"u{n}") for n in range(1, 10)]
    assert len(cloud(pending, active)) == 4  # 16 vCPUs left


def test_more_than_twenty_jobs_can_run_when_they_are_small():
    pending = [job(n, f"u{n}", "g4dn.xlarge") for n in range(1, 31)]
    assert len(cloud(pending, quota=128)) == 30


# ── Fairness ──────────────────────────────────────────────────────────────────

def test_every_users_first_job_goes_before_anyones_second():
    # A sweep of ten from one user, submitted first, then one job each from two others.
    pending = [job(n, "sweeper") for n in range(1, 11)] + [job(11, "b"), job(12, "c")]
    started = cloud(pending, quota=12)  # room for three
    assert sorted(j["session_id"] for j in started) == ["b", "c", "sweeper"]


def test_a_user_alone_may_use_the_whole_quota():
    pending = [job(n, "sweeper") for n in range(1, 21)]
    assert len(cloud(pending)) == 16


def test_a_users_share_is_capped_while_others_wait():
    """Fair order counts jobs, so one 32-vCPU job ranks its owner ahead of a
    user running three 4-vCPU ones. The share cap gives the room to whoever
    holds less of it."""
    active = [job(100, "heavy", "g6e.8xlarge")] + [job(101 + n, "light") for n in range(3)]
    pending = [job(1, "heavy"), job(2, "light")]
    assert ids(cloud(pending, active)) == [2]


def test_leftover_room_goes_back_to_a_user_over_their_share():
    # Once nobody else is waiting, idle quota earns nothing — let them use it.
    active = [job(100 + n, "sweeper") for n in range(8)]  # 32 vCPUs: half
    pending = [job(1, "sweeper"), job(2, "other")]
    assert ids(cloud(pending, active)) == [2, 1]


def test_the_share_cap_never_blocks_a_users_first_job():
    # 48 vCPUs is over half the quota, but nothing else of theirs is running.
    pending = [job(1, "big", "g6e.12xlarge"), job(2, "other")]
    assert 1 in ids(cloud(pending))


def test_resumes_go_first():
    pending = [job(1, "a"), job(2, "b", resume=True)]
    assert ids(cloud(pending, quota=4)) == [2]


# ── Big jobs are not starved ──────────────────────────────────────────────────

def test_small_jobs_do_not_jump_a_big_job_that_is_waiting_for_room():
    """If the next job in line needs 48 vCPUs and only 16 are free, starting
    the small jobs behind it would keep those 16 busy — and could keep the big
    job waiting forever as more small ones arrive."""
    active = [job(100 + n, f"r{n}", "g5.4xlarge") for n in range(3)]  # 48 used
    pending = [job(1, "big", "g6e.12xlarge"), job(2, "small"), job(3, "small2")]
    assert cloud(pending, active) == []


def test_a_job_held_back_by_its_share_does_not_block_others():
    active = [job(100, "heavy", "g6e.8xlarge")] + [job(101 + n, "light") for n in range(3)]
    pending = [job(1, "heavy"), job(2, "light"), job(3, "light")]
    assert sorted(ids(cloud(pending, active))) == [2, 3]


def test_a_job_larger_than_the_whole_quota_does_not_block_the_queue():
    # It can never start; holding the line for it would stop everything.
    pending = [job(1, "huge", "g6.16xlarge"), job(2, "small")]
    assert ids(cloud(pending, quota=48)) == [2]


def test_nothing_pending_is_nothing_started():
    assert cloud([]) == []


# ── Queue positions ───────────────────────────────────────────────────────────
# Positions used to be numbered oldest-first while jobs were started in fair
# order, so a sweep's 5th job could show "#2" and then wait behind three
# newer jobs from other users.


def test_queue_order_is_the_fair_order():
    pending = [job(1, "sweeper"), job(2, "sweeper"), job(3, "sweeper"), job(4, "b")]
    assert ids(queue_order(pending, [])) == [1, 4, 2, 3]


def test_queue_order_counts_jobs_already_running():
    active = [job(100, "busy")]
    pending = [job(1, "busy"), job(2, "idle")]
    assert ids(queue_order(pending, active)) == [2, 1]


def test_positions_follow_the_order_jobs_will_be_considered_in():
    pending = [job(1, "sweeper"), job(2, "sweeper"), job(3, "b")]
    assert queue_positions(pending, []) == {1: 1, 3: 2, 2: 3}


def test_resumes_keep_position_zero_and_are_counted_ahead():
    # 0 is what marks a resume for priority; the UI shows no number for it.
    pending = [job(1, "a"), job(2, "b", resume=True)]
    assert queue_positions(pending, []) == {2: 0, 1: 2}


def test_starting_jobs_use_the_same_order_as_the_positions():
    pending = [job(n, "sweeper") for n in range(1, 6)] + [job(6, "b"), job(7, "c")]
    started = cloud(pending, quota=12)
    assert ids(started) == ids(queue_order(pending, []))[:3]
