"""Which pending jobs to start — pure, so the policy can be tested without a DB.

Capacity is a budget (the EC2 GPU quota in vCPUs, in the cloud), each job has
a cost against it, and the queue is served fairly:

- Resumes first, then each user's 1st waiting job before anyone's 2nd, and so on.
- While other users are waiting, one user may hold at most `user_share` of the
  budget — except that a user with nothing running can always start a job, so
  a job bigger than the share is not locked out.
- If the next job in line does not fit the free budget, nothing behind it
  starts. Letting smaller jobs past would keep the budget busy and could keep
  a big job waiting indefinitely.
"""

from collections import Counter
from typing import Callable


class CapacityUnavailable(Exception):
    """Raised by a provisioner when the job cannot start yet but may later.
    The scheduler puts the job back in the queue instead of failing it."""


# EC2 sizes map to a fixed vCPU count across the g4dn/g5/g6/g6e families.
_SIZE_VCPUS = {
    "xlarge": 4,
    "2xlarge": 8,
    "4xlarge": 16,
    "8xlarge": 32,
    "12xlarge": 48,
    "16xlarge": 64,
}
# An unrecognised size is assumed big: counting it small would launch past the
# quota, which AWS refuses — recoverable, but each refusal is a wasted attempt.
_UNKNOWN_VCPUS = 64


def instance_vcpus(instance_type: str) -> int:
    size = (instance_type or "").rpartition(".")[2]
    return _SIZE_VCPUS.get(size, _UNKNOWN_VCPUS)


def plan_dispatch(
    pending: list[dict],
    active: list[dict],
    *,
    capacity: int,
    cost: Callable[[dict], int],
    user_share: float,
) -> list[dict]:
    """The pending jobs to start now, in the order to start them."""
    used = sum(cost(j) for j in active)
    user_used: Counter = Counter()
    for j in active:
        user_used[j["session_id"]] += cost(j)

    order = queue_order(pending, active)
    waiting = Counter(j["session_id"] for j in pending)
    started = []
    for j in order:
        user, need = j["session_id"], cost(j)
        waiting[user] -= 1
        if need > capacity:
            continue  # can never start; holding the line for it would stop everything

        others_waiting = any(n > 0 for u, n in waiting.items() if u != user)
        over_share = user_used[user] > 0 and user_used[user] + need > user_share * capacity
        if over_share and others_waiting:
            continue  # held back by fairness, not by room: does not block the line

        if used + need > capacity:
            break  # wait for room rather than let smaller jobs past

        started.append(j)
        used += need
        user_used[user] += need
    return started


def queue_order(pending: list[dict], active: list[dict]) -> list[dict]:
    """Pending jobs in the order they will be considered for starting.

    Resumes first. Then a job's rank is how many of its user's jobs are ahead
    of it, running or queued, so every user's k-th job comes before anyone's
    k+1-th; ties go to the oldest.
    """
    running = Counter(j["session_id"] for j in active)
    seen: Counter = Counter()
    ranked = []
    for j in sorted(pending, key=lambda j: (_is_normal(j), j["created_at"])):
        user = j["session_id"]
        ranked.append((_is_normal(j), running[user] + seen[user], j["created_at"], j))
        seen[user] += 1
    return [entry[-1] for entry in sorted(ranked, key=lambda e: e[:3])]


def queue_positions(pending: list[dict], active: list[dict]) -> dict:
    """Job id -> place in line, 1-based, from the same order jobs start in.

    Resumes stay at 0, which is what marks them for priority (and the UI shows
    no number for them); they still count toward the places behind them.
    """
    return {
        j["id"]: 0 if not _is_normal(j) else place
        for place, j in enumerate(queue_order(pending, active), start=1)
    }


def _is_normal(job: dict) -> int:
    """0 for a resume, which jumps the queue; 1 otherwise."""
    return 0 if job.get("queue_position") == 0 else 1
