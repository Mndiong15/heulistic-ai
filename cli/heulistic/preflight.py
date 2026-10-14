"""Pre-flight checks before starting the local stack."""

import shutil
import socket
import subprocess
import sys


def _say(message: str) -> None:
    """Progress to stdout, flushed — see local_runner._say for why."""
    print(message, flush=True)


def _warn(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0


_INSTALL_HINT = (
    "Docker is not installed — install it from https://docs.docker.com/get-docker/"
)

# On Windows the daemon is frequently absent because Docker lives inside WSL,
# not because anything needs starting. Saying only "start Docker" sends people
# to the wrong fix.
_WINDOWS_HINT = (
    "Docker Desktop must be running, and its WSL integration enabled. "
    "If you installed Docker inside WSL instead, run `heulistic start` from "
    "your WSL shell rather than from Windows."
)
_UNIX_HINT = "Start the Docker daemon and try again."


def _daemon_error(stderr: str) -> str:
    """The most informative line of `docker info`'s stderr."""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    for line in lines:
        if "cannot connect" in line.lower() or "error" in line.lower():
            return line
    return lines[0] if lines else "Docker is not running."


def docker_problem(platform: str | None = None) -> str | None:
    """Describe why Docker is unusable, or None when it is fine.

    Reports the daemon's own error rather than a guess — the cause differs
    enough between platforms that a fixed message misdirects as often as it
    helps. `platform` defaults to the running one and exists so tests need not
    mutate `sys.platform`, which would send `shutil.which` down the wrong code
    path.
    """
    platform = platform or sys.platform
    if shutil.which("docker") is None:
        return _INSTALL_HINT

    try:
        result = subprocess.run(["docker", "info"], capture_output=True, text=True)
    except OSError as exc:
        return f"Could not run docker: {exc}"

    if result.returncode == 0:
        return None

    hint = _WINDOWS_HINT if platform == "win32" else _UNIX_HINT
    return f"{_daemon_error(result.stderr or '')}\n    {hint}"


_PORT_ATTEMPTS = 20


def find_free_port(preferred: int, attempts: int = _PORT_ATTEMPTS, avoid=()) -> int | None:
    """First free port at or above `preferred`, or None if none is found."""
    claimed = set(avoid)
    for candidate in range(preferred, preferred + attempts):
        if candidate not in claimed and _port_free(candidate):
            return candidate
    return None


def resolve_ports(
    frontend_port: int,
    api_port: int,
    frontend_explicit: bool,
    api_explicit: bool,
) -> tuple[int, int, list[str]]:
    """Pick usable ports, returning (frontend, api, notes).

    A port the user named is never moved — honouring the request and letting
    `run()` report the clash is better than quietly starting somewhere else.
    A defaulted port shifts to the next free one, and every substitution is
    reported so the browser URL is never a surprise.

    When nothing free can be found the original is returned unchanged, so the
    normal "port is in use" error still explains the situation.
    """
    notes: list[str] = []
    claimed: set[int] = set()

    def pick(preferred: int, explicit: bool, label: str) -> int:
        if explicit:
            claimed.add(preferred)
            return preferred
        chosen = find_free_port(preferred, avoid=claimed)
        if chosen is None:
            return preferred
        if chosen != preferred:
            notes.append(f"{label} port {preferred} is in use — using {chosen} instead")
        claimed.add(chosen)
        return chosen

    frontend = pick(frontend_port, frontend_explicit, "Frontend")
    api = pick(api_port, api_explicit, "API")
    return frontend, api, notes


def _nvidia_driver_present() -> bool:
    return shutil.which("nvidia-smi") is not None and (
        subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    )


def _nvidia_runtime_in_docker() -> bool:
    result = subprocess.run(
        ["docker", "info", "--format", "{{range $k, $v := .Runtimes}}{{$k}} {{end}}"],
        capture_output=True, text=True,
    )
    return "nvidia" in result.stdout


def run(frontend_port: int, api_port: int, use_gpu: bool) -> bool:
    """Run pre-flight checks. Returns the effective use_gpu value (may be downgraded to False)."""
    errors: list[str] = []
    warnings: list[str] = []

    problem = docker_problem()
    if problem:
        errors.append(problem)

    if not _port_free(frontend_port):
        errors.append(f"Port {frontend_port} is already in use — use --port to pick a different one")

    if not _port_free(api_port):
        errors.append(f"Port {api_port} is already in use — use --api-port to pick a different one")

    if errors:
        for e in errors:
            _warn(f"  ✗ {e}")
        sys.exit(1)

    if not use_gpu:
        return False

    # Check GPU availability in two steps so we can give targeted guidance.
    has_driver = _nvidia_driver_present()
    has_runtime = _nvidia_runtime_in_docker()

    if has_driver and not has_runtime:
        # GPU present but Docker can't see it — NVIDIA Container Toolkit missing.
        _warn(
            "\n  ✗ NVIDIA GPU detected but Docker cannot access it.\n"
            "    Install the NVIDIA Container Toolkit:\n\n"
            "      # Ubuntu / Debian\n"
            "      curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg\n"
            "      curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \\\n"
            "        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \\\n"
            "        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list\n"
            "      sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit\n"
            "      sudo nvidia-ctk runtime configure --runtime=docker\n"
            "      sudo systemctl restart docker\n\n"
            "    Then run 'heulistic start' again.\n"
            "    To skip GPU and train on CPU: heulistic start --no-gpu\n"
        )
        sys.exit(1)

    if not has_driver:
        warnings.append(
            "No NVIDIA GPU detected — switching to CPU-only mode. "
            "Training will be very slow. Use --no-gpu to suppress this warning."
        )
        for w in warnings:
            _say(f"  ⚠  {w}")
        return False  # downgrade to CPU

    return True  # GPU confirmed
