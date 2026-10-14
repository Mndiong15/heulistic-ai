"""Manages the local Heulistic stack via Docker Compose."""

import os
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error
import webbrowser
from pathlib import Path

_COMPOSE_SRC = Path(__file__).parent / "local_compose.yml"

#: Override the container image tag, for development or to recover from a bad
#: release. An explicit --image-tag beats this.
IMAGE_TAG_ENV = "HEULISTIC_IMAGE_TAG"
_FALLBACK_TAG = "latest"

_HEULISTIC_IMAGES = ("api", "scheduler", "frontend-local", "training")
_THIRD_PARTY_IMAGES = ("postgres:15", "minio/minio", "minio/mc")
_HEULISTIC_DIR = Path.home() / ".heulistic"


def _say(message: str) -> None:
    """Write progress to stdout, flushed.

    The flush is the point. stdout is block-buffered when redirected to a file
    or a pipe while stderr is not, so without it a failure message surfaces
    *above* the attempts that produced it — inverting cause and effect in
    exactly the log someone pastes into a bug report.
    """
    print(message, flush=True)


def _warn(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _compose_file() -> Path:
    _HEULISTIC_DIR.mkdir(exist_ok=True)
    dest = _HEULISTIC_DIR / "local_compose.yml"
    shutil.copy(_COMPOSE_SRC, dest)
    return dest


def _installed_version() -> str | None:
    """The installed heulistic version, or None when running from source."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - Python < 3.8
        return None
    try:
        return version("heulistic")
    except PackageNotFoundError:
        return None


def image_tag(override: str | None = None) -> str:
    """Which image tag to run.

    Pinned to the installed version, so `pip install heulistic==0.1.3` can only
    ever run `v0.1.3` images. Before this they were pinned to `:latest`, and a
    months-old cached image masked a release that had never been published.

    An override exists because the pin is occasionally wrong: an editable
    install reports the version recorded when it was installed, not what the
    source says, and a bad release needs a way back.
    """
    if override:
        return override
    from_env = os.environ.get(IMAGE_TAG_ENV, "").strip()
    if from_env:
        return from_env
    installed = _installed_version()
    return f"v{installed}" if installed else _FALLBACK_TAG


def _env(
    frontend_port: int,
    api_port: int,
    hf_token: str | None,
    use_gpu: bool,
    image_tag: str = _FALLBACK_TAG,
    gpus_per_job: int = 0,
    gpu_count: int = 0,
) -> dict[str, str]:
    return {
        **os.environ,
        "PORT": str(frontend_port),
        "API_PORT": str(api_port),
        "HF_TOKEN": hf_token or "",
        "LOCAL_GPU": "true" if use_gpu else "false",
        # 0 gives each job every GPU, one job at a time. The scheduler cannot
        # see the GPUs from its container, so the count is detected here.
        "GPUS_PER_JOB": str(gpus_per_job if use_gpu else 0),
        "LOCAL_GPU_COUNT": str(gpu_count),
        IMAGE_TAG_ENV: image_tag,
    }


def _image_exists_locally(image: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _pull_images(tag: str) -> None:
    images = [f"heulistic/{name}:{tag}" for name in _HEULISTIC_IMAGES]
    images += list(_THIRD_PARTY_IMAGES)
    for image in images:
        if _image_exists_locally(image):
            _say(f"  Using cached {image}")
            continue
        for attempt in range(1, 4):
            _say(f"  Pulling {image}{'  (retry)' if attempt > 1 else ''}…")
            result = subprocess.run(["docker", "pull", image], stdout=subprocess.DEVNULL)
            if result.returncode == 0:
                break
            if attempt == 3:
                _warn(
                    f"  ✗ Failed to pull {image} after 3 attempts.\n"
                    f"    Images are pinned to this CLI's version, so {tag} may not\n"
                    f"    have been published yet. To run the latest published set:\n"
                    f"      heulistic start --image-tag latest\n"
                    f"    or export {IMAGE_TAG_ENV}=latest\n"
                    f"    If you have a local build, tag it: docker tag <local> {image}"
                )
                sys.exit(1)
            time.sleep(5)


def _wait_for_api(api_port: int, timeout: int = 120) -> None:
    url = f"http://localhost:{api_port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    _warn("  API did not become ready in time — check logs with: heulistic logs")
    sys.exit(1)


def start(
    frontend_port: int,
    api_port: int,
    hf_token: str | None,
    use_gpu: bool,
    image_tag_override: str | None = None,
    gpus_per_job: int = 0,
    gpu_count: int = 0,
) -> None:
    compose = _compose_file()
    tag = image_tag(image_tag_override)
    env = _env(frontend_port, api_port, hf_token, use_gpu, image_tag=tag,
               gpus_per_job=gpus_per_job, gpu_count=gpu_count)

    _say(f"  Using images tagged {tag}")
    _pull_images(tag)

    _say("  Starting services…")
    subprocess.run(
        ["docker", "compose", "-f", str(compose), "up", "-d"],
        env=env,
        check=True,
    )

    _say("  Waiting for API to be ready…")
    _wait_for_api(api_port)

    url = f"http://localhost:{frontend_port}"
    _say(f"\n  Heulistic is running at {url}\n")
    webbrowser.open(url)


def stop() -> None:
    compose = _HEULISTIC_DIR / "local_compose.yml"
    if not compose.exists():
        _say("No local stack found.")
        return
    subprocess.run(["docker", "compose", "-f", str(compose), "down"], check=True)
    _say("Local stack stopped.")


def logs(follow: bool) -> None:
    compose = _HEULISTIC_DIR / "local_compose.yml"
    if not compose.exists():
        _say("No local stack found — run 'heulistic start' first.")
        return
    cmd = ["docker", "compose", "-f", str(compose), "logs"]
    if follow:
        cmd.append("-f")
    subprocess.run(cmd)
