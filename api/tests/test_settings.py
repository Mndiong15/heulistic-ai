"""API settings loading."""

from app.config import Settings


def test_a_dotenv_from_an_older_release_still_loads(tmp_path):
    """Settings that were removed must not stop the API from starting. The
    per-session and global job caps moved to the scheduler; a .env written
    before that still sets them."""
    env = tmp_path / ".env"
    env.write_text(
        "DATABASE_URL=postgresql+asyncpg://u:p@localhost/db\n"
        "S3_BUCKET=b\n"
        "MAX_JOBS_PER_SESSION=3\n"
        "MAX_CONCURRENT_JOBS_GLOBAL=20\n"
    )
    settings = Settings(_env_file=env)
    assert settings.max_unfinished_jobs_per_user == 50
