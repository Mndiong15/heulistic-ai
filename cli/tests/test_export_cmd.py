"""`heulistic export` — from a finished local job to a model Ollama can run.

Free in the local product: the merge and quantization run in the local stack.
What can stop it is disk — an export briefly holds two full-precision copies
of the model — so that is checked first, and anything too big for the machine
is pointed at the cloud.
"""

import pytest
from typer.testing import CliRunner

from heulistic import client, export_cmd, gpu, stack
from heulistic.main import app

runner = CliRunner()

TRAINED = {
    "job_id": "src-1234-5678",
    "kind": "train",
    "status": "completed",
    "config_yaml": "base_model: org/m-7b\nadapter: lora\n",
    "exports": [],
}
MODELFILE = "FROM ./m-7b-exp1-Q4_K_M.gguf\n"


@pytest.fixture
def api(monkeypatch, tmp_path):
    state = {"started": [], "fetched": [], "ollama": [], "jobs": {"src-1234-5678": dict(TRAINED)}}
    monkeypatch.setattr(stack, "api_url", lambda: "http://localhost:8000")
    monkeypatch.setattr(client, "get_job", lambda u, j: state["jobs"][j])

    def start_export(u, j, quant):
        state["started"].append((j, quant))
        state["jobs"]["exp1"] = {
            "job_id": "exp1", "kind": "export", "status": "completed",
            "export_quant": quant, "modelfile": MODELFILE.replace("Q4_K_M", quant),
        }
        return {"job_id": "exp1"}

    monkeypatch.setattr(client, "start_export", start_export)
    monkeypatch.setattr(client, "stream_logs", lambda u, j: iter([("line", "[export] ..."), ("done", "completed")]))
    monkeypatch.setattr(
        client, "download_url",
        lambda u, j: f"http://localhost:8001/local-download/models/{j}/m-7b-exp1-Q4_K_M.gguf",
    )

    def fetch(url, dest):
        state["fetched"].append((url, dest))
        dest.write_bytes(b"gguf")
        return dest

    monkeypatch.setattr(export_cmd, "_fetch", fetch)
    monkeypatch.setattr(export_cmd, "_free_disk_gb", lambda: 500.0)
    monkeypatch.setattr(export_cmd, "_ollama", lambda: None)
    monkeypatch.setattr(gpu, "detect_hardware", lambda: None)
    monkeypatch.setattr(export_cmd, "_system_ram_gb", lambda: None)
    monkeypatch.chdir(tmp_path)
    return state


def run(*args):
    return runner.invoke(app, ["export", *args])


def test_exports_downloads_and_writes_the_modelfile(api, tmp_path):
    result = run("src-1234-5678", "--quant", "Q4_K_M")
    assert result.exit_code == 0, result.output
    assert api["started"] == [("src-1234-5678", "Q4_K_M")]
    out = tmp_path / "heulistic-src-1234"
    assert (out / "m-7b-exp1-Q4_K_M.gguf").read_bytes() == b"gguf"
    assert (out / "Modelfile").read_text() == MODELFILE


def test_tells_the_user_how_to_run_it_without_ollama_installed(api):
    result = run("src-1234-5678", "--quant", "Q4_K_M")
    assert "ollama create heulistic-src-1234 -f Modelfile" in result.output


def test_registers_the_model_with_ollama_when_it_is_installed(api, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(export_cmd, "_ollama", lambda: "/usr/bin/ollama")
    monkeypatch.setattr(export_cmd, "_run", lambda cmd, cwd: calls.append((cmd, cwd)) or 0)
    result = run("src-1234-5678", "--quant", "Q4_K_M", "--name", "my-model")
    assert result.exit_code == 0, result.output
    assert calls == [(["/usr/bin/ollama", "create", "my-model", "-f", "Modelfile"], tmp_path / "my-model")]
    assert "ollama run my-model" in result.output


class TestChoosingAFormat:
    def test_picks_the_best_that_fits_the_gpu(self, api, monkeypatch):
        from heulistic_axolotl.vram import Hardware
        monkeypatch.setattr(gpu, "detect_hardware",
                            lambda: Hardware(total_vram_gb=24, gpu_count=1, label="RTX 4090"))
        result = run("src-1234-5678")
        assert result.exit_code == 0, result.output
        assert api["started"] == [("src-1234-5678", "Q8_0")]
        assert "RTX 4090" in result.output

    def test_without_a_gpu_sizes_by_ram(self, api, monkeypatch):
        # Macs and CPU-only machines run GGUF from system (unified) memory.
        monkeypatch.setattr(export_cmd, "_system_ram_gb", lambda: 8.0)
        run("src-1234-5678")
        assert api["started"] == [("src-1234-5678", "Q5_K_M")]

    def test_nothing_known_falls_back_to_q4(self, api):
        run("src-1234-5678")
        assert api["started"] == [("src-1234-5678", "Q4_K_M")]

    def test_warns_when_even_q4_will_not_fit(self, api, monkeypatch):
        monkeypatch.setattr(export_cmd, "_system_ram_gb", lambda: 4.0)
        result = run("src-1234-5678")
        assert api["started"] == [("src-1234-5678", "Q4_K_M")]
        assert "will not fit" in result.output.lower()

    def test_an_unknown_format_is_refused(self, api):
        result = run("src-1234-5678", "--quant", "Q2_K")
        assert result.exit_code != 0
        assert api["started"] == []


def test_not_enough_disk_points_at_the_cloud(api, monkeypatch):
    monkeypatch.setattr(export_cmd, "_free_disk_gb", lambda: 10.0)
    result = run("src-1234-5678", "--quant", "Q4_K_M")
    assert result.exit_code == 1
    assert "disk" in result.output.lower()
    assert "heulistic.com" in result.output
    assert api["started"] == []


def test_unknown_free_disk_does_not_block(api, monkeypatch):
    # Docker Desktop keeps its disk inside a VM the CLI cannot measure.
    monkeypatch.setattr(export_cmd, "_free_disk_gb", lambda: None)
    assert run("src-1234-5678", "--quant", "Q4_K_M").exit_code == 0


def test_reuses_an_export_that_already_exists(api):
    api["jobs"]["src-1234-5678"]["exports"] = [{"job_id": "exp1", "quant": "Q8_0", "status": "completed"}]
    api["jobs"]["exp1"] = {"job_id": "exp1", "kind": "export", "status": "completed",
                           "export_quant": "Q8_0", "modelfile": "FROM ./x-Q8_0.gguf\n"}
    result = run("src-1234-5678", "--quant", "Q8_0")
    assert result.exit_code == 0, result.output
    assert api["started"] == []
    assert api["fetched"]


@pytest.mark.parametrize("job", [
    {**TRAINED, "status": "running"},
    {**TRAINED, "kind": "export"},
])
def test_only_a_finished_training_job(api, job):
    api["jobs"]["src-1234-5678"] = job
    result = run("src-1234-5678", "--quant", "Q4_K_M")
    assert result.exit_code == 1
    assert api["started"] == []


def test_a_failed_export_is_reported(api, monkeypatch):
    monkeypatch.setattr(client, "stream_logs", lambda u, j: iter([("done", "failed")]))

    def start_export(u, j, quant):
        api["jobs"]["exp1"] = {"job_id": "exp1", "kind": "export", "status": "failed",
                               "error_message": "out of disk"}
        return {"job_id": "exp1"}

    monkeypatch.setattr(client, "start_export", start_export)
    result = run("src-1234-5678", "--quant", "Q4_K_M")
    assert result.exit_code == 1
    assert "out of disk" in result.output


def test_refuses_when_the_stack_is_down(monkeypatch):
    monkeypatch.setattr(stack, "api_url", lambda: None)
    result = run("job-1")
    assert result.exit_code == 1
    assert "heulistic start" in result.output
