"""After training: the held-out split, the before/after evaluation, and export.

The orchestration that drives Axolotl and llama.cpp is checked here against a
recorded command sequence; the pure decisions — how many rows to hold out, how
an example splits into prompt and answer, which prompt format an Ollama
Modelfile must use — are checked directly, since getting those wrong produces a
model that runs and quietly answers worse.
"""

import json
import math
from pathlib import Path

import pytest
import yaml

import export
import model_eval
import train


# ── Held-out split ────────────────────────────────────────────────────────────


class TestPlanEvalSplit:
    def test_holds_out_five_percent_up_to_fifty(self):
        assert train.plan_eval_split({}, rows=400) == 20
        assert train.plan_eval_split({}, rows=10_000) == 50

    def test_small_datasets_keep_every_row_for_training(self):
        assert train.plan_eval_split({}, rows=99) is None

    def test_unknown_size_holds_nothing_out(self):
        # A fixed count could take most of a small Hub dataset.
        assert train.plan_eval_split({}, rows=None) is None

    @pytest.mark.parametrize("cfg", [
        {"val_set_size": 0},          # explicitly off
        {"val_set_size": 0.1},        # the user's own split is used as is
        {"test_datasets": [{"path": "x"}]},
    ])
    def test_the_users_own_choice_wins(self, cfg):
        assert train.plan_eval_split(cfg, rows=10_000) is None

    def test_preference_training_is_not_evaluated(self):
        # DPO/ORPO data has no single reference answer to score against.
        assert train.plan_eval_split({"rl": "dpo"}, rows=10_000) is None


def test_counts_rows_in_an_uploaded_jsonl(tmp_path):
    f = tmp_path / "data.jsonl"
    f.write_text('{"a": 1}\n\n{"a": 2}\n{"a": 3}\n')
    assert train.count_rows(f) == 3


class TestPatchConfigHoldsOut:
    def _patch(self, tmp_path, content, holdout):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(content)
        patched = train.patch_config(cfg_path, None, tmp_path / "out", None, holdout=holdout)
        return yaml.safe_load(patched.read_text())

    def test_sets_the_split_and_disables_eval_packing(self, tmp_path):
        cfg = self._patch(tmp_path, "base_model: m\nsample_packing: true\n", holdout=20)
        assert cfg["val_set_size"] == 20
        # A 20-row eval set cannot be packed; Axolotl refuses to start.
        assert cfg["eval_sample_packing"] is False

    def test_no_holdout_leaves_the_config_alone(self, tmp_path):
        cfg = self._patch(tmp_path, "base_model: m\n", holdout=None)
        assert "val_set_size" not in cfg


# ── Evaluation ────────────────────────────────────────────────────────────────


class TestSplitPrompt:
    def test_the_prompt_is_everything_before_the_first_trained_token(self):
        # Axolotl masks the prompt with -100; the answer carries real labels.
        ids = [1, 2, 3, 4, 5, 6]
        labels = [-100, -100, -100, 4, 5, 6]
        assert model_eval.split_prompt(ids, labels) == ([1, 2, 3], [4, 5, 6])

    def test_fully_trained_text_is_split_at_a_quarter(self):
        # Completion data trains on every token, so there is no prompt mask;
        # the model is asked to continue from the opening.
        ids = list(range(8))
        assert model_eval.split_prompt(ids, ids) == ([0, 1], [2, 3, 4, 5, 6, 7])


class TestEvalConfigs:
    def test_lora_base_is_the_untrained_adapter(self):
        # A fresh LoRA adapter starts at zero, so it computes exactly the base
        # model — and loads with the same quantization the run trained with.
        base, tuned = model_eval.eval_configs(
            {"base_model": "org/m-7b", "adapter": "qlora", "load_in_4bit": True}, "/out"
        )
        assert "lora_model_dir" not in base
        assert tuned["lora_model_dir"] == "/out"
        assert base["load_in_4bit"] is tuned["load_in_4bit"] is True

    def test_full_fine_tune_compares_against_the_original_weights(self):
        base, tuned = model_eval.eval_configs({"base_model": "org/m-1b"}, "/out")
        assert base["base_model"] == "org/m-1b"
        assert tuned["base_model"] == "/out"

    def test_multi_gpu_settings_are_dropped(self):
        base, tuned = model_eval.eval_configs(
            {"base_model": "m", "adapter": "lora", "deepspeed": "zero3.json", "fsdp": ["x"]}, "/out"
        )
        for cfg in (base, tuned):
            assert "deepspeed" not in cfg and "fsdp" not in cfg


def test_summary_reports_loss_and_perplexity():
    report = model_eval.summarize(
        base_losses=[(2.0, 10), (4.0, 30)],   # (mean loss, labelled tokens)
        tuned_losses=[(1.0, 10), (1.0, 30)],
        samples=[{"prompt": "p", "reference": "r", "base": "b", "tuned": "t"}],
    )
    assert report["examples"] == 2
    assert report["base"]["loss"] == pytest.approx(3.5)  # token-weighted
    assert report["tuned"]["loss"] == pytest.approx(1.0)
    assert report["tuned"]["perplexity"] == pytest.approx(math.e)
    assert report["samples"][0]["tuned"] == "t"


# ── Export: the prompt format ─────────────────────────────────────────────────


def _cfg(ds_type, **extra):
    return {"base_model": "org/m-7b", "datasets": [{"path": "x", "type": ds_type}], **extra}


class TestPromptFormat:
    @pytest.mark.parametrize("cfg,fmt", [
        (_cfg("alpaca"), "alpaca"),
        (_cfg("completion"), "raw"),
        (_cfg("input_output"), "raw"),
        (_cfg("chat_template", chat_template="chatml"), "chatml"),
        (_cfg("chat_template", chat_template="llama3"), "llama3"),
        (_cfg("sharegpt", chat_template="chatml"), "chatml"),
        (_cfg("alpaca:chatml"), "chatml"),
    ])
    def test_follows_how_the_model_was_trained(self, cfg, fmt):
        assert export.prompt_format(cfg) == fmt

    def test_the_tokenizers_own_template_is_left_to_the_gguf(self):
        # The trained tokenizer's chat template is written into the GGUF, and
        # Ollama reads it from there.
        assert export.prompt_format(_cfg("chat_template")) is None
        assert export.prompt_format(_cfg("chat_template", chat_template="tokenizer_default")) is None

    def test_an_uploaded_dataset_without_a_type_trained_as_alpaca(self):
        # The trainer gives an untyped uploaded dataset type: alpaca.
        assert export.prompt_format({"base_model": "m"}) == "alpaca"


class TestModelfile:
    def test_points_at_the_gguf_beside_it(self):
        mf = export.ollama_modelfile(_cfg("alpaca"), "model-Q4_K_M.gguf")
        assert mf.splitlines()[0] == "FROM ./model-Q4_K_M.gguf"

    def test_alpaca_reproduces_axolotls_training_prompt(self):
        mf = export.ollama_modelfile(_cfg("alpaca"), "m.gguf")
        # Exactly what Axolotl's AlpacaPrompter wrote around every example.
        assert "### Instruction:\n{{ .Prompt }}\n\n### Response:\n" in mf
        assert ('SYSTEM """Below is an instruction that describes a task. '
                'Write a response that appropriately completes the request."""') in mf
        assert 'PARAMETER stop "### Instruction:"' in mf

    def test_chatml_stops_at_end_of_turn(self):
        mf = export.ollama_modelfile(_cfg("chat_template", chat_template="chatml"), "m.gguf")
        assert "<|im_start|>assistant" in mf
        assert 'PARAMETER stop "<|im_end|>"' in mf

    def test_raw_passes_the_prompt_through(self):
        mf = export.ollama_modelfile(_cfg("completion"), "m.gguf")
        assert 'TEMPLATE """{{ .Prompt }}"""' in mf

    def test_no_template_when_the_gguf_carries_it(self):
        mf = export.ollama_modelfile(_cfg("chat_template"), "m.gguf")
        assert "TEMPLATE" not in mf

    def test_context_length_follows_training(self):
        mf = export.ollama_modelfile(_cfg("alpaca", sequence_len=2048), "m.gguf")
        assert "PARAMETER num_ctx 2048" in mf


def test_gguf_name_is_readable_and_unique():
    name = export.gguf_name("meta-llama/Llama-3.2-1B-Instruct", "abcdef12-3456", "Q4_K_M")
    assert name == "llama-3.2-1b-instruct-abcdef12-Q4_K_M.gguf"


# ── Export: the pipeline ──────────────────────────────────────────────────────


class TestExportConfig:
    def test_merges_and_exports_from_the_trained_model(self, tmp_path):
        cfg = export.export_config(
            {"base_model": "org/m", "adapter": "qlora", "load_in_4bit": True,
             "deepspeed": "zero3.json", "datasets": [{"path": "/gone/data.jsonl", "type": "alpaca"}]},
            model_dir=tmp_path / "model", work_dir=tmp_path, quant="Q5_K_M",
        )
        assert cfg["output_dir"] == str(tmp_path / "model")
        assert cfg["lora_model_dir"] == str(tmp_path / "model")
        assert cfg["export"]["quantize"] == ["Q5_K_M"]
        assert cfg["export"]["outfile"].endswith("-{ftype}.gguf")
        assert "deepspeed" not in cfg

    def test_full_fine_tune_has_no_adapter_to_merge(self, tmp_path):
        cfg = export.export_config({"base_model": "org/m"}, tmp_path / "model", tmp_path, "Q4_K_M")
        assert "lora_model_dir" not in cfg


class TestRunExport:
    def _model(self, tmp_path, adapter=True):
        model = tmp_path / "model"
        model.mkdir()
        if adapter:
            (model / "adapter_config.json").write_text("{}")
        return model

    def _fake_run(self, calls, tmp_path):
        def run(cmd, **kwargs):
            calls.append(cmd)
            if "merge-lora" in cmd:
                (tmp_path / "model" / "merged").mkdir()
                (tmp_path / "model" / "merged" / "model.safetensors").write_bytes(b"w" * 10)
            if "export" in cmd:
                for ftype in ("auto", "Q4_K_M"):
                    (tmp_path / f"out-{ftype}.gguf").write_bytes(b"g")
            return type("R", (), {"returncode": 0})()
        return run

    def test_merges_then_exports_then_cleans_up(self, tmp_path):
        model = self._model(tmp_path)
        calls = []
        cfg = {"base_model": "org/m", "adapter": "lora"}
        gguf = export.run_export(cfg, model, tmp_path, "Q4_K_M", run=self._fake_run(calls, tmp_path))

        assert [c[1] for c in calls] == ["merge-lora", "export"]
        assert "--model-dir" in calls[1] and str(model / "merged") in calls[1]
        assert gguf == tmp_path / "out-Q4_K_M.gguf"
        # The full-precision intermediates are as big as the model; gone before upload.
        assert not (tmp_path / "out-auto.gguf").exists()
        assert not (model / "merged").exists()

    def test_a_full_fine_tune_is_exported_directly(self, tmp_path):
        model = self._model(tmp_path, adapter=False)
        calls = []
        export.run_export({"base_model": "org/m"}, model, tmp_path, "Q8_0",
                          run=_exporting(calls, tmp_path, "Q8_0"))
        assert [c[1] for c in calls] == ["export"]
        assert str(model) in calls[0]

    def test_a_failed_step_stops_the_export(self, tmp_path):
        model = self._model(tmp_path)

        def failing(cmd, **kwargs):
            return type("R", (), {"returncode": 1})()

        with pytest.raises(RuntimeError, match="merge-lora"):
            export.run_export({"base_model": "org/m", "adapter": "lora"}, model, tmp_path,
                              "Q4_K_M", run=failing)


def _exporting(calls, tmp_path, quant):
    def run(cmd, **kwargs):
        calls.append(cmd)
        (tmp_path / f"out-{quant}.gguf").write_bytes(b"g")
        (tmp_path / "out-auto.gguf").write_bytes(b"g")
        return type("R", (), {"returncode": 0})()
    return run


def test_disk_needed_scales_with_the_model():
    # Merged weights and the full-precision GGUF coexist at the peak.
    small, large = export.disk_needed_gb(1), export.disk_needed_gb(70)
    assert small < 20
    assert large > 2 * 70 * 2


# ── Wiring into the job ───────────────────────────────────────────────────────


class TestShouldEvaluate:
    def test_needs_held_out_rows(self):
        assert train.should_evaluate({"val_set_size": 20}, plan=None)[0] is True
        assert train.should_evaluate({"test_datasets": [{"path": "x"}]}, plan=None)[0] is True
        ok, reason = train.should_evaluate({}, plan=None)
        assert ok is False and "held out" in reason

    def test_not_when_the_model_had_to_be_sharded(self):
        # Evaluation loads the model in one process on one GPU; a model that
        # only trained by being split across GPUs will not fit there.
        plan = type("P", (), {"strategy": "zero3"})()
        ok, reason = train.should_evaluate({"val_set_size": 20}, plan=plan)
        assert ok is False and "GPU" in reason

    def test_replicated_multi_gpu_runs_still_fit_one_gpu(self):
        plan = type("P", (), {"strategy": "ddp"})()
        assert train.should_evaluate({"val_set_size": 20}, plan=plan)[0] is True


class TestEvaluateAfterTraining:
    def test_reports_what_the_evaluation_wrote(self, tmp_path, monkeypatch):
        monkeypatch.setattr(train, "WORK_DIR", tmp_path)
        posted = []
        monkeypatch.setattr(train, "post_logs", lambda lines: None)
        monkeypatch.setattr(train, "_api", lambda *a, **k: posted.append(k.get("json")))

        def run(cmd, **kwargs):
            report = Path(cmd[cmd.index("--report") + 1])
            report.write_text(json.dumps({"examples": 2, "base": {}, "tuned": {}, "samples": []}))
            return type("R", (), {"returncode": 0})()

        train.evaluate_after_training(tmp_path / "cfg.yaml", tmp_path / "out", run=run)
        assert posted == [{"report": {"examples": 2, "base": {}, "tuned": {}, "samples": []}}]

    def test_a_failed_evaluation_never_fails_the_job(self, tmp_path, monkeypatch):
        logs = []
        monkeypatch.setattr(train, "post_logs", logs.extend)
        monkeypatch.setattr(train, "_api", lambda *a, **k: pytest.fail("nothing to report"))

        def boom(cmd, **kwargs):
            raise RuntimeError("CUDA out of memory")

        train.evaluate_after_training(tmp_path / "cfg.yaml", tmp_path / "out", run=boom)
        assert any("evaluation" in line.lower() and "skipped" in line.lower() for line in logs)
