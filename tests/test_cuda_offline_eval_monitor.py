import json

import pandas as pd

from drkernel.kernel.scripts.eval.monitor_cuda_offline_eval_results import validate_and_fix_run


def test_validate_and_fix_run_converts_mislabeled_jsonl_parquet(tmp_path):
    run_dir = tmp_path / "cuda-qwen35.run.test"
    step_dir = run_dir / "eval_results" / "step_0"
    step_dir.mkdir(parents=True)

    metrics = {
        "val/test_score/kernelbench_level2_validation": -0.1,
        "val/test_score/kernelbench_level2_validation_pass@1": 0.0,
        "val/test_score_extra/compilation_kernelbench_level2_validation": 0.0,
        "val/kernel/best_by_turn_3/correctness_rate": 0.0,
        "val/multiturn/num_turns/mean": 3.0,
    }
    (step_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (step_dir / "raw_responses.jsonl").write_text('{"input": "x", "outputs": ["y"]}\n', encoding="utf-8")

    records = [{"data_source": "kernelbench_level2_validation", "prompt": str(i), "solve_rate": 0.0} for i in range(100)]
    with (step_dir / "graded_results.parquet").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    (run_dir / "main.log").write_text(
        "\n".join(
            [
                "Qwen3.5-27B",
                "Key Eval Parameters",
                "KERNEL_BACKEND: cuda_agent",
                "REFERENCE_CACHE_ENABLE: true",
                "REWARD_TASK_TIMEOUT: 30",
                "NUM_WARMUP: 5",
                "NUM_PERF_TRIALS: 50",
                "PERF_TRIM_COUNT: 5",
                "VLLM_LANGUAGE_MODEL_ONLY: true",
            ]
        ),
        encoding="utf-8",
    )

    ok, details = validate_and_fix_run(run_dir, "Qwen3.5-27B", tmp_path / "monitor.log")

    assert ok
    assert details["graded_rows"] == 100
    assert (step_dir / "graded_results.jsonl").exists()
    converted = pd.read_parquet(step_dir / "graded_results.parquet")
    assert len(converted) == 100
