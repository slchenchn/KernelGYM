# Progress

#### Qwen3.5 27B CUDA RL smoke uses the hfsdp8/eager launcher and Qwen-compatible rollout path

##### Problem & Impact

- The Qwen3.5 CUDA smoke path had incorrectly reused the `mrs_pr_prs` launcher family; the intended baseline is `drkernel/kernel/scripts/rl/14b_coldstart_trloo_hfsdp8_pytorch_eager.sh`.
- The requested smoke settings for `/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.5-27B` required `.18`, TP=2, and `MAX_RESPONSE_LENGTH=12000`, but the earlier CUDA launch path could silently ignore TP overrides, import code from the older `KernelGYM-vllm018` workdir, miss the CUDA primary Hydra search path, or override the configured reward URL with an empty value.
- The Qwen3.5 vLLM path needed `language_model_only=True`; without it, this checkpoint family could take a non-language-model loading path.
- The shared `.18` stack initially did not support `model_type=qwen3_5`; after upgrading, Transformers 5 and vLLM 0.19 exposed additional API changes around auto-model imports, LoRA imports, and vLLM external-launcher worker GPU visibility.
- The `.18` run `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent/drkernel/logs/trloo-qwen35-27b-cuda-18.train.8XA800.reward.16x4090.run.20260421-083114` reached validation but failed inside vLLM input validation with `TypeError: '>' not supported between instances of 'str' and 'int'`.
- Local reproduction showed Qwen3.5 with Transformers 5 can return a `BatchEncoding` from `apply_chat_template(tokenize=True)`, and parquet prompts arrive as numpy object arrays; passing that object to `TokensPrompt` makes vLLM see string keys instead of integer token ids.
- After the token-id fix, run `20260421-085239` advanced to vLLM engine initialization but failed because colocated actor/FSDP weights left about `53.9 GiB` free per A800 while the inherited `ROLLOUT_GPU_MEMORY_UTIL=0.75` required `59.44 GiB` free.
- The follow-up `20260421-085933` run with `ROLLOUT_GPU_MEMORY_UTIL=0.60` passed the free-memory precheck but still failed during validation `wake_up(tags=["weights"])` with `CUDA Error: out of memory` in vLLM's `cumem_allocator`; lowering KV-cache utilization alone did not address the actor/vLLM weight remap peak.
- Run `20260421-094753` restored `gpu_memory_utilization=0.75` and reached initial validation, but exited before any train step completed: one vLLM worker first raised `RuntimeError: State error: sample_tokens() must be called after execute_model() returns None`, then a TP worker hit a 600s NCCL `_ALLGATHER_BASE` watchdog timeout and the trainer surfaced `EngineDeadError`.
- The async vLLM wrapper had left `actor_rollout_ref.rollout.max_num_seqs` at the config default `1024` and did not pass it to `AsyncEngineArgs`; validation therefore queued `200` requests per engine for `10k+12k` long generations, above the observed full-length KV concurrency envelope for this setup.
- Run `20260421-104450` confirmed `max_num_seqs=32` was applied, but still failed before any validation metrics or train step: rollout stayed at `servers=0/4 prompt_rows=0/800`, vLLM raised the same `sample_tokens()` / `execute_model()` state error at `10:57:12` and `11:01:30`, then a TP worker hit the 600s NCCL watchdog and the trainer surfaced `EngineDeadError`.
- The remaining issue was not CUDA reward failure; the custom external Ray executor was still using vLLM's base two-RPC `execute_model()` / `sample_tokens()` flow. Under vLLM 0.19 TP rollout, EngineCore could decide whether to call `sample_tokens()` from only the driver-rank result while another TP worker still held `execute_model_state`.

##### Resolution

- Replaced the incorrectly named CUDA/Qwen `mrs_pr_prs` launcher with `drkernel/kernel/scripts/rl/cuda_qwen3_5_27b_trloo_hfsdp8_pytorch_eager.sh`, based on the hfsdp8 pytorch-eager baseline while keeping CUDA neutral data, `cuda_kernel_trainer`, Qwen3.5-27B, TP=2, and `MAX_RESPONSE_LENGTH=12000`.
- Launched through `drkernel/kernel/scripts/rl/start_training.sh` on `.18` with explicit current-repo workdir, `--single-node`, `--skip-reward`, `ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=2`, `MAX_RESPONSE_LENGTH=12000`, and `KERNELGYM_SERVER_URL=http://192.168.16.39:8111`; earlier `.50/.51` attempts were stopped and superseded.
- Split the shared trainer config into primary wrappers around `kernel_trainer_core.yaml`, made the shared RL launcher pass `reward_model.server_url` only when non-empty, and added shared vLLM engine-kwargs handling so `cuda_kernel_trainer.yaml` can set `actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=true`.
- Upgraded the shared `.18` venv to `transformers==5.5.4` and `vllm==0.19.1`, then added compatibility fallbacks for Transformers 5 auto-model imports, vLLM 0.19 LoRA imports, and vLLM worker-side external-launcher marking.
- Added `drkernel/kernel/workers/rollout/vllm_rollout/chat_template_utils.py` so local rollout first normalizes chat messages and then extracts a plain `list[int]` from either legacy `list[int]` or Transformers 5 `BatchEncoding` outputs.
- Routed local vLLM, vLLM multi-iter, and OpenAI multi-iter rollout prompt-token generation through the new helper instead of passing raw chat-template outputs through.
- Initially lowered the Qwen3.5 CUDA launcher's default `ROLLOUT_GPU_MEMORY_UTIL` from the hfsdp8 baseline `0.75` to `0.60` to isolate the single-node colocated `.18` free-memory failure.
- Set the Qwen3.5 CUDA launcher default `ACTOR_PARAMETER_OFFLOAD=True` so actor FSDP shards are offloaded before vLLM remaps weights during rollout wake-up, then restored the launcher default `ROLLOUT_GPU_MEMORY_UTIL=0.75` once actor offload was active.
- Made the shared RL launcher pass `ROLLOUT_DISABLE_LOG_STATS`, `VLLM_LOGGING_LEVEL`, and `KERNELGYM_VLLM_STATS_LOG_INTERVAL` into the runtime config, and added a lightweight stats loop in the custom multi-turn vLLM wrapper so vLLM's `do_log_stats()` is called periodically instead of relying on OpenAI-server behavior that this rollout path does not use.
- Relaunched the smoke through `start_training.sh` on `.18` with the new hfsdp8/eager CUDA/Qwen launcher, the existing `.39` reward endpoint, and IPv4 transport settings.
- Made the async vLLM engine pass `max_num_seqs=rollout_config.max_num_seqs` into `AsyncEngineArgs`, added a shared `ROLLOUT_MAX_NUM_SEQS` launcher override, and set the Qwen3.5 CUDA smoke default to `32`, below the observed `37.46x` full-length KV concurrency for `22,240` tokens/request on TP=2 A800 rollout.
- Relaunched the smoke with the same model, data, TP=2, `MAX_RESPONSE_LENGTH=12000`, `gpu_memory_utilization=0.75`, actor parameter offload, vLLM INFO stats, and the new `ROLLOUT_MAX_NUM_SEQS=32` cap.
- Removed the dead duplicate executor definitions from `drkernel/kernel/workers/rollout/vllm_rollout/vllm_async_engine.py`, leaving one active `_get_model_runner_workers`, `ExternalRayDistributedExecutor`, and `AgentLoopOutput` definition in that module.
- Updated the active custom `ExternalRayDistributedExecutor` to mirror vLLM 0.19 Ray executor sampler semantics: sampled batches are deferred in `execute_model()`, then `sample_tokens()` calls worker-side `execute_model_ray` so every TP worker clears `execute_model_state` in one RPC wave.
- Added targeted unit coverage for the external executor's deferred sampled-batch path, all-worker `execute_model_ray` dispatch, and immediate unsampled-batch path.

##### Result & Current State

- The old `mrs_pr_prs` launcher path is no longer used for the Qwen3.5 CUDA smoke; the active Qwen script is `cuda_qwen3_5_27b_trloo_hfsdp8_pytorch_eager.sh`.
- Current shared `.18` venv versions are `transformers==5.5.4`, `vllm==0.19.1`, `torch==2.10.0`, and `numpy==2.2.6`; `pip check` still reports the pre-existing protobuf conflict and the `verl` metadata mismatch because `verl` declares `numpy<2.0.0`.
- Local validation against the neutral validation parquet now returns `list[int]` for the first Qwen3.5 prompt (`327` integer token ids), avoiding the vLLM `str/int` failure class seen in `083114`.
- Verification for this fix passes: `python -m py_compile` on patched rollout files, `bash -n` on the new and baseline launchers, targeted tests `tests/test_chat_template_utils.py tests/test_vllm_compat.py tests/test_vllm_engine_kwargs.py` with `11 passed` after adding coverage for async `max_num_seqs`, and the earlier full local suite with `84 passed`.
- The fixed path moved past the earlier `qwen3_5` model-type failure and vLLM `local_world_size (2) <= visible devices (1)` assertion; logs showed `Qwen3_5ForConditionalGeneration contains 27.36B parameters`, `MultiTurnAsyncvLLMEngine` startup, vLLM EngineCore Ray connection, and safetensors shard loading.
- Run `20260421-085239` confirmed the token-id fix progressed past the previous `083114` failure class, then stopped at the separate vLLM free-memory check described above.
- Run `20260421-085933` confirmed that `gpu_memory_utilization=0.60` was not sufficient by itself; the separate actor parameter offload change was needed for the single-node Qwen3.5 colocated topology.
- Run `20260421-093700` confirmed that after actor parameter offload, restoring `gpu_memory_utilization=0.75` passes vLLM initialization and raises observed `.18` GPU residency to about `66 GiB` per A800, but it was superseded before completion because the custom rollout wrapper still did not call vLLM `do_log_stats()` periodically.
- Run `20260421-094753` is stopped after the vLLM/NCCL failure described above; no validation metrics, retry result, or train step completed in that run.
- Run `20260421-104450` is stopped after confirming the max-seq cap was not enough by itself; no validation metrics, retry result, or train step completed in that run, and `.18` GPUs were empty after the crash.
- The executor fix is implemented locally but has not yet been validated by a new distributed smoke run. Current verification passes: `py_compile`, `git diff --check`, `tests/test_vllm_engine_kwargs.py` with `7 passed`, and the targeted chat-template / vLLM compatibility set with `14 passed`.

#### CUDA RL prompts now split backend-neutral data from CUDA templates

##### Problem & Impact

- The materialized RL and validation parquet files still stored the original first-turn Triton prompt in the data itself, so even the CUDA overlay only changed follow-up feedback turns while the first rollout turn started with Triton instructions and examples.
- This made the data backend-specific and prevented clean switching between Triton/CUDA prompt templates.

##### Resolution

- Added `drkernel/kernel/scripts/materialize_backend_neutral_data.py` to extract only the fenced PyTorch model architecture from the source prompt and write backend-neutral parquet files under the current repo's `drkernel/data`.
- Generated:
  - `drkernel/data/drkernel-rl-data-neutral/cuda_llm_rl_thinking_1025.parquet`
  - `drkernel/data/drkernel-validation-data-neutral/validation_data_thinking.parquet`
- Generated matching human-readable text copies for manual prompt inspection:
  - `drkernel/data/drkernel-rl-data-neutral/cuda_llm_rl_thinking_1025.txt`
  - `drkernel/data/drkernel-validation-data-neutral/validation_data_thinking.txt`
- Generated a deterministic 100-row sample of train prompts after CUDA first-turn formatting:
  - `drkernel/data/drkernel-rl-data-neutral/cuda_llm_rl_thinking_1025.formatted_sample100.txt`
- Generated current CUDA multi-template review dumps from real data:
  - `drkernel/data/drkernel-validation-data-neutral/validation_data_thinking.cuda_template_review_sample.txt`
  - `drkernel/data/drkernel-rl-data-neutral/cuda_llm_rl_thinking_1025.cuda_template_sample100.txt`
- Updated `drkernel/kernel/config/prompt_config/multi_turn_cuda_kernel.yaml` so the first turn puts CUDA instructions and the required `CUDA_KERNELS` / `APPLY_BINDINGS` / `MODEL_NEW` output format before the neutral PyTorch problem, then ends with `Let's think step by step.`.
- Updated the earlier 14B RL launcher path so `HYDRA_CONFIG_NAME` can select `cuda_kernel_trainer`, and made the then-active `14b_coldstart_trloo_mrs_pr_prs.sh` default to the CUDA config, neutral data, and `30s` CUDA reward timeout; the later Qwen3.5 smoke uses the hfsdp8/eager CUDA launcher recorded above.
- Compared against `/nfs/FM/lihongbin/CODE/KernelGYM`: that external CUDA path keeps the long CUDA first-turn prompt and CUDA skeleton examples materialized inside parquet `user` messages, while its `multi_turn_cuda_kernel.yaml` leaves `first_turn.template: null` and only controls feedback turns.

##### Result & Current State

- The generated neutral train parquet has `71996` rows, with `0` rows containing `Triton` and `0` rows containing CUDA section markers.
- The generated neutral validation parquet has `100` rows, with `0` rows containing `Triton` and `0` rows containing CUDA section markers.
- Some train rows still contain the literal word `cuda` inside the PyTorch problem code itself, for example `device="cuda"` in `get_inputs`; these are problem semantics, not prompt/template instructions.
- Follow-up code inspection found that only `data.system_prompt_config` creates a true `role=system` message, and the CUDA launcher path does not set it; `prompt_config_path` per-turn templates are user-message content, and the shared rollout helper now supports a Jinja `{{ problem }}` placeholder so the CUDA first-turn template controls where the neutral PyTorch problem appears.
- The text copies have the expected row separators: `71996` train rows and `100` validation rows; the formatted sample has `100` sampled train prompts with `100` CUDA output-format instructions and `100` neutral PyTorch model prompts.
- The current CUDA template review dumps cover real validation rows `0,1,2,7,42` across all three first-turn templates and both feedback templates, plus `100` deterministic train rows with random first-turn template selection; checks found no active `Triton` instruction residue and no unrendered `{{ problem }}` / `{{ feedback }}` placeholders. Case-insensitive `triton` appears only in dump metadata naming `migrate_from_triton.jinja`.
- Manual inspection of validation row 0 across `csl_cuda_agent`, `lhb_v3`, and `migrate_from_triton`, plus both tool-feedback templates and sampled train rows, confirmed that the CUDA section contract appears before the neutral PyTorch problem, the problem intro is not duplicated, and feedback turns do not restate the full first-turn constraints.
- `tests/test_backend_neutral_data.py` and the expanded `tests/test_prompt_templates.py` pass, and the broader targeted CUDA support regression set passes `70/70`.

#### CUDA reward comparison against the external KernelGYM repo found one local parameter-propagation gap

##### Problem & Impact

- A follow-up comparison against `/nfs/FM/lihongbin/CODE/KernelGYM` showed that this branch already carries the core CUDA reward algorithm pieces needed for RL, but the local workflow split path did not propagate `num_warmup` and `perf_trim_count` from an `EvaluationTask` into the derived reference/kernel subtasks.
- This affected the intended training timing protocol: `drkernel/kernel/config/kernel_trainer.yaml` sets `num_perf_trials=50`, `num_warmup=30`, and `perf_trim_count=5`, and `drkernel/run_cuda_reward_dir.py` uses the same `30/5/50` defaults, but pre-fix paired workflow children preserved only `num_perf_trials` and fell back to schema defaults `num_warmup=3`, `perf_trim_count=0`.
- Correctness, compilation, precheck, and decoy classification were not changed by this parameter propagation bug, but speedup/reward timing was noisier than the intended protocol because it used fewer warmup iterations and no trimmed mean.

##### Resolution

- Updated `kernelgym/workflow/kernelbench_helpers.py` so both `ReferenceTimingTask` and `KernelEvaluationTask` preserve `num_warmup` and `perf_trim_count`.
- Updated the cache-eviction fallback path in `kernelgym/workflow/kernelbench.py` so a late-created reference timing task also preserves timing controls, toolkit, backend adapter, and resources.
- Added rollout-side tool-feedback truncation using the existing `rollout.multi_turn.max_tool_response_length` and `tool_response_truncate_side` config in the vLLM, multi-iter vLLM, and OpenAI multi-iter rollout engines.
- Added regression coverage in `tests/test_cuda_agent_support.py` for paired-task propagation, the reference-cache fallback path, and bounded feedback truncation.
- Installed the project-pinned pytest tooling into the current Python environment after user approval so the CUDA-Agent test file can be run directly.

##### Result & Current State

- `tests/test_cuda_agent_support.py` now passes `37/37`.
- `git diff --check` and targeted `py_compile` checks pass after the fix.
- Remaining differences from the external repo are implementation/feature-mode differences rather than missing core reward algorithm paths: `tvm_ffi` compile mode, CUPTI/flashinfer timing, and a looser profiler-based CUDA detect helper are not currently ported into this branch.

#### Reward nodes are locked and the CUDA full-set reward sweep completed against the live 16-worker service

##### Problem & Impact

- The CUDA reward path had been repaired and smoke-tested, but there was still no full-set validation proving that the branch could score the entire stored result directory against live `39/40` reward workers.
- The reward infra also was not immediately launch-ready for this branch:
  - `.40` did not have the expected `/nfs/FM` mount on the host
  - the canonical reward startup path still defaulted to the old non-CUDA repo root
- Without stabilizing reward-host clocks and the branch-specific reward service first, a full-set sweep would have mixed algorithm validation with infra noise.

##### Resolution

- Re-locked both reward hosts so all `16` RTX 4090s on `.39` and `.40` now report `Graphics/SM = 2700 MHz` with persistence enabled and `400 W` power limit.
- Restored `.40` host access to `/nfs/FM`, then relaunched the reward stack with `REWARD_REPO_PATH=/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent` so the live service imports this branch instead of the older baseline repo.
- Re-verified the IPv4 reward API on `http://192.168.16.39:8111`, and confirmed `16` online workers split evenly as `8` on `reward-39` plus `8` on `reward-40`.
- Updated `drkernel/kernel/scripts/rl/infra_common.sh` so `start_reward.sh` now defaults `REWARD_REPO_PATH` to the current repo root instead of the older `KernelGYM-vllm018` path, removing the need for a manual branch-specific override on future restarts.
- Added a repo-local batch harness at `drkernel/run_cuda_reward_dir.py` for directory-wide CUDA reward evaluation, including:
  - local CUDA-Agent structural precheck classification so obvious malformed samples are scored as `precheck_fail` without consuming GPU workers
  - batched remote submission for the remaining samples through the repaired reward client
  - resumable `results.jsonl` and `summary.json` outputs
- Enabled `use_reference_cache=True` by default for future CUDA batch-harness remote tasks, with `--no-reference-cache` kept as an explicit opt-out for diagnostics.
- Ran an `.18` smoke batch over `8` representative samples from `/nfs/FM/gongoubo/cuda_kernel/parallel_drkernel_minimax_results`, which returned:
  - `2` local `precheck_fail`
  - `6` completed remote evaluations
  - `0` server failures
- Launched the full-set sweep from `.18` inside tmux session `cuda_reward_fullset_20260420` with output rooted at `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent/tmp/cuda_reward_fullset_20260420_1251`.

##### Result & Current State

- The branch now has a completed full-directory CUDA reward validation sweep instead of only single-sample checks and unit tests.
- The full-set run used the shared `.18` venv, the IPv4 reward endpoint on `.39`, and the active `16`-worker `39/40` reward pool.
- The active full-set tmux process was launched before the reference-cache default change, so it will not retroactively pick up `use_reference_cache=True`; future or restarted harness runs will use the reference cache unless `--no-reference-cache` is passed.
- A single-sample `.18` diagnostic using `result_0.json` showed the same UUID path dropping from `17.58s` wall time on cache miss to `12.03s` on cache hit, while candidate CUDA compile/load still dominated at about `11.6-12.7s`.
- A follow-up compile-path review found that the CUDA-Agent backend was compiling four translation units per sample, including an invariant `binding_registry.cpp`; this has been collapsed into header-only registry code so future CUDA-Agent cold builds no longer spend a separate compile step on that fixed scaffold file.
- CUDA-Agent compile work directories now prefer `/dev/shm` when enough tmpfs space is available, with `KERNELGYM_CUDA_AGENT_TMPDIR` as an explicit override and automatic fallback to the default temp directory when tmpfs is unavailable or too small.
- CUDA-specific timeout defaults now use `30s` in both the CUDA batch harness and the `cuda_kernel_trainer.yaml` overlay instead of inheriting the general `600s` default.
- The current offline result directory has `8920 / 8920` unique extracted candidate-code hashes and `8920 / 8920` unique reference-code hashes, so content-hash extension caching would not materially accelerate this specific full-set sweep except for retries or repeated RL samples.
- Final observable state:
  - final progress snapshot on `2026-04-21 10:52 +08:00` is `8920 / 8920` samples recorded in `results.jsonl`
  - final result mix is `7385` completed and `1535` failed
  - final summary reports `reward_nonzero=5380`, `compiled_true=7377`, `correct_true=4195`, `decoy_true=8`, `local_precheck_fail=91`, and `server_fail=1444`
  - dominant failure classes are kernel compilation failure (`1094`), `Kernel evaluation failed: 'NoneType' object has no attribute 'metadata'` (`221`), local/code precheck failure (`91`), and decoy detection (`8`)
  - the earlier batch-1 long-tail on `result_24.json` did not prevent completion of the full sweep
  - representative local failures are being classified as `Task failed: code pre-check error` with `reward=-0.5`

#### Failed reward outcomes now route by failure type instead of collapsing to one generic penalty

##### Problem & Impact

- The CUDA reward-path port had preserved the main `weighted` / `speedup` success formulas, but it had dropped the original reward client's failure-type routing.
- As a result, `precheck` failures, CUDA compilation failures, and unrelated generic task failures could all collapse to the same `penalty_score`, which changed the reward algorithm even when the underlying evaluation metadata was correct.

##### Resolution

- Restored `apply_precheck_fail_penalty` / `apply_compilation_fail_penalty` handling in `drkernel/kernel/rewards/reward_client.py`, including:
  - failure-message normalization into `code pre-check error` versus `kernel compilation error`
  - dedicated `precheck_fail` and `compilation_fail` penalty lookups
  - routing in `calculate_reward_like_kernel`, `calculate_reward_weighted`, and `calculate_reward_speedup`
- Reintroduced the corresponding config knobs and `precheck_fail` penalty field in both `drkernel/kernel/config/kernel_trainer.yaml` and `drkernel/kernel/config/kernel_grading.yaml`, and set the defaults on this branch to enabled so the restored logic is active instead of dormant.
- Expanded `tests/test_cuda_agent_support.py` with explicit regression coverage for:
  - precheck-failure reward routing
  - compilation-failure reward routing
  - generic timeout / fallback penalty routing
  - disabled-toggle fallback to generic penalty
- Manually verified representative CUDA-result samples from `/nfs/FM/gongoubo/cuda_kernel/parallel_drkernel_minimax_results` on `.18` using the repaired reward logic:
  - `result_1429.json`: missing `.cu` file, normalized to `code pre-check error`
  - `result_1759.json`: missing `binding_registry.h`, normalized to `code pre-check error`
  - `result_0.json`: compiled and correct, measured `1.94x` speedup, reward `1.0`
  - `result_10.json`: compiled but incorrect, reward `0.0`

##### Result & Current State

- This branch now uses failure-type-aware reward routing again instead of treating all failed tasks as one generic outcome.
- Direct regression verification now passes `33/33` cases in `tests/test_cuda_agent_support.py`.
- Manual sample checks confirm that representative CUDA-result files map to the expected reward branch after the fix.

#### This branch now has a repo-native CUDA-Agent RL path instead of only Triton-specific rollout and reward wiring

##### Problem & Impact

- The branch started from a Triton-focused baseline where reward submission, multi-turn extraction, and prompts still assumed a single Python `ModelNew` answer.
- Porting the external CUDA work verbatim would have cut across the current repo's cleaner `kernelgym` backend layering and still left rollout responses collapsing the CUDA triple-section format before reward evaluation.

##### Resolution

- Added a repo-native `kernelbench.cuda_agent` backend in `kernelgym`, with dispatcher selection, API enum support, and a cheap CUDA-Agent precheck in `kernelgym/toolkit/validation.py`.
- Kept the port aligned with this repo's backend design by reusing `KernelBenchBackendBase` for model construction and runtime execution instead of carrying over a separate session lifecycle.
- Added shared CUDA response extraction in `drkernel/kernel/utils/kernel_code.py`, then used it in both `drkernel/kernel/rewards/kernel_reward.py` and `drkernel/kernel/workers/agent/kernel_agent.py` so CUDA submissions survive intact from rollout through reward scoring.
- Switched `drkernel/kernel/rewards/reward_client.py` to honor per-task `kernel_backend` instead of always submitting `backend="triton"`.
- Added a CUDA-specific multi-turn prompt template and a small Hydra overlay config at `drkernel/kernel/config/cuda_kernel_trainer.yaml` so the CUDA path can be enabled explicitly without overwriting the existing Triton default.
- Added a repo-local single-sample harness at `drkernel/test_cuda_reward.py` and kept it lightweight by loading `drkernel/kernel/utils/kernel_code.py` directly, avoiding the unrelated `watchdog` dependency pulled in by `drkernel.kernel.utils.__init__` on the shared eval venv.
- Ran a real `.18` A800 smoke against `/nfs/FM/gongoubo/cuda_kernel/datasets/0`, which exposed a Triton-only profiling assumption in `kernelgym/toolkit/kernelbench/pipeline.py`; fixed the performance path so Triton coverage and decoy detection are skipped when `enable_triton_detection=False`.
- Wired backend-native CUDA profiling hints from `kernelgym/backend/kernelbench/cuda_agent_backend.py` into the evaluation pipeline, then taught `kernelgym/toolkit/kernelbench/pipeline.py` and `kernelgym/toolkit/kernelbench/profiling.py` to compute coverage from CUDA `__global__` kernel names instead of only from Triton hook matches.
- Restored the `detect_decoy_kernel` control path end to end by adding it to the server/task schemas and toolkit evaluation flags, while keeping coverage computation independent from the decoy penalty so CUDA and Triton both still emit coverage metrics when profiling succeeds.
- Normalized reward-side decoy propagation by writing both `decoy_kernel` and `is_decoy_kernel` into `reward_extra_info`, and updated trainer/metric consumers to read either key.
- Moved the profiler CUDA self-test outside the active profiling window so backend coverage denominators are no longer polluted by the profiler bootstrap op itself.

##### Result & Current State

- The repository can now route `backend="cuda_agent"` requests through `kernelgym` and preserve the expected `CUDA_KERNELS` / `APPLY_BINDINGS` / `MODEL_NEW` answer format end to end.
- The default trainer and grading configs still default to `kernel_backend: "triton"`; the explicit CUDA entrypoint is `drkernel/kernel/config/cuda_kernel_trainer.yaml`.
- Verification now includes a full GPU compile-and-run smoke on `root@192.168.16.18:20629` using the shared `vllm0180` venv and sample dataset `0`.
- Current observed `.18` sample result for the CUDA path: compiled successfully, passed `5/5` correctness trials, mean kernel runtime `0.0201 ms` vs reference `0.0387 ms`, measured speedup `1.93x`, and no decoy marking.
- The CUDA reward path now also emits nonzero coverage metadata on `.18`, and the reward-side coverage helper resolves that sample to `num_custom_kernel=1`, `num_total_kernels=1`, and `time_coverage=1.0` instead of silently falling back to zero.
- A `.18` negative-case smoke where `ModelNew` ignored the compiled extension now returns `decoy_kernel=True` with `num_custom_kernels=0` and profiled only ATen kernels, confirming the backend-native CUDA decoy path is active instead of Triton-only.

#### CUDA reward-path regression coverage now includes paper-aligned decoy and lazy-optimization cases

##### Problem & Impact

- The first CUDA unit-test pass primarily covered extraction, schema wiring, coverage-field propagation, and one decoy branch, but it still left the paper’s two core failure modes under-tested at the training-signal layer.
- Without explicit regression coverage for profiling-based rewards and coverage-based rejection sampling, future changes could preserve basic execution while silently weakening the protections against reward hacking and lazy optimization.

##### Resolution

- Moved CUDA test snippets out of inline strings and into dedicated fixtures under `tests/fixtures/cuda_agent_support/`, then rewired `tests/test_cuda_agent_support.py` to load them from disk.
- Expanded the CUDA test suite to cover decoy-penalty precedence in `calculate_reward_weighted`, paper-style lazy-vs-better-fusion reward shaping using low (`0.00014`) versus high (`0.8615`) `time_coverage`, and coverage-based rejection sampling for both per-turn and geometric aggregation.
- Added deterministic PRS checks with hard thresholds so the suite now verifies:
  - low-coverage correct samples are masked while high-coverage correct samples survive
  - incorrect samples are not filtered by coverage RS
  - the speedup OR-condition can preserve a low-coverage sample

##### Result & Current State

- The CUDA reward-path regression suite now covers both reward-hacking rejection and lazy-optimization mitigation instead of only backend execution mechanics.
- Current direct test sweep for `tests/test_cuda_agent_support.py` passes `29/29`, and `git diff --check` passes.
- The suite now provides explicit regression protection for `PR` and `PRS` semantics in addition to the earlier `.18` runtime smokes.

#### Early eval results for run `20260419-131619` are blocked by reward-service outage on `39/40`

##### Problem & Impact

- The post-fix `14B` run produced checkpoints `10/20/30`, but the early eval outputs were ambiguous and `step_20` collapsed to zero metrics.
- Without checking the reward path itself, the team could easily misread an infrastructure outage as a model-quality regression.

##### Resolution

- Recomputed the available checkpoint set from on-disk state and ran dedicated `.18` eval for `10/20/30`.
- Read the per-turn eval artifacts and confirmed repeated reward connectivity failures (`Connection refused` and `No route to host`).
- Checked `192.168.16.39` and `192.168.16.40` directly and confirmed both reward containers were down after host-level reboot; neither restarted automatically because they were configured with `restart=no`.
- Traced the reboot trigger and the underlying GPU fault signature:
  - both hosts entered a clean `systemd` reboot sequence immediately after `root` SSH login from `192.168.120.100`
  - previous-boot NVIDIA logs show real `Xid 109` (`CTX SWITCH TIMEOUT`) events on multiple GPUs across both hosts
  - those `Xid 109` events are surrounded by large volumes of `Xid 13` / `Xid 31` / occasional `Xid 43` from `python3`
  - the detailed fault strings are dominated by `MMU Fault`, `MMU NACK Errors`, `Out Of Range Address`, and `Invalid Address Space`

##### Result & Current State

- Early eval outputs for run `20260419-131619` should be treated as contaminated by reward outage rather than as clean model-quality signals.
- Reward recovery on `39/40` is the gating item for trustworthy fresh-run eval, not `.18` GPU availability.
- Current evidence points to reward-workload GPU faulting rather than one isolated bad card:
  - `.39` logged `Xid 109` on `GPU 7` (`PCI d6:00`) and `GPU 6` (`PCI d5:00`)
  - `.40` logged `Xid 109` on `GPU 1/3/4/6/7` (`PCI 52:00, 57:00, ce:00, d5:00, d6:00`)
  - the same signature has recurred across earlier `kern.log` history on both hosts
  - this is more consistent with generated-kernel evaluation provoking illegal-address / MMU-fault chains that escalate into context-switch timeout than with a single physically defective GPU

#### Historical `20260409-092519` checkpoint coverage now uses per-step `metrics.json` as the source of truth

##### Problem & Impact

- `eval_results/summary.txt` had drifted from the actual checkpoint/eval state for the historical `14B` run, so the remaining backlog could be misidentified.
- `step_260` also had a stale partial eval directory without a valid `metrics.json`.

##### Resolution

- Recomputed the remaining set from the checkpoint tree and the presence of `eval_results/step_*/metrics.json` instead of trusting `summary.txt`.
- Cleared the remaining backlog on `.18` for checkpoints `260/280/300`.
- The final summary rewrite hit `Disk quota exceeded`, so the textual summary file was left stale even though the per-step artifacts were complete.

##### Result & Current State

- The historical run no longer has missing checkpoint evals on disk.
- `metrics.json` under each step is the authoritative record; `eval_results/summary.txt` is stale for `260/280/300`.

#### The post-fix `14B` experiment was restarted as a fresh `50/51` A800 run with checkpoint eval isolated to `.18`

##### Problem & Impact

- The `time_coverage` semantic fix created a new experiment boundary, but earlier restarts still reused the historical run directory.
- Training and checkpoint eval had also been bouncing across different node layouts, which mixed runtime recovery with experiment changes.

##### Resolution

- Relaunched without `RUN_LOG_DIR` so `start_training.sh` created a fresh run directory instead of appending to historical logs.
- Standardized the operating split on `192.168.16.50/51` for training and `.18` for checkpoint eval.
- Kept validation disabled (`VAL_BEFORE_TRAIN=False`, `TEST_FREQ=0`) and retained `REWARD_TASK_TIMEOUT=30` for the new run.
- Revalidated the launch environment inside the training containers, including Python selection, virtualenv selection, and IB visibility.

##### Result & Current State

- The active post-fix experiment is `trloo-14b-hfsdp8-pytorch-eager.train.16xA800.reward.16x4090.run.20260419-131619`.
- The current `14B` A800 run is a fresh experiment, not a resume from the historical pre-fix tree.
- Training and checkpoint eval now use a stable node split instead of sharing the same machines.

#### `time_coverage` now uses CUDA-only runtime semantics

##### Problem & Impact

- `time_coverage` had been computed from matched custom CUDA time divided by CUDA time plus CPU profiler time.
- That denominator diluted coverage and could push PRS to reject otherwise valid samples.

##### Resolution

- Patched the producer in `kernelgym/toolkit/kernelbench/profiling.py` so `total_kernel_run_time_in_profiling_us` now means CUDA-only runtime.
- Updated downstream consumers in the pipeline and reward paths to use the CUDA-only denominator consistently.
- Added diagnostic fields for both CUDA-only total time and CPU+CUDA total time, then validated the change with `py_compile` and a synthetic ratio check.

##### Result & Current State

- The repository now computes `time_coverage` from CUDA-only runtime.
- The old CPU+CUDA denominator is retained only as a diagnostic field for comparison and debugging.

#### Empty post-adv batches no longer crash the `14B` trainer

##### Problem & Impact

- At `step 221`, post-advantage filtering removed all examples from the batch, but training still entered `update_actor`.
- That produced a `DataProto` chunking failure (`split_size must be a positive integer, but got 0`) and stopped training despite healthy rollout and reward infrastructure.

##### Resolution

- Patched `drkernel/kernel/kernel_trainer.py` so the trainer logs and skips actor/critic update when post-adv filtering leaves an empty batch.
- Rechecked the failure in the run logs and syntax-validated the patch before reuse.

##### Result & Current State

- The empty-batch guard is recorded in commit `9da31c2`.
- Later low-survivor steps progressed instead of crashing, so zero-row post-adv batches are no longer a known trainer stop condition.

#### Checkpoint eval was standardized around `eval_results/step_*`, auto-discovery, and GPU merge

##### Problem & Impact

- Checkpoint eval outputs and helper behavior had drifted across runs, which made resume and backfill eval dependent on hand-maintained step lists.
- The helper scripts also carried a broken summary formatter and multiple merge-path assumptions.

##### Resolution

- Standardized eval outputs under `eval_results/step_*`.
- Reduced merge handling to a single GPU-merge path.
- Made `EVAL_STEPS` optional so the helper can auto-discover missing checkpoints by scanning `CKPT_BASE/global_step_*` and checking for `metrics.json`.
- Fixed the broken summary-formatting path so the helper no longer depends on shell interpolation that produced invalid output.

##### Result & Current State

- Checkpoint backfill can now start from a checkpoint root and infer missing evals from artifact state instead of operator-maintained step lists.
- The helper behavior is aligned around one stable output layout and one merge strategy.

#### Checkpoint eval now supports split-disk H20 storage and isolated local-Ray execution

##### Problem & Impact

- H20 checkpoint eval had assumed one node could see a complete local checkpoint tree, a shared `eval_results` tree, current repo-root validation paths, and container-specific network settings.
- Those assumptions broke head-only eval on split-storage nodes and could also cause eval to attach to the live training Ray cluster by mistake.

##### Resolution

- Reworked `merge_and_eval_checkpoints.sh` so it can stage missing shards and metadata into a local merge dir, then sync worker-side `eval_results` back to the head-visible tree.
- Updated `main_grading.py` to honor `RAY_ADDRESS`, and made checkpoint eval default to `EVAL_RAY_ADDRESS=local` while allowing `EVAL_SOCKET_IFNAME`, `NNODES`, and `N_GPUS_PER_NODE` overrides.
- Added a validation-data remap helper in `grading_common.sh` so stale old-repo paths can be redirected into the current worktree when the local file exists.

##### Result & Current State

- The eval tooling now supports both shared-disk and split-disk storage layouts and can run under an isolated local-Ray session.
- One limitation remains explicit: `global_step_10` cannot be re-evaluated unless the missing worker-side shards are restored.

#### Reward long-tail diagnosis is split between incident recovery and structural synchronous-HTTP risk

##### Problem & Impact

- Extreme rollout long tails initially looked like generic networking noise, but one A800 incident also involved host-level NVIDIA/UVM failure on `16.40`.
- Treating every slowdown as one class of failure made recovery slower and hid the design risk in the reward path itself.

##### Resolution

- Correlated training logs (`main.log`, `trainer.log`, `reward.log`, `vllm.log`) with reward-host diagnostics.
- Separated the immediate A800 incident root cause from the structural effect of synchronous `/evaluate`, where reward-side stalls hold client-side requests open for the full HTTP duration.
- Wrote the incident analysis into the reward long-tail handoff instead of scattering the conclusions across restart notes.

##### Result & Current State

- The earlier A800 incident was recovered operationally at the time, but the durable lesson is architectural: reward-host health and synchronous `/evaluate` coupling must be analyzed separately.
- That split remains the right framework for later outages, including the current reboot-driven reward loss on `39/40`.

#### The `8B` H20 training thread was reframed as three distinct failure classes

##### Problem & Impact

- After fixing the early oversampling bottleneck, later H20 resumes still failed, but the symptoms did not share one root cause.
- Treating them as a single thread obscured which mitigation belonged to which failure.

##### Resolution

- Raised prompt oversampling so the run could clear the original low-batch retry loop.
- Separated the later failures into reward relay loss, async vLLM wake-up OOM, and actor-side OOM caused by foreign GPU jobs on the worker slice.
- Tied the later slow-step behavior back to reward-backed rollout long-tail rather than actor update or `old_log_prob`.

##### Result & Current State

- The H20 thread is no longer treated as one umbrella instability report.
- Later resumes and handoffs should reference the specific failure class they are addressing.

#### Training, eval, and working docs now follow canonical repo-local workflows

##### Problem & Impact

- Launch, stop, status checking, and plotting had drifted into ad hoc shell history.
- Working docs and handoffs were also accumulating as patch notes, which made active investigations harder to read.

##### Resolution

- Added repo-local skills for `start-training`, `stop-training`, and `check-training-status`, and kept orchestration anchored in repo scripts such as `drkernel/kernel/scripts/rl/start_training.sh`.
- Moved local cluster-default selection into untracked `.infra_profile.local.sh` rather than tracked infra defaults.
- Reorganized `handoffs/` into `completed/` and `in_progress/`, and kept stable repo guidance in harness files instead of transient run notes.

##### Result & Current State

- Training launch, stop, and status checks now follow canonical repo-local paths.
- Handoffs are organized around active investigations rather than incremental patch notes.
- `PROGRESS.md` is intended to remain a decision log, not a live-ops transcript.

#### Qwen3.5 CUDA model test moved from trainer val-only to offline eval_results

##### Problem & Impact

- `openai_async_engine_multi_iter.py` still carried an obsolete first executor group after the vLLM 0.19 external-executor fix.
- Keeping two `ExternalRayDistributedExecutor` definitions in one file made the active code path ambiguous and risked future fixes landing in the wrong block.
- The stop-training helper for the `a800` profile stopped the requested `.18` head Ray processes, then returned non-zero while trying the legacy `.24` worker that is not part of the current single-node smoke run.
- After run `114345` proved the external-executor fix reached vLLM validation rollout, the requested model test was still misinterpreted once as trainer `VAL_ONLY=True`; the user clarified that the desired artifact is the historical checkpoint-eval shape under `eval_results/step_*`, not a trainer val-only run.
- The existing grading scripts defaulted to Triton prompt/backend settings, did not expose CUDA prompt/backend, reference-cache, `language_model_only`, or `max_num_seqs` overrides, and computed `max_num_batched_tokens` before task-specific scripts changed prompt/response lengths.
- The first tmux launch of the new eval command had a shell heredoc quoting error and exited before creating a log. The next launch exposed a script variable-order bug: `source grading_common.sh` populated defaults first, so the CUDA script's `${VAR:-default}` assignments preserved `sync`, TP=1, 4 samples, 4096 response, and 600s timeout. That incorrect run was stopped before useful output was produced.
- The successful Qwen3.5 eval launch still wrote its primary live stdout to `/tmp`, which made the run directory incomplete for later inspection.
- The completed Qwen3.5 eval produced a readable `metrics.json`, but `main_grading.py` wrote JSONL content to `graded_results.parquet`, which would break downstream readers expecting real parquet.
- A later accidental Qwen3.5 relaunch reached vLLM `sleep(level=1)` during standalone manager initialization and failed with `CUDA error: invalid argument`; this did not invalidate the first completed metrics, but it made the log end with a fatal-looking cleanup/relaunch error.
- The first Qwen3-14B restart after this diagnosis also showed that wrapping the eval script with an outer `tee` duplicates log lines because the script already owns run-dir self-logging.
- Manual inspection after both completed evals showed the earlier validation was only artifact-level, not enough content-level verification. The recorded `pass@1=0.0` is internally consistent because all `2400` turns per model family report `compilation=False`, `correctness=False`, and `success=False`, but the result should not be treated as final until prompt rendering is corrected.
- The same inspection found a multi-turn CUDA prompt issue introduced by backend-neutral data materialization: first-turn prompts carry the CUDA instruction before the problem, but later-turn prompts rendered the initial problem as neutral history and relied only on the tool-feedback template for CUDA constraints.
- The first two prompt-fix rerun attempts failed before using GPUs: one launched without the shared `.18` venv and hit `ModuleNotFoundError: No module named 'hydra'`; the next launched from repo root and Hydra could not resolve the relative `ppo_trainer` config path.
- The first Qwen3-14B TP=1 smoke launch found a CUDA reward infra issue: reward containers mount `/dev/shm` with `noexec`, so CUDA agent extensions compiled under `/dev/shm/kernelgym_cuda_agent_*` failed to load with `failed to map segment from shared object`.
- A 5-row Qwen3-14B TP=1 smoke after the reward fix completed, but manual content inspection found the current first-turn CUDA templates still forbid extra text outside the three sections while Qwen3 emits `<think>...</think>` before the sections; this conflicts with think-enabled Qwen3 generation and contributed to missing-section / prompt-overlong failures.

##### Resolution

- Removed the obsolete first executor group from `openai_async_engine_multi_iter.py`; the file now has one `ExternalRayDistributedExecutor`, one `_get_model_runner_workers`, and one `AgentLoopOutput`.
- Revalidated the patched vLLM files with `py_compile` and the targeted chat-template / vLLM compatibility tests.
- Relaunched the CUDA Qwen3.5 27B smoke test through `start_training.sh` with `--profile a800`, `--single-node`, `--skip-reward`, and the current-repo CUDA launcher.
- Treated the `.24` stop-helper failure as a profile/tooling mismatch for this single-node run; no persistent config workaround was added.
- Stopped the `114345` run before it could enter PPO training after the user clarified that the requested run was a model test, not training.
- Changed the CUDA Qwen3.5 launcher default to `ROLLOUT_MAX_NUM_SEQS=512`.
- Added CUDA-aware offline grading support to `drkernel/kernel/scripts/eval/grading_common.sh`: CUDA kernel backend, CUDA prompt config, reference-cache overrides, vLLM `language_model_only`, rollout `max_num_seqs`, rollout log-stat controls, optional max model length, and delayed `max_num_batched_tokens` computation after script overrides.
- Added `drkernel/kernel/scripts/eval/cuda_qwen3_5_27b_maxturns3_temp1_0_w5t50trim.sh`, which writes checkpoint-eval-compatible outputs to `.../eval_results/step_0` for `/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.5-27B` without entering trainer/PPO.
- Fixed the CUDA eval script ordering so task-specific defaults are set before sourcing `grading_common.sh`, then relaunched the offline eval under tmux on `.18`.
- Added run-dir self-logging to the CUDA eval launcher and mirrored the active Qwen3.5 `/tmp` log into its run directory as `main.log`.
- Added `drkernel/kernel/scripts/eval/cuda_qwen3_14b_maxturns3_temp1_0_w5t50trim.sh` and started a `.18` tmux watcher that waits for the active Qwen3.5 eval to finish, then launches `/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3-14B` offline eval under the same CUDA grading settings.
- Added an explicit `Key Eval Parameters` block to the shared offline grading wrapper so model/data paths, prompt settings, sampling, rollout, reward, timeout, decoy, coverage, and reference-cache settings are written into each eval `main.log` before `kernel.main_grading` starts.
- Changed `main_grading.py` so `data.output_path=*.parquet` writes real parquet and non-parquet paths retain JSONL behavior.
- Converted the completed Qwen3.5 `graded_results.parquet` artifact into real parquet with `100` rows, preserving the original JSONL content as `graded_results.jsonl`.
- Made standalone-vLLM `sleep(level=1)` non-fatal for offline grading only, while leaving the training manager path strict; this prevents a vLLM sleep utility failure from discarding completed generation in a model-only eval topology.
- Added runtime environment logging (`which python`, `sys.executable`, `VIRTUAL_ENV`) to the CUDA eval launchers and relaunched Qwen3-14B without an outer tee so the active run has a clean single-copy `main.log`.
- Added `drkernel/kernel/scripts/eval/monitor_cuda_offline_eval_results.py` and launched it in tmux on `.18`; it validates Qwen3.5 and Qwen3-14B artifacts every `600s`, converts mislabeled JSONL artifacts if needed, and restarts Qwen3-14B if it exits without complete results.
- Added `tests/test_cuda_offline_eval_monitor.py` to cover the JSONL-to-parquet repair path that caught the completed Qwen3.5 artifact issue.
- Manually inspected `graded_results_conversations.jsonl` and `eval_outputs/problem_*_sample_*/turn_*_{eval,kernel,state}.json` for Qwen3.5 and Qwen3-14B, including precheck, compilation, timeout, `NoneType.metadata`, and missing-`ModelNew` examples.
- Added prompt rendering support that reapplies the first-turn CUDA instruction to the initial user problem when constructing later-turn prompts, without mutating the stored backend-neutral request state.
- Simplified CUDA tool-feedback templates so later turns focus on the server feedback and requested full three-section answer instead of restating the full first-turn CUDA constraints.
- Extended `tests/test_prompt_templates.py` to assert the actual CUDA first-turn and tool-feedback templates as complete strings, including the later-turn path that carries both initial instruction and feedback.
- Relaunched the corrected eval sequence from `.18` tmux with the shared venv activated and working directory set to `drkernel/`, so Hydra relative config search paths and package imports match the earlier successful offline eval runs.
- Organized CUDA first-turn and tool-response prompts as standalone `.jinja` files under `drkernel/kernel/config/prompt_config/cuda_templates`, while data stays prompt-neutral and `multi_turn_cuda_kernel.yaml` only points to template directories.
- Current first-turn directory contains `csl_cuda_agent.jinja`, `lhb_v3.jinja`, and `migrate_from_triton.jinja`; current tool-response directory contains `default.jinja` and `short.jinja`.
- Converted CUDA prompt rendering to Jinja `{{ problem }}` / `{{ feedback }}` in rollout, later-turn feedback wrapping, and formatted prompt materialization; old Triton-style `{problem}` / `{feedback}` placeholders are intentionally not treated as template syntax.
- Added directory-level template loading: `template_dir` / `template_dirs` load all template files from the given directory or directories, then rollout randomly chooses one concrete template at use time.
- Cleaned the CUDA prompt contract after review: removed directory-only wording, removed the invalid C++ prose inside the binding code fence, forbade inline CUDA compilation from `MODEL_NEW`, and replaced over-broad "no torch" wording with boundary-vs-compute rules that match the actual backend.
- Removed the extra first-turn problem headings such as `reference pytorch code:` / `Reference problem:` because backend-neutral prompts already start with `You are given the following PyTorch model:`.
- Added `noexec` mount detection to the CUDA agent backend work-dir selection so `/dev/shm` is used only when executable; otherwise the backend falls back to the default tempfile location. Covered this with targeted CUDA agent support tests and restarted the `.39/.40` reward stack so workers import the fixed backend.
- Created a 5-row neutral validation sample and launched Qwen3-14B on `.18` with TP=1, `N_SAMPLES=1`, `BATCH_SIZE=5`, `MAX_RESPONSE_LENGTH=12000`, `ROLLOUT_MAX_NUM_SEQS=8`, CUDA backend, reference cache enabled, and reward task timeout `30s`.
- Wrote a manual review dump for the TP=1 smoke at `drkernel/logs/cuda-qwen3-14b-tp1-smoke5-temp1.0-w5t50trim.run.20260422-032241/eval_results/step_0/manual_review_summary.md`.
- Changed multi-turn assistant-history handling so rollout preserves tokenizer-native assistant message fields instead of merging reasoning back into plain `content`: `chat_template_utils.py` now keeps full message dicts plus `reasoning_content`, and both `vllm_async_engine.py` and `openai_async_engine_multi_iter.py` now append assistant history through that path.
- Extended `tests/test_chat_template_utils.py` to verify actual Qwen3 tokenizer behavior for both `reasoning_content` and inline `<think>...</think>` history stripping.
- Ran a 1-row TP=1 regression smoke with `DATAPROTO_PATH` enabled at `drkernel/logs/cuda-qwen3-14b-tp1-smoke1-dataproto-temp1.0-w5t50trim.run.20260422-045511/eval_results/step_0` so the exact per-turn prompt token tensors were saved for manual inspection.
- Added prompt-feedback compaction in `event_logging.py` and wired it into all three multi-turn rollout entry points (`vllm_async_engine.py`, `vllm_async_engine_multi_iter.py`, `openai_async_engine_multi_iter.py`): later turns now receive a compact payload with short `error_message`, selected reward metrics, and a small metadata subset instead of the full `env_state` JSON dump.
- Added targeted tests for prompt-feedback compaction in `tests/test_cuda_agent_support.py`, covering precheck extraction, compiler-error-line extraction, and runtime traceback extraction.
- Stopped the first 16-row Qwen3-14B TP=1 smoke after the tokenizer-history fix because it had been launched before feedback compaction and was still evaluating with oversized later-turn error payloads; relaunched the same 16-row smoke after the compaction patch under a fresh run directory.

##### Result & Current State

- The stopped `114345` run proved that the external-executor fix reaches vLLM validation rollout: four engines initialized, KV cache size was `256,368` tokens, maximum concurrency was `37.46x` for `22,240` tokens/request, and vLLM throughput logs reached `Running: 31-32 reqs` without the old `sample_tokens()` state error.
- The incorrect trainer `VAL_ONLY=True` run `115748` was stopped and is no longer the active model test.
- Qwen3.5 completed with validated artifacts under `drkernel/logs/cuda-qwen35-27b-maxturns3-temp1.0-w5t50trim.run.20260421-201330/eval_results/step_0`: `metrics.json`, `raw_responses.jsonl`, `graded_results_conversations.jsonl`, real `graded_results.parquet`, and original `graded_results.jsonl` backup.
- The Qwen3.5 supervisor validation reports `100` graded rows, `val/test_score/kernelbench_level2_validation=-0.33458333333333334`, `pass@1=0.0`, compilation score `0.0`, best-by-turn-3 correctness `0.0`, and mean turns `3.0`; the zero score is model output quality, not an artifact-read failure.
- The active Qwen3-14B eval run is `drkernel/logs/cuda-qwen3-14b-maxturns3-temp1.0-w5t50trim.run.20260421-211115`, tmux session `cuda-qwen3-14b-eval-step0-18` on `.18`.
- The Qwen3-14B run log confirms the shared venv, current-repo neutral validation data, `standalone_vllm`, `Qwen3-14B`, `N_SAMPLES=8`, `BATCH_SIZE=128`, `MAX_PROMPT_LENGTH=10240`, `MAX_RESPONSE_LENGTH=12000`, TP=2, `max_num_batched_tokens=23240`, `max_num_seqs=512`, `language_model_only=true`, CUDA prompt config, `kernel_backend=cuda_agent`, `reference_cache=true`, `task_timeout=30`, `num_warmup=5`, `perf_trim_count=5`, and `num_perf_trials=50`.
- Qwen3-14B completed with validated artifacts under `drkernel/logs/cuda-qwen3-14b-maxturns3-temp1.0-w5t50trim.run.20260421-211115/eval_results/step_0`: `metrics.json`, `raw_responses.jsonl`, `graded_results_conversations.jsonl`, and real `graded_results.parquet`.
- The Qwen3-14B supervisor validation reports `100` graded rows, `val/test_score/kernelbench_level2_validation=-0.27166666666666667`, `pass@1=0.0`, compilation score `0.0`, best-by-turn-3 correctness `0.0`, and mean turns `3.0`; the run ended with no active tmux session, no remaining `main_grading` process, and `.18` GPUs empty.
- The overnight supervisor `cuda-eval-overnight-supervisor-18` exited after writing `all_eval_results_valid` at `2026-04-21 13:52:57 UTC`; no Qwen3-14B restart was needed.
- Content-level inspection found Qwen3.5 failures dominated by precheck, compilation, `NoneType.metadata`, and timeout classes; Qwen3-14B showed the same classes, with Qwen3-specific `<think>` output in all sampled turns. Several sampled kernels contained concrete invalid patterns such as `load_inline` in `MODEL_NEW`, missing `REGISTER_BINDING`, missing or undefined `cuda_extension`, missing `ModelNew`, and generated references to Triton only inside model text rather than the active prompt template.
- The completed Qwen3.5 and Qwen3-14B metrics are now classified as pre prompt-carry-fix evals. They prove the artifact pipeline can produce readable CUDA eval results, but corrected pass@1 should be obtained by rerunning after the prompt fix.
- Local verification for the prompt fix passed: `python -m pytest -q tests/test_prompt_templates.py tests/test_chat_template_utils.py tests/test_cuda_offline_eval_monitor.py` reported `24 passed, 1 skipped`, the touched rollout files passed `py_compile`, and `git diff --check` was clean.
- Local verification for the Jinja CUDA template conversion and prompt-contract cleanup passed: `PYTHONPATH=$PWD:$PWD/drkernel pytest -q tests/test_prompt_templates.py tests/test_backend_neutral_data.py` reported `42 passed`; `PYTHONPATH=$PWD:$PWD/drkernel pytest -q tests/test_cuda_agent_support.py -k 'extract_kernel_submission or section_extractor or precheck_cuda_agent_submission'` reported `10 passed, 31 deselected` with only existing Pydantic deprecation warnings; the touched prompt/rendering files passed `py_compile`; structured real-data prompt validation reported `rendered_prompt_intro_validation_ok`; `git diff --check` on the prompt/doc/test files was clean.
- Corrected sequential rerun is active on `.18` in tmux `cuda-eval-promptfix-qwen35-then-qwen14-18`. The active Qwen3.5 run is `drkernel/logs/cuda-qwen35-27b-maxturns3-temp1.0-w5t50trim.run.20260422-002143`; it passed config validation, initialized four vLLM engines with TP=2 and `max_seq_len=22240`, entered generation for `800` rows, and showed about `63.3 GiB` used per GPU at the latest check. Qwen3-14B is queued behind it in the same tmux command.
- The Qwen3-14B TP=1 smoke completed under `drkernel/logs/cuda-qwen3-14b-tp1-smoke5-temp1.0-w5t50trim.run.20260422-032241/eval_results/step_0` with readable `metrics.json`, `raw_responses.jsonl`, `graded_results_conversations.jsonl`, real `graded_results.parquet`, and the manual review summary.
- The TP=1 smoke confirms the deployment shape works: one A800 loaded `Qwen3ForCausalLM`, `tensor_parallel_size=1`, `max_seq_len=22240`, model load used about `27.52 GiB`, generation throughput reached roughly `200-245 tok/s`, and `.18` GPUs were empty after completion.
- The TP=1 smoke result is not good enough to scale: `val/test_score/kernelbench_level2_validation=-0.26666666666666666`, final correctness `0.0`, turn-1 compilation `0.2`, later-turn compilation `0.0`, one sample hit `Prompt Overlong`, and sampled failures included missing `ModelNew`, precheck syntax errors, `COMPILATION_ERROR`, and `NoneType.metadata` runtime feedback.
- The earlier `/dev/shm` shared-object mapping failure did not recur after reward restart; current remaining blockers are prompt/output-contract quality and the unclear `NoneType.metadata` runtime feedback, not TP=1 model deployment.
- The tokenizer-native reasoning-history fix is now validated on a real eval path: decoding `test_batch.dataproto` from `drkernel/logs/cuda-qwen3-14b-tp1-smoke1-dataproto-temp1.0-w5t50trim.run.20260422-045511/eval_results/step_0` shows turn-2 and turn-3 prompts contain neither `<think>` nor `</think>`, while still carrying the prior assistant final answer and the server feedback.
- This removes one previously suspected blocker: later-turn prompt construction is no longer reinserting prior hidden reasoning. The remaining failures in the regression smoke are model-quality issues such as missing `cuda_extension`, missing `binding_registry.h`, and 30s reward timeout cases, not tokenizer/history assembly.
- Live validation of the feedback-compaction patch is now in progress on `.18` under `drkernel/logs/cuda-qwen3-14b-tp1-smoke16-feedbackfix-temp1.0-w5t50trim.run.20260422-051936`: sampled `tool_response` payloads show short actionable errors like `'ModelNew' object has no attribute 'weight'`, `'NoneType' object has no attribute 'metadata'`, and reduced compiler-error lines, instead of reinjecting full tracebacks or full extension build logs into later turns.
- Traced the remaining `'NoneType' object has no attribute 'metadata'` feedback to the CUDA reward backend rather than prompt assembly: `kernelgym/toolkit/kernelbench/pipeline.py` still had compile-retry error branches that returned `None`, and `kernelgym/toolkit/kernelbench/toolkit.py` still assumed a non-`None` pipeline result. Fixed both layers so compile-retry failures now return explicit failed `KernelExecResult` objects with `compilation_error` metadata, and toolkit callers degrade to explicit `RUNTIME_ERROR` results if a future backend path still returns `None`.
- Local verification for that backend fix passed before relaunch: `PYTHONPATH=$PWD:$PWD/drkernel pytest -q tests/test_cuda_agent_support.py -k 'build_prompt_feedback_payload or truncate_feedback_for_prompt or handles_none_pipeline_result or compile_retry_error_returns_failed_result'` reported `7 passed`; the touched toolkit files passed `py_compile`; `git diff --check` was clean.
- Stopped the earlier 16-row TP=1 smoke because it had been launched before the backend fix, then relaunched a clean replacement on `.18` under `drkernel/logs/cuda-qwen3-14b-tp1-smoke16-backendfix-temp1.0-w5t50trim.run.20260422-053812` with the same sample-16 dataset, TP=1, `ROLLOUT_MAX_NUM_SEQS=8`, reference cache enabled, and reward task timeout `30s`.
- The replacement smoke completed cleanly and confirms the targeted fix: `main.log` contains no new `'NoneType' object has no attribute 'metadata'` occurrences, while the run now produces real non-zero reward/correctness instead of the earlier all-zero pattern.
- Final smoke-16 metrics under `drkernel/logs/cuda-qwen3-14b-tp1-smoke16-backendfix-temp1.0-w5t50trim.run.20260422-053812/eval_results/step_0` are:
  - `val/test_score/kernelbench_level2_validation=-0.2704161008199056`
  - `val/kernel/final/correctness_rate=0.0625`
  - `val/kernel/best_by_turn_3/correctness_rate=0.125`
  - `val/kernel/turn_1/compilation_rate=0.125`
  - `val/kernel/turn_2/compilation_rate=0.25`
  - `val/kernel/turn_3/compilation_rate=0.1875`
- Manual review for that completed smoke is written to `drkernel/logs/cuda-qwen3-14b-tp1-smoke16-backendfix-temp1.0-w5t50trim.run.20260422-053812/eval_results/step_0/manual_review_summary.md`: sampled cases confirm that `<think>` no longer breaks section extraction, one sample recovers from a compile error to a successful turn-2 answer, and another still fails with `missing class ModelNew`, so the remaining bottleneck is model/prompt quality rather than backend error propagation.
