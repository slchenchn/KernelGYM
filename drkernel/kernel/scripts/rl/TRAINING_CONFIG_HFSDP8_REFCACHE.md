# Training Config: drkernel-14b-coldstart-trloo-16a800

**Launch date**: 2026-04-03
**Script**: `kernel/scripts/rl/14b_coldstart_trloo_16a800.sh`

## Model

| Param | Value |
|---|---|
| Base model | `hkust-nlp/drkernel-14b-coldstart` |
| Model path | `/nfs/FM/chenshuailin/checkpoints/hkust-nlp/drkernel-14b-coldstart` |
| Architecture | Qwen3-14B (14.77B params) |

## Training Infrastructure

| Param | Value |
|---|---|
| Training GPUs | 16× A800 (2 nodes × 8) |
| FSDP size | 8 (intra-node sharding, DP across nodes) |
| Sequence parallel | 4 (Ulysses) |
| Optimizer offload | Yes |
| Parameter offload | Yes |
| vLLM rollout TP | 1 |
| vLLM GPU memory util | 0.75 |

## Reward Infrastructure

| Param | Value |
|---|---|
| Reward GPUs | 16× RTX 4090 (2 nodes × 8) |
| Worker pool size | 2 (1 active + 1 warm spare) |
| Max tasks per worker | 1 (GPU memory isolation) |
| Max concurrent | 48 |
| Reward server URL | `http://192.168.16.39:8111` |
| Reference backend | `torch_compile` |
| Reference cache | Enabled, auto UUID |

## Timing Config (Reward)

| Param | Value |
|---|---|
| Warmup | 5 |
| Perf trials | 50 |
| Trim | 5 from each end (20%) |
| Correct trials | 5 |
| GPU clock | Half-locked (graphics locked, memory unlocked) |

## Algorithm

| Param | Value |
|---|---|
| Algorithm | TRLOO |
| Learning rate | 1e-6 |
| Clip ratio | 0.2 / 0.28 (dual-clip) |
| KL coef | 0.0 |
| Entropy coef | 0.0 |
| Gradient clip | 1.0 |
| Gamma | 1.0, Lambda | 1.0 |

## Data & Sampling

| Param | Value |
|---|---|
| Train dataset | `cuda_llm_rl_thinking_1025.parquet` |
| Valid dataset | `validation_data_thinking.parquet` |
| Train batch size | 16 |
| PPO mini-batch | 16 |
| PPO micro token | 5120 (× SP4 = 20480) |
| Rollout N | 16 |
| Prompt oversampling | 1.7 |
| Sample selection | efficiency_stochastic |
| Temperature | 1.0 |
| Top-P | 1.0, Top-K | -1, Min-P | 0.0 |

## Multi-Turn

| Param | Value |
|---|---|
| Enabled | Yes |
| Max turns | 3 |
| Validation samples | 8 |
| Mask void turn | Yes |

## Reward Shaping

| Param | Value |
|---|---|
| Reward weights | compilation=0.3, correctness=0.4, performance=0.3 |
| Speedup upper bound | 3.0 |
| Coverage reward | Enabled, weight=0.5, type=time_coverage |
| Coverage RS | turn, threshold=0.3, factor=0.1 |
| Decoy kernel detection | Yes |
| Penalties | compilation_fail=-0.5, correctness_fail=-0.3, perf_degrade=-0.1 |

## Checkpointing

| Param | Value |
|---|---|
| Save freq | Every 10 steps |
| Test freq | Every 10 steps |
| Checkpoint dir | `<log_dir>/<run_name>` |
| Val before train | Yes |
| Total epochs | 1000 |

## Lengths

| Param | Value |
|---|---|
| Max prompt length | 10240 |
| Max response length | 8192 |
| Max batched tokens | 19432 |
