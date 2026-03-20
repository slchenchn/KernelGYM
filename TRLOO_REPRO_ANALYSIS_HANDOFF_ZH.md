# TRLOO 复现与代码分析接手文档

## 1. 文档目的

这份文档合并并替代之前分散的几份实验报告，目标是让后续接手分析代码的人只看这一份就能知道：

- 如何复现 `14B coldstart -> TRLOO` 的关键路径
- 如何快速跑 `initial val probe`
- 当前训练时间瓶颈在哪里
- reward 拓扑实验已经验证了什么
- 代码级 root-cause 分析已经排除了什么、还剩什么

适用场景：

- 继续复现 `TRLOO`
- 继续做 reward stack / `KernelGYM` 代码诊断
- 不想再从零阅读多份历史报告

## 2. 当前结论摘要

### 2.1 训练侧

- 当前 `14B` 训练不是被 `update_actor` 拖慢。
- 当前可见 step 的墙钟时间主要花在：
  - `gen`
  - `old_log_prob`
- 其中 `gen` 内部又主要是 reward 侧，而不是纯 rollout。

基于当前训练日志 `drkernel/logs/trloo-14b-20260318-012043.log`：

| 项目 | 结论 |
| --- | --- |
| successful step 净时间 | `~1065.75s` |
| 可见 step 墙钟时间 | `~50-57 min` |
| `gen` 占 successful step | `~80.5%` |
| `old_log_prob` 占 successful step | `~18.1%` |
| `update_actor` 占 successful step | `~1.0%` |
| `turn/reward` vs `turn/rollout` | `~4x` |

工程上的实用结论：

- 当前训练是 reward-bound，不是 optimizer-bound。

### 2.2 reward 拓扑侧

在统一的小规模 `initial val probe` 口径下：

- `val_sample_size=10`
- `n=8`
- `num_perf_trials=100`
- 截到前 `100` 个加权 `Env Result`

已得到的直接对比如下：

| 拓扑 | 到 first100 的时间 |
| --- | ---: |
| `local 4xA800` | `424.27s` |
| `local 4x4090` | `521.46s` |
| `local 8x4090` | `358.27s` |
| `remote 8x4090` | `323.66s` |

还需要单独看待的点：

- 早先一轮 `remote 16x4090` 明显更慢
- 但后续 instrumented rerun 并没有稳定复现 “16 卡一定比 8 卡差” 这个结论
- 所以现在不能把 “`16x4090` 拓扑本身有问题” 当成最终结论

更稳妥的总结是：

- `4xA800` 在 per-task reward timing 上通常更好
- `8x4090` 在 early-sample throughput 上更好
- `remote 8x4090` 的 first100 吞吐目前最好

### 2.3 代码级 root-cause 侧

当前已经确认：

1. 父 `Env Result.env_state.metadata.tm_*` 不是 child kernel task 的真实 completion timing  
   原因：
   - child metadata 会先被带到 parent
   - parent API completion 又会覆盖同名 `tm_*`

2. child kernel task 的大等待是真实的，而且在 worker 本地就能看到  
   证据：
   - `gpu_worker.py` 中 `[WorkerTiming]`
   - `[WorkerCompleteBreakdown]`
   - child Redis `result:*_kernel` 里的 `tm_*`

3. 这个大等待不是普通 Redis RTT，也不是普通网络 RTT  
   证据：
   - `SLOWLOG` 为空
   - 合成 `HSET` 探针只有毫秒级

4. 把 `complete_task()` 的写入顺序从 `status -> result` 改成 `result -> status`，并不能消掉 worker-local 的 `~50s` wave  
   结果：
   - 只是把大等待从 `tm_status_hset_s` 挪到了 `tm_result_hset_s`
   - worker `total_s` 基本不变

因此当前最稳的结论是：

- root cause 不在 “先写 status 还是先写 result” 这个顺序本身
- 更像在 live worker completion path 更深一层的 await / barrier / lifecycle 行为里

## 3. 当前代码入口

如果你要继续分析代码，最值得先看的文件是：

- `kernelgym/server/task_manager.py`
- `kernelgym/worker/gpu_worker.py`
- `kernelgym/worker/subprocess_pool.py`
- `kernelgym/toolkit/kernelbench/pipeline.py`
- `kernelgym/toolkit/kernelbench/timing.py`
- `drkernel/kernel/workers/reward_manager/kernel_async.py`
- `drkernel/kernel/scripts/rl/train_rl_common.sh`

分析脚本：

- `drkernel/kernel/analysis/analyze_initial_val_probe.py`
- `drkernel/kernel/analysis/extract_initial_val_metrics.py`
- `drkernel/kernel/analysis/plot_training_times.py`

probe 启动脚本：

- `drkernel/kernel/scripts/rl/14b_coldstart_initial_val_probe.sh`

## 4. 如何复现

### 4.1 训练前固定检查

任何分布式启动前，都先在实际运行节点验证：

```bash
which python
python - <<'PY'
import os, sys
print(sys.executable)
print(os.environ.get("VIRTUAL_ENV"))
PY
```

还要确认：

- model path 可见
- dataset path 可见
- reward API endpoint 可达
- `KERNELGYM_SERVER_URL` 已显式设置
- 当前命令会使用仓库下共享的 `uv` 虚拟环境：
  - `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM/.venv`

### 4.2 跑 full training

训练脚本入口：

- `drkernel/kernel/scripts/rl/14b_coldstart_trloo_mrs_pr_prs.sh`

执行位置：

- 必须在 training node 上执行
- 不在 reward 节点执行

原因：

- 训练节点负责加载 `14B` 模型、启动 rollout / vLLM / trainer
- reward 节点只提供 `KernelGYM` API / worker

如果只想确认训练主链路：

1. 先把 reward stack 起好
2. 在训练节点从 `drkernel/` 目录启动
3. 观察：
   - `Initial validation metrics`
   - `Training Progress`
   - `step:N`
   - `over_sampling`
   - WandB 是否 live

### 4.3 跑 initial val probe

probe 脚本入口：

- `drkernel/kernel/scripts/rl/14b_coldstart_initial_val_probe.sh`

注意：

- 必须在 training node 上执行
- 必须使用当前仓库的共享 `uv` 环境：
  - `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM/.venv`
- `KERNELGYM_SERVER_URL` 是必填运行参数
- 这个脚本本身不给默认值
- 如果不显式传入，reward manager 不知道连接哪个 `KernelGYM` API
- 脚本现在会自动把 stdout / stderr 重定向到：
  - `drkernel/logs/init-val-probe-YYYYmmdd-HHMMSS.log`
- 如果要自定义日志文件，可以显式传：
  - `PROBE_LOG=/path/to/log`

推荐口径：

```bash
VAL_SAMPLE_SIZE=10
N_VAL=8
NUM_PERF_TRIALS=100
```

停止边界：

- 不等 full `Initial validation metrics`
- 直接用前 `100` 个加权 `Env Result`

分析命令：

```bash
python drkernel/kernel/analysis/analyze_initial_val_probe.py \
  --max-weighted-results 100 \
  drkernel/logs/<probe-log>.log
```

一个最小可运行模板是：

```bash
cd /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM/drkernel

KERNELGYM_SERVER_URL=http://<kernelgym-host>:<port> \
MODEL_PATH=/nfs/FM/chenshuailin/checkpoints/hkust-nlp/drkernel-14b-coldstart \
VAL_SAMPLE_SIZE=10 \
N_VAL=8 \
NUM_PERF_TRIALS=100 \
bash kernel/scripts/rl/14b_coldstart_initial_val_probe.sh
```

如果要自定义日志文件：

```bash
cd /nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM/drkernel

PROBE_LOG=logs/my-init-val-probe.log \
KERNELGYM_SERVER_URL=http://<kernelgym-host>:<port> \
MODEL_PATH=/nfs/FM/chenshuailin/checkpoints/hkust-nlp/drkernel-14b-coldstart \
VAL_SAMPLE_SIZE=10 \
N_VAL=8 \
NUM_PERF_TRIALS=100 \
bash kernel/scripts/rl/14b_coldstart_initial_val_probe.sh
```

之后再用生成的日志文件分析：

```bash
python kernel/analysis/analyze_initial_val_probe.py \
  --max-weighted-results 100 \
  logs/init-val-probe-<timestamp>.log
```

### 4.4 训练时间分析

训练 timing 图和 CSV：

```bash
python drkernel/kernel/analysis/plot_training_times.py \
  drkernel/logs/trloo-14b-20260318-012043.log
```

输出：

- `step_timings.csv`
- `turn_timings.csv`
- `summary.csv`
- `step_timings.svg`
- `turn_timings_series.svg`
- `turn_timings_by_turn.svg`

## 5. reward 拓扑实验结论

### 5.1 five-way 初始结论

| 拓扑 | first100 |
| --- | ---: |
| `remote 8x4090` | `323.66s` |
| `local 8x4090` | `358.27s` |
| `local 4xA800` | `424.27s` |
| `local 4x4090` | `521.46s` |

实用解释：

- 如果只看 early throughput，`8x4090` 更划算
- 如果只看 per-task timing，`4xA800` 往往更好
- `4x4090` 在这组实验里并不占优

### 5.2 关于 `remote 16x4090`

这部分要谨慎表达：

- 早先一轮 `remote 16x4090` 很慢
- 但 instrumented rerun 没有稳定复现 “16 卡一定更差”
- 所以不能把这个写成硬结论

当前更可靠的判断是：

- `remote 16x4090` 是否显著更差，不稳定
- 但 worker completion wave 这个问题，在 `remote8` 上已经足够明显，值得单独追

## 6. worker completion wave 分析

### 6.1 已确认的现象

在 `remote8-v2g` 上，child kernel task 的 `TaskManagerTiming` 显示：

- `status_hset_s`
  - mean: `16.3643s`
  - median: `12.7066s`
  - max: `49.3846s`
- `result_hset_s`
  - median 只有毫秒级
- 很多 task 满足：
  - `total_s ≈ status_hset_s`

这说明当时的大等待主要压在第一条 `HSET task:... status=completed` 上。

### 6.2 为什么 parent `Env Result` 看起来没这么慢

因为 parent task 会覆盖 child metadata：

- child kernel task 的 `tm_*` 可以很大
- parent evaluation task 的 `tm_*` 可以很小

所以：

- 看 parent `Env Result.env_state.metadata.tm_*` 会误判
- authoritative source 是：
  - child Redis result key
  - worker log

### 6.3 为什么不能怪 Redis / 网络

已经做过两类排除实验：

1. Redis `SLOWLOG`  
   结果：空

2. `.39 -> .23` 的合成 `HSET` 探针  
   结果：毫秒级

所以不能再用：

- “Redis 本身慢”
- “普通网络 RTT 慢”

来解释 `10s-49s`

### 6.4 `result-first` 受控实验的结论

新实验 `remote8-v2h` 把顺序改成：

- `result -> status`

结果：

- `tm_status_hset_s`
  - mean: `0.00395s`
- `tm_result_hset_s`
  - mean: `25.2352s`
- worker `total_s`
  - 仍然在 `~52s`

与 `remote8-v2g` 对比：

| Run | `run_toolkit_s` mean | `complete_task_s` mean | `total_s` mean | `total_s` median |
| --- | ---: | ---: | ---: | ---: |
| `remote8-v2g` | `28.8967s` | `22.3509s` | `51.2473s` | `52.05s` |
| `remote8-v2h` | `30.4728s` | `21.6802s` | `52.1521s` | `52.32s` |

因此当前结论是：

- `result-first` 不是 root-cause fix
- 它只是把大等待从一个 `HSET` 移到另一个 `HSET`

## 7. 下一步最值得做什么

现在最值得继续的不是再做拓扑广撒网，而是沿着 worker completion path 往下拆。

建议优先级：

1. 继续追 worker-local wave  
   重点：
   - `gpu_worker.py`
   - `subprocess_pool.py`
   - `task_manager.py`

2. 查 live completion path 里的共享 barrier / await 行为  
   因为：
   - 顺序换了，大等待还在
   - 说明更像某种共享 completion wave

3. 如果还要做性能对比，先固定任务集合再 replay  
   否则：
   - success bucket 会受 sample mix 影响
   - 容易把 throughput 结论和单任务 latency 结论混在一起

## 8. 日志与产物位置

训练日志：

- `drkernel/logs/`

probe 归档日志：

- `logs/probes/kernelgym/remote8-probe-v2f/`
- `logs/probes/kernelgym/remote8-probe-v2g/`
- `logs/probes/kernelgym/remote8-probe-v2h/`
- `logs/probes/kernelgym/remote16-probe-v2f/`

这些目录放在根目录 `logs/` 下，是为了：

- 避免再被 git 跟踪
- 统一收口实验产物

## 9. 本文替代关系

这份文档替代并吸收了之前分散的几份分析：

- `INITIAL_VAL_4090_VS_A800_PROBE.md`
- `LOCAL_A800_4090_PROBE_COMPARISON.md`
- `REMOTE_8X_VS_16X_4090_ANALYSIS.md`
- `TRAINING_TIME_ANALYSIS.md`

后续如果继续分析，优先更新这一份，而不是再分裂出新的平行报告。
