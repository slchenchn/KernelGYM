# 云端训练通过本机 Relay 访问 Reward API 报告

## 摘要

本次验证的结论是：云端训练侧当前不应继续使用直连地址 `http://192.168.16.39:8111`，而应通过云端宿主机上的 reverse-SSH relay 访问 Reward API。

当前有效入口是：

```text
http://127.0.0.1:18111
```

该入口在云端宿主机上已经实测可用，并且 `/workers/status` 返回中包含 `reward-40_gpu_*`，说明 A 上的 Reward API 已经能够调度到 B 上的 4090 worker 池。

## 背景

当前拓扑由三部分组成：

- 机器 A
  - 承载 Reward API
  - 可以访问本地局域网内的 reward worker
  - 可以主动 SSH 到云端训练主机
- 机器 B
  - 承载 4090 reward worker
  - 通过 A 上的 Reward API 注册和接收评测任务
- 云端训练主机
  - 负责 rollout、trainer、PPO update 等训练流程
  - 不能直接访问 A 的内网地址 `192.168.16.39:8111`

在这个网络条件下，云端训练与本地 reward 体系能够继续协同，但前提是由 A 主动建立 reverse-SSH tunnel，把 A 上的 Reward API 暴露为云端本机可访问的 relay 入口。

## 约束

- 云端训练主机不能把 `192.168.16.39:8111` 当作稳定可达的默认入口。
- 云端训练主机也不能直接访问 B 上的 4090 worker。
- Reward API 的统一调度入口仍然应保留在 A，而不是把 B 暴露给云端训练侧。

这些约束意味着训练侧不需要改成新的 standalone orchestrator，重点是把训练侧的 Reward API URL 切换到云端本机 relay。

## 方案

推荐方案如下：

1. A 主动连接云端训练主机，建立 reverse-SSH tunnel。
2. 云端宿主机通过 `127.0.0.1:18111` 访问 A 上的 Reward API。
3. A 上的 Reward API 继续向本地 worker 池分发任务，其中包括 B 上的 4090 worker。
4. 云端训练侧只认 relay URL，不再尝试直连 `192.168.16.39:8111`。

逻辑链路如下：

```text
cloud trainer
    -> cloud host 127.0.0.1:18111
    -> reverse-SSH tunnel
    -> A reward API 192.168.16.39:8111
    -> B reward workers / local worker pool
```

## 验证方法

验证重点不是“云端能否直连 A 的内网地址”，而是“云端是否能通过 relay 访问 A 上的 Reward API，并且该 Reward API 是否已经挂载 B 的 4090 worker”。

在云端宿主机上，关键检查项为：

- `curl --max-time 8 http://127.0.0.1:18111/health`
- `curl --max-time 8 http://127.0.0.1:18111/workers/status`
- `curl --max-time 8 http://192.168.16.39:8111/health`

其中前两项用于验证 relay 链路和 worker 注册状态，最后一项用于确认旧的直连路径是否仍然可用。

## 验证结果

在云端宿主机 `gz01-h20-03` 上的实测结果为：

- `http://127.0.0.1:18111/health` 返回 `200 OK`
- `http://127.0.0.1:18111/workers/status` 返回 `200 OK`
- `workers/status` 返回内容中包含 `reward-40_gpu_*`
- `http://192.168.16.39:8111/health` 请求超时

这些结果说明：

- reverse-SSH relay 已经工作正常
- 云端宿主机已经可以通过 relay 访问 A 上的 Reward API
- A 上的 Reward API 已经接入 B 上的 4090 worker
- 旧的直连地址 `http://192.168.16.39:8111` 不应再作为云端训练侧默认入口

## 最终建议

当前训练侧默认应使用：

```text
http://127.0.0.1:18111
```

如果训练跑在 Docker `host network` 容器里，容器内也应继续使用这个地址。

如果训练跑在普通 Docker bridge 网络里，则容器内应改为可访问宿主机 relay 的入口，例如宿主机网关 IP 对应的端口，或者在宿主机上增加额外 relay。无论使用哪种容器内入口，原则都不变：训练侧不要继续使用 `http://192.168.16.39:8111`。

## 结论

这项工作已经完成决策闭环：

- 不需要为“云端训练 + 本地 reward”场景先重写 standalone orchestrator
- 当前可行方案是“云端训练侧通过本机 relay 访问 A 上 Reward API”
- 当前已经验证可用的云端默认 Reward API URL 是 `http://127.0.0.1:18111`
