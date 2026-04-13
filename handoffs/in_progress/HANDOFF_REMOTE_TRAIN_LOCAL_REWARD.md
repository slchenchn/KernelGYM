# 云端训练 + A 上 Reward API + B 上 4090 Worker 拓扑说明

## 背景

当前网络关系如下：

- **机器 A**
  - 是当前 reward API 主机
  - 可以通过 VPN 访问云端 server
  - 可以通过局域网访问机器 B
- **机器 B**
  - 挂着 4090
  - 适合运行 reward worker / 具体评测执行
- **云端 server**
  - 适合运行 rollout、PPO update 等计算密集部分
- **限制**
  - 云端 server 和机器 B **不能直接互联**

这意味着：

- 云端训练侧不能直接访问 B
- 但只要云端能访问 A，就仍然可以通过 A 上的 reward API 间接使用 B 的 4090 能力

## 结论

**不要先急着拆 standalone orchestrator。**

在这个拓扑下，第一步应该优先验证：

- 云端能不能直接访问 A 上现有的 reward API
- A 上的 reward API 能不能正常调度到 B 上的 4090 worker

当前优先方案就是：

- **A 作为 reward API 主机**
- **B 作为 4090 reward worker 主机**
- **云端训练侧只访问 A 上现有的 reward API**

## 需要记住的固定入口

- 云端登录命令：`ssh 10.0.18.3-H20`
- 云端训练侧应使用的 Reward API URL：`http://192.168.16.39:8111`

## 推荐方案

### 方案：A 作为 reward API 主机

这是优先推荐的方案，改动最小。

思路是：

- A 上运行当前 reward API
- B 上运行 4090 reward worker
- 云端训练侧只访问 A
- A 再把评测任务分发给 B

逻辑链路：

```text
cloud trainer / rollout / PPO
    -> A: reward API
    -> B: reward worker
    -> A
    -> cloud
```

这个方案的优点：

- 不需要先重写训练框架
- 不要求云端直接访问 B
- A 已经是现有 reward API 主机，改动最小

## 决策顺序

建议按这个顺序判断：

1. 先验证 A 上 reward API 是否健康
2. 再验证 A 是否能正常调度到 B 上的 4090 worker
3. 最后验证云端是否能访问 A 的 reward API URL

## 不依赖任何现有 repo 的测试方法

下面所有测试都不依赖当前 repo，可以在另一台机器上直接照抄。

---

## 测试 1：验证 A 上 reward API 是否可用

在 **机器 A** 上验证：

```bash
curl http://127.0.0.1:8111/health
curl http://192.168.16.39:8111/health
```

如果这一步失败，问题在 A 本机 reward API，不要继续。

---

## 测试 2：验证 A -> B 的 reward worker 链路

在这一步，不需要新写 relay。  
只需要验证：

- A 能访问 B
- A 上 reward API 发出的任务最终能在 B 上执行

```bash
nc -vz <B的局域网IP> 22
```

这一条只是最小网络检查。  
如果你要进一步确认 A->B 的 reward worker 调度是否正常，需要在 A 上发起一次真实 `/evaluate` 请求，并在 B 上观察是否有对应 worker 执行。

---

## 测试 3：验证云端 -> A 的 reward API 可达性

先登录云端：

```bash
ssh 10.0.18.3-H20
```

在云端验证：

```bash
curl http://192.168.16.39:8111/health
```

如果这一步成功，说明云端训练侧已经可以直接访问 A 上现有的 reward API。

## 当前建议

在你给定的 A / B / 云端 拓扑下，建议下一步是：

1. 先确认 A 上的 reward API 健康
2. 再确认 A 能访问并调度到 B 的 4090 worker
3. 最后确认云端能访问 `http://192.168.16.39:8111`
4. 如果这三步都通，训练侧就直接使用 A 上现有 reward API

之后训练侧统一使用：

```text
http://192.168.16.39:8111
```

**不要先从重写 orchestrator 开始。**
