# DR. KERNEL: Reinforcement Learning Done Right for Triton Kernel Generations

Wei Liu1, Jiawei $\mathbf { \boldsymbol { x } } _ { \mathbf { \boldsymbol { u } } } { } ^ { 3 }$ , Yingru $\mathbf { L i } ^ { 2 }$ , Longtao Zheng4, Tianjian $\mathbf { L i } ^ { 2 }$ , Qian Liu2, Junxian He1 1HKUST 2TikTok 3CUHK(SZ) 4NTU 

# Abstract

High-quality kernel is critical for scalable AI systems, and enabling LLMs to generate such code would advance AI development. However, training LLMs for this task requires sufficient data, a robust environment, and the process is often vulnerable to reward hacking and lazy optimization. In these cases, models may hack training rewards and prioritize trivial correctness over meaningful speedup. In this paper, we systematically study reinforcement learning (RL) for kernel generation. We first design KERNELGYM, a robust distributed GPU environment that supports reward hacking check, data collection from multi-turn interactions and long-term RL training. Building on KERNELGYM, we investigate effective multi-turn RL methods and identify a biased policy gradient issue caused by self-inclusion in GRPO. To solve this, we propose Turn-level Reinforce-Leave-One-Out (TRLOO) to provide unbiased advantage estimation for multi-turn RL. To alleviate lazy optimization, we incorporate mismatch correction for training stability and introduce Profiling-based Rewards (PR) and Profiling-based Rejection Sampling (PRS) to overcome the issue. The trained model, DR. KERNEL-14B, reaches performance competitive with Claude-4.5-Sonnet in Kernelbench. Finally, we study sequential test-time scaling for DR. KERNEL-14B. On the KernelBench Level-2 subset, $3 1 . 6 \%$ of the generated kernels achieve at least a $1 . 2 \times$ speedup over the Torch reference, surpassing Claude-4.5-Sonnet $( 2 6 . 7 \% )$ and GPT-5 $( 2 8 . 6 \% )$ . When selecting the best candidate across all turns, this $1 . 2 \times$ speedup rate further increases to $4 7 . 8 \%$ . All resources, including environment, training code, models, and dataset, are included in github.com/hkust-nlp/KernelGYM. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-09/0aa500ad-1825-4ed8-9d27-1d5a17dca497/1816cf7fe720b73c7eabd4c4301e76cdcc7bd20e519c808fb47227c25537aec6.jpg)



Figure 1: Rate of generated kernels achieving at least a $1 . 2 \times$ speedup over the Torch reference on KernelBench across three level subsets. DR. KERNEL-14B is competitive with Claude-4.5-Sonnet and GPT-5, and applying sequential test-time scaling further improves DR. KERNEL-14B, surpassing both models on two of the three subsets.


# 1 Introduction

Efficient GPU kernels are critical for scalable AI systems. Seminal works like FlashAttention (Dao, 2024) and FlashInfer (Ye et al., 2025) have demonstrated that specialized kernels are essential for unlocking the full efficiency of modern LLMs. However, developing such 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-09/0aa500ad-1825-4ed8-9d27-1d5a17dca497/b2436cfe2000f224ff3a3baea7c24aa60c5b8a576983d3228f75352be0291852.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-09/0aa500ad-1825-4ed8-9d27-1d5a17dca497/a5948875d9fff3831931ce0bee27ceadf7a7074c5fc70ee5a59ed48bcfad309e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-09/0aa500ad-1825-4ed8-9d27-1d5a17dca497/e7bf7744628dfc1b057bb8f1bdeec632c82e3781f1213263a87c58789c6d831d.jpg)



Figure 2: Left: The plot uses a dual y-axis to compare two metrics. We report results from two models: Fast@1 of the model trained without reward hacking check (§ 3.3) (w/o hacking check), and Fast@1 / Fast@1.2 of the model trained with hacking check enabled. Evaluation is done using the same standard for all variants with hacking check. Multi-turn RL is run on Qwen3-8B-Base after cold-start SFT, using TRLOO for advantage estimation (§4.2) and KERNELGYM as the execution environment (§3). Right: Representative cases illustrating reward hacking and lazy optimization. In Hacked Kernel.py, the model emits a Triton kernel to satisfy the ${ \mathit { \Pi } } ^ { \prime \prime } ( { \widehat { \pmb { \theta } } } )$ triton.jit” heuristic but never calls it, and additionally skips the real computation under the default training mode, inflating the measured speedup. In Lazy Optimization.py, the model replaces only a trivial sub-operation (channel summation) with a kernel while leaving the remaining computation in Torch, missing the larger gains from fusion.


kernels remains difficult. It requires deep expertise spanning both algorithms and GPU hardware intricacies. While Domain-Specific Languages (DSLs) like Triton and TileLang (Wang et al., 2025) have lowered the entry barrier compared to CUDA, achieving peak performance still requires significant manual engineering. This difficulty makes kernel development a natural candidate for automation. 

Kernel generation is characterized by easily accessible optimization objectives. Correctness can be verified through execution, and efficiency can be measured via profiling, making these tasks naturally suited for RL, which, therefore, offers a promising approach to enhancing the ability of LLMs to generate kernel code. However, these optimization benefits come with potential pitfalls. During training, models can devolve into reward hacking, exploiting measurement loopholes or executing invalid operations that appear fast but result in meaningless optimizations. Alternatively, models may generate correct but trivial kernel implementations that fail to deliver meaningful speedup, a phenomenon we refer to as lazy optimization, as shown in Figure 2. Previous works only partially addressed these challenges and remain RL not fully explored for this field. For instance, AutoTriton (Li et al., 2025) optimizes solely for correctness, neglecting the crucial objective of speedup. TritonRL (Woo et al., 2025) identifies the risk of reward hacking but relies on imprecise LLM-as-a-judge mechanisms rather than rigorous execution-based verification. Similarly, while CudaLLM (CudaLLM Team, 2025) collects valuable data, it stops short of full-scale RL training, reporting only correctness metrics. Furthermore, most prior efforts are limited to single-turn generation while kernel optimization can be recursively refined for multiple rounds. Kevin (Baronio et al., 2025) attempts multi-turn RL, yet it is constrained by a small-scale dataset of only 280 samples split from the KernelBench benchmark. 

In this work, we systematically study RL training for kernel code generation. We follow the task definiton of previous works (Ouyang et al., 2025; Li et al., 2025; Baronio et al., 2025), where a Torch reference code is provided and models are asked to optimize the implementation via kernel code. We focus on Triton, a Pythonic, high-level GPU programming language that is more amenable to current LLMs than low-level CUDA. Moreover, Triton’s abstraction makes execution-based safeguards against reward hacking more tractable during training, making it an ideal testbed for developing RL methodologies for kernel generation. Following the adage, $^ { \prime \prime } A$ workman must first sharpen his tools if he is to do his work well,” we first build a robust environment, KERNELGYM. Unlike prior ad-hoc solutions, KERNELGYM is a scalable, distributed serving system tailored for long-horizon RL: it provides strict fault isolation to tolerate frequent CUDA runtime failures, and exposes granular environment 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-09/0aa500ad-1825-4ed8-9d27-1d5a17dca497/148772e2b2452d414262d3070122f0274d832b4bc09a9dd4e5bfc0421bca7ced.jpg)



Figure 3: Overview of KERNELGYM and our training framework. Left: We study RL training methods for kernel generation, including multi-turn RL with TRLOO, profilingbased rewards (PR), and profiling-based rejection sampling (PRS). Right: The architecture of KERNELGYM: a server-worker split distributed design. The server side (interface $^ +$ task manager) receives evaluation jobs and schedules to registered distributed GPU workers; each job runs in an isolated subprocess; toolkits produce structured signals for training, parallel evaluation and data collections.


feedback, including execution profiling and hacking checks, to enable rigorous evaluation and high-quality data collection from multi-turn interactions. 

Equipped with KERNELGYM, we investigate effective multi-turn RL methods for LLMs. We identify that standard GRPO can induce biased policy-gradient updates; to address this, we propose Turn-level Reinforce-Leave-One-Out (TRLOO), an unbiased advantage estimator for multi-turn RL. Furthermore, we alleviate “lazy optimization” from stability and optimization objective prospectives. We find mismatch correction contributes to the training stability. And we further propose Profiling-based Rewards (PR) and Profiling-based Rejection Sampling (PRS) to explicitly incentivize alleviating performance bottlenecks. Finally, we study sequential test-time scaling (STTS) to maximize the inference capability of our trained models. Experimental results demonstrate KERNELGYM with modular design and hacking check enables long-term RL training. Our final multi-turn RL method, DR. KERNEL, yields strong DR. KERNEL-14B model which achieves substantial gains on two KernelBench (Ouyang et al., 2025) subsets, reaching performance competitive with frontier models like Claude-4.5-Sonnet. And for DR. KERNEL-14B with STTS, on the KernelBench Level-2 subset, $3 1 . 6 \%$ of the generated kernels achieve at least a $1 . 2 \times$ speedup over the Torch reference, surpassing Claude-4.5-Sonnet $( 2 6 . 7 \% )$ and GPT-5 $( 2 8 . 6 \% )$ . When selecting the best candidate across all turns, the $1 . 2 \times$ speedup rate further increases to $4 7 . 8 \%$ . 

# 2 Pitfalls in Kernel Generation

Correctness and speedup are the two objectives of kernel generation. However, triton kernel generation is particularly vulnerable to reward hacking (Baronio et al., 2025; Woo et al., 2025): the model can produce outputs that appear correct and fast under the evaluation while being actually meaningless. A simple example is copying the Torch reference implementation, it passes correctness checks but yields only ${ \bar { \sim } } 1 . 0 { \bar { \times } }$ speedup, providing an easy way to harvest reward without learning kernel generation. Another common failure mode occurs when Triton kernels are emitted but never executed in the kernel entry function, resulting in misleading timing measurements. As shown in Figure 2 (right), although the model generates the kernel implementation for LayerNorm, the kernel is never actually executed. 

Beyond hacking, we observe a second bottleneck that is specific to performance-oriented kernel generation. In Figure 2 (Left), we validate Fast@1 and Fast $@ 1 . 2$ on KernelBench Level 

2 during training. Fast@1 / Fast@1.2 represent the fraction of kernels passing correctness checks with at least $1 \times / 1 . 2 \times$ speedup over the Torch reference. Fast $@ 1$ improves steadily, while the stricter Fast $@ 1 . 2$ saturates quickly (around ${ \sim } 1 0 0$ steps). Unlike standard code generation, where functional correctness is often the end goal, kernel generation ultimately targets meaningful speedup. Many KernelBench (Ouyang et al., 2025) tasks admit trivialbut-correct implementations (e.g., local rewrites such as swapping a simple activation) that do not address the true runtime bottlenecks. As shown in Figure 2 (right), the model leverages the kernel for the simple summation operation, while leaving other operations to the Torch implementation, thereby overlooking the potential benefits of more advanced fusion optimizations. Since these solutions still yield only $\sim 1 \times$ speedup, the policy tends to exploit such low-hanging fruits to improve Fast@1, without producing kernels that clear the higher bar required by Fast@1.2. We refer to this behavior as lazy optimization. Cases are provided in Figure 2 (Right) and Appendix E.1. 

These two issues are also evident and remain unresolved in previous works. For instance, in AutoTriton (Li et al., 2025), despite implementing a rule-based reward that assigns 0 reward to code without the ${ \mathit { \Pi } } ^ { \prime \prime } ( { \widehat { \pmb { \theta } } } )$ triton.jit” decorator, the model still encounters hacking cases, such as neglecting to call the function, as shown in Figure 2. In our evaluation, code generated by their released model still exhibits approximately $1 0 \%$ hacking cases in the KernelBench level-1 subset. In addition to reward hacking, the issue of lazy optimization persists. As shown in Table 1, while AutoTriton achieves a notable $3 0 . 6 \%$ performance on the Fast@1 metric in the Kernelbench level-2 subset, its performance on the stricter Fast@1.2 metric quickly saturates to just only $9 . 2 \%$ . 

Next, to make RL for kernel generation practically feasible, we start with a robust execution environment with hacking checks and torch profiler (§ 3), then introduce an unbiased multi-turn RL estimator (§ 4), and finally improve training stability and align optimization objectives toward real speedup to alleviate lazy optimization $( \ S 5 )$ . 

# 3 KERNELGYM: A Gym for Kernel Generations

# 3.1 Design Principles

Training LLM agents for iterative kernel generation requires an environment that can (1) evaluate correctness and performance at scale, (2) remain resilient against frequent CUDA runtime failures, and (3) provide granular feedback for RL optimization. To meet these requirements under constrained GPU resources, we architect KERNELGYM as a scalable, distributed serving system that strictly decouples agent clients from kernel execution (Figure 3, right). This separation keeps the client implementation lightweight—agents focus on policy learning, while the system handles scheduling, resource management, and failure recovery. 

Concretely, the design of KERNELGYM follows four principles tailored to GPU workloads: (i) Serialized execution: profiling is highly sensitive to contention, so we enforce a one-GPU-one-task policy to prevent context pollution and ensure reliable timing; (ii) Elastic scalability: GPU workers can be added or removed dynamically without interrupting training; (iii) Fault isolation and self-recovery: unsafe generated kernels frequently trigger illegal memory access or unrecoverable CUDA errors, hence failures must be isolated at the task level and recovered automatically to maintain long-horizon availability; (iv) Rich environmental feedback: coarse signals such as pass/fail or a single speedup value are insufficient for RL, so KERNELGYM exposes structured feedback (e.g., profiling summaries and reward-hacking detection) to support optimization and data collection. 

# 3.2 Architecture and Modular Design

Server KERNELGYM adopts a server–worker architecture. The server serves as the central coordinator, consisting of an Interface (FastAPI) and a Task Manager (Redis $^ +$ scheduler). The Interface exposes REST APIs for task submission/querying and worker registration. The Task Manager uses Redis to maintain persistent task/worker states and dispatches tasks to available workers with timeout-based re-queuing to sustain throughput. 

GPU Worker and Monitor Kernel evaluations are executed by distributed GPU workers, where each GPU is treated as an independent worker instance. Each worker pulls scheduled tasks from the server and runs them sequentially using the configured backend/toolkits (§3.3). To contain CUDA/runtime failures from generated kernels without corrupting longrunning processes, each evaluation runs in a fresh spawned subprocess, while the parent worker remains CUDA-clean and continues serving subsequent tasks. A worker monitor tracks liveness (e.g., heartbeat/process health), automatically restarts failed workers, and reassigns unfinished tasks to healthy workers to maintain RL training stability. 

# 3.3 Backends and Toolkits

Backends Following Ouyang et al. (2025), KERNELGYM runs the generated kernel code and evaluates it on two basic toolkits: correctness and speedup against a Torch reference. In this work we mainly use a Triton backend, but the same interface can also support other kernel languages (e.g., CUDA, TileLang). More broadly, KERNELGYM can be extended to other GPU tasks by adding other toolkits that define how to run the code and how to leverage the output. 

For correctness, the backend compares the generated code with a reference implementation under a fixed test protocol (e.g., multiple randomized inputs) and returns a discrete status such as pass, mismatch, runtime error, or compilation error. For performance, the backend measures running time using a consistent timing procedure (e.g., warmup followed by repeated runs) and reports the speedup relative to the baseline. The performance metric would only be measured on the correct kernel code. 

Hacking Check As discussed in §2, reward hacking is a major failure mode in performanceoriented kernel RL. To mitigate such behaviors, KERNELGYM implements an executionbased hacking check that filters suspicious candidates from optimization. 

Concretely, KERNELGYM instruments Triton’s launch path to record executed Triton kernels and measures end-to-end runtime in both train and eval modes. Motivated by Figure 2 (right), where the model branches on self.training to bypass execution of kernel code and inflate speedup, we mark a candidate as incorrect if it executes no Triton kernel in either mode. 

Profiler Beyond scalar feedbacks, KERNELGYM exposes profiling summaries to provide richer and more reliable feedback, which mainly serves as informative context for multiturn optimization where the models recursively optimize the code during multi-turn RL training or test-time scaling. For incorrect candidates, the profiler returns structured failure diagnostics (e.g., exception type and traceback) to help the model localize and fix errors in subsequent turns. For correct executions, it provides kernel-level summaries of the executed code, alongside the default correctness and performance measurements, enabling the model to identify unoptimized operators. These signals are also used during training to reduce the lazy optimization behavior in §2. We operationalize this idea via Profiling-based Rewards and Profiling-based Rejection Sampling (§5). The examples for profiling feedback are shown in Figure 10 (Appendix E.1). 

# 4 Multi-Turn RL with KERNELGYM

Kernel generation naturally lends itself to multi-turn refinement. Just as human developers iterate by writing, executing, and revising kernels based on runtime profiling, LLMs can improve solutions through repeated propose–evaluate–refine cycles. Previous work like AlphaEvolve (Novikov et al., 2025) has demonstrated that such interaction with an execution environment can even lead to the discovery of fundamental algorithms. 

Motivated by this, KERNELGYM enables long-term multi-turn RL for kernel generation: at each turn, the model proposes a revised kernel conditioned on the history, and KERNELGYM executes it to provide immediate feedback. This setup is distinct from recent agentic RL with multi-step tool use (Wei et al., 2025; Jin et al., 2025), where learning is typically driven 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-09/0aa500ad-1825-4ed8-9d27-1d5a17dca497/deba819cedad55248ef180b7f0b2fba67e552819f5d79995d75336be2e42aed3.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-09/0aa500ad-1825-4ed8-9d27-1d5a17dca497/534884689feb49eb07da8f9666353511731a5f4b53685cfd240c6fb0da2f3d80.jpg)



Figure 4: Fast@1 on KernelBench Level 2. Left: Fast@1 at turn 3 over training steps. Right: Fast@1 across turns (evaluated at the selected checkpoint). Since all methods besides AutoTriton achieve their best performance at turn 3, we select checkpoints based on turn 3 performance. For AutoTriton we use their released model.


by a sparse, single-outcome reward. In contrast, our environment yields dense, turn-level rewards after each execution. 

# 4.1 Cold-Start Data Collections

To alleviate the data scarcity for kernel code and teach models basic kernel-generation skills (e.g., tiling, fusion, etc.), we distill multi-turn trajectories from a proprietary model (e.g., GPT-5) interacting with KERNELGYM. The prompting template is provided in Appendix F.1. 

We start from 8K kernel-generation queries from CUDALLM-SFT (CudaLLM Team, 2025) and use GPT-5 to generate 5-turn Triton implementations. At each turn, the generated code is executed in KERNELGYM and the model receives feedbacks from the environment (§ 3.3), including correctness status, error diagnostics for failed runs or runtime, and profiling summaries for successful kernels. This feedback is appended to the next-turn query, prompting the model to refine the implementation conditioned on the full interaction history. 

# 4.2 Multi-Turn RL

Reward Design As in cold-start collection, we provide the model with environment feedback and optimize it in a multi-turn RL setting. Following prior work (Baronio et al., 2025; Woo et al., 2025), we combine correctness and speedup to define the per-turn reward for the response $i$ at the $t { \cdot }$ -th turn $y _ { i , t }$ as: 

$$
R _ {i, t} = C \left(y _ {i, t}\right) + C \left(y _ {i, t}\right) \cdot \operatorname {s p e e d u p} _ {i, t}. \tag {1}
$$

Here $C ( y _ { i , t } )$ is a binary correctness reward, and speedupi,t is computed from runtime measurements. We clip the speedup term to improve training stability and reduce the impact of anomalous evaluations: speed $\begin{array} { r } { \mathbf { \boldsymbol { \mathsf { u p } } } _ { i , t } = \bar { \operatorname* { m i n } } \Bigl ( \frac { T _ { \mathrm { r e f e r e n c e } } } { T _ { \mathrm { k e r n e l } } } , 3 \Bigr ) . } \end{array}$  TreferenceT , 3 . This clipping prevents rare timing artifacts from producing excessively large rewards, and is also consistent with the observation that speedups beyond $3 \times$ are uncommon given the current capabilities of LLMs in such tasks. 

Multi-Turn Advantage We use a reward-to-go formulation to compute turn-level returns, assigning credit to each turn based on subsequent rewards in the interaction. This is natural for multi-turn kernel refinement: earlier turns influence later turns through accumulated code and environment feedback. Specifically, we define the return at turn $t$ as 

$$
G _ {i, t} = \sum_ {t ^ {\prime} = t} ^ {T} \gamma^ {t ^ {\prime} - t} R _ {i, t ^ {\prime}}, \tag {2}
$$

where $\gamma \in ( 0 , 1 ]$ is a discount factor and $R _ { i , t }$ is the per-turn reward. We fix $\gamma = 1$ in this work. 

After computing $G _ { i , t } ,$ we form turn-level advantages using a GRPO-style in-batch mean baseline. For each prompt (i.e., a fixed kernel task specification), we sample $K$ independent 

rollouts. At turn $t ,$ some rollouts may be invalid (e.g., masked out or terminated early); we denote $\mathcal { G } _ { t }$ as the set of valid rollouts for a given prompt at turn $t ,$ and $N _ { t } = | \mathcal { G } _ { t } | \leq \dot { K }$ . We compute average returns within each turn group: 

$$
\bar {G} _ {t} = \frac {1}{N _ {t}} \sum_ {j \in \mathcal {G} _ {t}} G _ {j, t}, \quad A _ {i, t} ^ {\text {G R P O}} = G _ {i, t} - \bar {G} _ {t}, \quad i \in \mathcal {G} _ {t}. \tag {3}
$$

Self-Inclusion Issue in GRPO We identify that the in-group mean baseline in Eq. (3) suffers from self-inclusion: $\bar { G } _ { t }$ includes $G _ { i , t }$ itself for any $i \in \mathcal G _ { t }$ . Since $G _ { i , t }$ depends on the current action $y _ { i , t }$ through rewards from turn t onward (i.e., $R _ { i , t : T } )$ ), the baseline can become action-dependent, violating the standard requirement for an unbiased REINFORCE baseline (Sutton et al., 1999; Sutton & Barto, 2018). For the mean-centering form, this manifests as a biased policy-gradient estimator within each prompt–turn group: 

$$
\mathbb {E} \left[ \hat {g} _ {\mathrm {G R P O}} \right] = \left(1 - \frac {1}{N _ {t}}\right) \nabla_ {\theta} J (\theta). \tag {4}
$$

That is, the update is systematically shrunk by a factor depending on the (effective) group size. We provide a detailed derivation in Appendix A. 

TRLOO To remove self-inclusion, we propose Turn-level REINFORCE Leave-One-Out (TRLOO), a multi-turn adaptation of the Leave-One-Out (Kool et al., 2019; Ahmadian et al., 2024) baseline. For each group $\mathcal { G } _ { t }$ and sample $i \in \mathcal { G } _ { t }$ with $N _ { t } > 1 .$ , define 

$$
\bar {G} _ {t} ^ {(- i)} = \frac {1}{N _ {t} - 1} \sum_ {j \in \mathcal {G} _ {t}, j \neq i} G _ {j, t}, \quad A _ {i, t} ^ {\mathrm {T R L O O}} = G _ {i, t} - \bar {G} _ {t} ^ {(- i)}. \tag {5}
$$

Equivalently, $\begin{array} { r } { A _ { i , t } ^ { \mathrm { T R L O O } } = \frac { N _ { t } } { N _ { t } - 1 } \left( G _ { i , t } - \bar { G } _ { t } \right) } \end{array}$ . Because $\bar { G } _ { t } ^ { ( - i ) }$ excludes $G _ { i , t } ,$ it does not depend on the current action $y _ { i , t }$ under independent rollouts, yielding an unbiased turn-level advantage estimator for multi-turn RL. 

Beyond unbiasedness, we claim TRLOO is beneficial for hard tasks with sparse positive rewards, where successful trajectories are rare. First, it avoids self-penalization in GRPO: under mean-centering, a rare high-return sample contributes to the $\bar { G } _ { t }$ and thus partially suppresses its advantage by subtracting the baseline. TRLOO excludes $G _ { i , t }$ from the baseline, so rare successes obtain a larger learning signal. Second, TRLOO is robust to varying group sizes. In multi-turn refinement, later turns may have fewer valid samples due to context limits or early termination, making 1N $\frac { 1 } { N _ { t } }$ larger in Eq. (4). TRLOO removes this self-inclusion effect and preserves the correct scale across varying group sizes, improving sample efficiency when positive feedback is scarce. 

# 4.3 Empirical Results

We report empirical results of multi-turn RL training under different design choices. We use the Qwen3-8B-Base (Team, 2025) model after training with our cold-start data. During training, we sample 16 rollouts per question. And each turn in a trajectory becomes a training sample (Baronio et al., 2025). We evaluate on KernelBench (Ouyang et al., 2025) Level $2 ^ { 1 }$ , whose difficulty is better matched to current LLM capabilities. Additional experimental details are provided in §6.1. We measure performance using the standard KernelBench metric Fast $@ 1$ and sample 8 candidates per question. 

Our default setting uses TRLOO with a maximum of 3 turns, enables Hacking Check, and sets $\gamma = 1 . 0$ for return computation; we denote this run as w/ TRLOO. To isolate the effect of each component, we compare against the following variants: w/o Hacking Check disables the hacking-check module in KERNELGYM while keeping other settings unchanged; w/ Single 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-09/0aa500ad-1825-4ed8-9d27-1d5a17dca497/32e4c2984fdba58a2b0d6ee0991df66b775de8d78f4f77827623439649176fa3.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-09/0aa500ad-1825-4ed8-9d27-1d5a17dca497/9b77900c67837291b88e281bdff6c72a2c6e28acb0e832e3f9a0ac52166022bf.jpg)



Figure 5: Left: Fast@1.2 at turn 3 over training steps. While MRS stabilizes training, profilingbased methods (PR and PRS) are required to significantly improve the stricter Fast@1.2 metric. Right: Entropy over training steps. While MRS improves training stability, PR and PRS further enhance stability on top of MRS. Additional training dynamics are shown in Figure 7.


Turn sets the maximum number of turns to $1 ; \gamma = 0$ sets the discount factor to zero to ablate reward-to-go credit assignment; and w/ GRPO replaces TRLOO with GRPO. 

Figure 4 shows that TRLOO under the default setting achieves the best overall performance. The variant without the hacking check shows that the hacking check is necessary for effective RL training. Otherwise, training would saturate after only $\sim 5 0$ steps. Compared to the GRPO variant, w/ TRLOO attains higher Fast@1 at every turn and exhibits a more stable learning curve, whereas GRPO saturates after roughly 200 training steps. Relative to singleturn training, multi-turn RL yields substantial gains as the number of turns increases, and it also improves first-turn quality. We attribute this to reward-to-go credit assignment: since early turns condition all subsequent refinements, they are incentivized to produce higher-quality intermediate kernels. This effect is corroborated by the $\gamma = 0$ ablation, which substantially degrades first-turn performance because the first-turn advantage no longer incorporates the impact on later interactions. Additionally, we observe that, unlike our model, the baseline AutoTriton fails to refine the kernel through multi-turn feedback. 

# 5 From Stability to Effectiveness: Overcoming Lazy Optimization

With KERNELGYM and TRLOO, long-term RL training becomes feasible, yet the “lazy optimization” issue persists as discussed in §2 and Figure 2. We systematically investigate this bottleneck through two hypotheses. Hypothesis 1: the saturation is caused by optimization instability arising from training–inference mismatch. Hypothesis 2: the optimization objective remains misaligned with meaningful speedup, incentivizing low-impact solutions. 

# 5.1 Hypothesis 1: Training Instability

Our initial speculation was that this premature saturation stemmed from training instability. Training–inference mismatch (Yao et al., 2025; Liu et al., 2025) is a fundamental challenge in RL for LLMs, where discrepancies between the rollout (inference) and training engines induce off-policy drift. Theoretically, this drift can lead to gradient variance and reward collapse, preventing the model from reaching higher performance peaks. 

To investigate this, we monitor training dynamics via entropy, gradient norms, and perplexity (Figure 7). As observed, the multi-turn RL for kernel generation run exhibits excessively high values across these metrics, clearly indicating training instability. Following Liu et al. (2025), we adopt geometric Mismatch Rejection Sampling (MRS) to mitigate this drift. We compute the geometric-mean importance ratio 

$$
w = \exp \left(\frac {1}{| T |} \sum_ {t \in T} \log \frac {\pi_ {\text {t r a i n}} \left(a _ {t} \mid s _ {t}\right)}{\pi_ {\text {r o l l o u t}} \left(a _ {t} \mid s _ {t}\right)}\right), \tag {6}
$$

retaining samples only if $w \in [ 0 . 9 9 9 , 1 . 0 0 1 ]$ . Additionally, we enforce a strict token-level veto: the entire sequence is rejected if the likelihood ratio $\dot { \pi } _ { \mathrm { t r a i n } } / \pi _ { \mathrm { r o l l o u t } }$ for any single token drops below $1 0 ^ { - 4 }$ . 

As visualized in Figure 7, MRS successfully stabilizes the training dynamics. However, Figure 5 reveals a critical insight: while mismatch correction prevents early collapse (smoothing the learning curve), it does not fundamentally lift the performance ceiling of Fast@1.2. This indicates that while Hypothesis 1 effectively accounts for the training instability, addressing it alone does not fully resolve the performance saturation. Consequently, this directs our focus to the optimization objective itself. 

# 5.2 Hypothesis 2: Misaligned Objective

Given that improving stability alone is insufficient, we speculate that the standard reward signal fails to distinguish between trivial improvements and meaningful bottlenecks. While a kernel may be correct and achieve some speedup, it might still fail to address the true performance bottlenecks. To move from producing merely correct kernels to effective ones, we must make rewards bottleneck-aware. 

Profiling-based Rewards (PR) A key symptom of this misalignment is the tendency for models to optimize trivial sub-operations (e.g., replacing a simple summation operation) without affecting the dominant bottlenecks in the computation. As demonstrated in the case study of lazy optimization versus better fusion (Figure 10), in the lazy optimization case, the model-generated kernel accounted for only $0 . 0 \bar { 1 } 4 \%$ of the total CUDA execution time, indicating that the kernel optimization did not affect the main bottlenecks. In contrast, with better fusion, the model generated kernels that covered $8 6 . 1 5 \%$ of the total CUDA runtime, resulting in better and more meaningful speedup. 

To formalize this intuition, we leverage the profiling toolkit in KERNELGYM to isolate the runtime contribution of the generated kernels (Tgenerated) from the overall CUDA execution time $( T _ { \mathrm { t o t a l } } )$ . We define the profiling ratio as: 

$$
\mathrm {P R} _ {i, t} = \frac {T _ {\text {g e n e r a t e d}}}{T _ {\text {t o t a l}}}. \tag {7}
$$

Intuitively, $\mathrm { P R } _ { i , t }$ assigns higher credit when the candidate optimizes kernels that dominate the end-to-end runtime. We then augment the per-turn reward with this signal (applied only to correct kernels): 

$$
R _ {i, t} = C \left(y _ {i, t}\right) + C \left(y _ {i, t}\right) \cdot \operatorname {s p e e d u p} _ {i, t} + C \left(y _ {i, t}\right) \cdot \Pr_ {i, t}. \tag {8}
$$

This encourages the model to focus on kernel optimizations that contribute significantly to performance, explicitly driving learning toward optimizations with larger real speedup. Besides, since $P R _ { i , t }$ is bounded in [0, 1], the speedup term naturally dominates, preventing the model from maximizing coverage via inefficient code. 

Profiling-based Rejection Sampling (PRS) Even with bottleneck-aware rewards, the exploration process can still be dominated by a high volume of low-impact (“lazy”) samples. To further filter the training distribution, we introduce profiling-based rejection sampling (PRS). For each sample $( i , t )$ , we retain it with probability: 

$$
p _ {i, t} = \operatorname {c l i p} \left(\frac {\mathrm {P R} _ {i , t} - \tau}{s}, 0, 1\right), \tag {9}
$$

where $\tau$ is a cutoff threshold and s controls the softness of the filter. In our experiments, we fix $\tau = 0 . 3$ and $s = 0 . 1$ . We ablate the design choice of PRS in Appendix D. 

# 5.3 Empirical results

Figure 5 confirms this staged diagnosis. MRS improves training stability but does not, by itself, raise the Fast $@ 1 . 2$ ceiling. Furthermore, adding PR and PRS substantially lifts Fast@1.2. And the stability is even further improved by PR and PRS as shown in Figure 5 (Right) and Figure 7. 


Table 1: Performance across levels under different Fast thresholds. As we treat samples with reward hacking as incorrect, our evaluation is more strict than the original Kernelbench. ∗The Cold-Start-8B refers to Qwen3-8B-Base after training with our cold-start data. We contain our DR. KERNEL-14B with sequential test-time scaling (STTS) using context management 6.3 as reference. DR. KERNEL-14B-STTS† reports the results from selecting the best turn across all history turns.


<table><tr><td rowspan="2">Model</td><td colspan="4">LEVEL1</td><td colspan="4">LEVEL2</td><td colspan="4">LEVEL3</td></tr><tr><td>Fast1</td><td>Fast1.2</td><td>Fast1.5</td><td>Fast2</td><td>Fast1</td><td>Fast1.2</td><td>Fast1.5</td><td>Fast2</td><td>Fast1</td><td>Fast1.2</td><td>Fast1.5</td><td>Fast2</td></tr><tr><td>GPT-5</td><td>19.5</td><td>16.5</td><td>12.5</td><td>11.0</td><td>46.7</td><td>28.6</td><td>13.1</td><td>3.0</td><td>21.0</td><td>12.0</td><td>4.0</td><td>2.0</td></tr><tr><td>Claude-4.5-Sonnet</td><td>15.5</td><td>13.5</td><td>11.0</td><td>8.5</td><td>50.0</td><td>26.7</td><td>9.2</td><td>1.8</td><td>21.0</td><td>11.0</td><td>5.0</td><td>4.0</td></tr><tr><td>Deepseek-V3.2-Thinking</td><td>7.5</td><td>5.5</td><td>4.5</td><td>4.0</td><td>11.0</td><td>6.5</td><td>2.5</td><td>0.5</td><td>2.0</td><td>1.0</td><td>0.0</td><td>0.0</td></tr><tr><td>GLM-4.7</td><td>19.4</td><td>17.2</td><td>13.1</td><td>10.4</td><td>30.0</td><td>20.5</td><td>8.5</td><td>3.5</td><td>5.0</td><td>2.0</td><td>2.0</td><td>2.0</td></tr><tr><td>Qwen3-8B</td><td>5.8</td><td>4.8</td><td>4.1</td><td>3.4</td><td>13.0</td><td>5.6</td><td>2</td><td>1.1</td><td>5.7</td><td>0.2</td><td>0.0</td><td>0.0</td></tr><tr><td>Qwen3-32B</td><td>6.1</td><td>4.9</td><td>4.3</td><td>4</td><td>14.0</td><td>9.4</td><td>2.4</td><td>0.2</td><td>3.5</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Qwen3-Coder-A30BA3</td><td>6.0</td><td>5.2</td><td>5.1</td><td>3.8</td><td>12.6</td><td>5.0</td><td>1.5</td><td>0.3</td><td>7.0</td><td>1.0</td><td>0.0</td><td>0.0</td></tr><tr><td>AutoTriton</td><td>4.5</td><td>3.6</td><td>2.8</td><td>2.1</td><td>30.6</td><td>9.2</td><td>2.6</td><td>0.5</td><td>7.5</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Cold-Start-8B*</td><td>7.5</td><td>6.6</td><td>5.0</td><td>4.3</td><td>8.8</td><td>5.6</td><td>1.8</td><td>0.4</td><td>0.5</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>DR. KERNEL-8B</td><td>15.9</td><td>12.8</td><td>10.9</td><td>8.4</td><td>46.0</td><td>20.0</td><td>5.0</td><td>1.5</td><td>10.8</td><td>1.0</td><td>0.0</td><td>0.0</td></tr><tr><td>DR. KERNEL-14B</td><td>20.3</td><td>16.9</td><td>13.2</td><td>11.6</td><td>49.2</td><td>25.6</td><td>7.4</td><td>2.1</td><td>8.8</td><td>1.2</td><td>0.2</td><td>0.0</td></tr><tr><td>DR. KERNEL-14B-STTS</td><td>24.1</td><td>18.8</td><td>15.3</td><td>12.8</td><td>59.8</td><td>31.6</td><td>9.6</td><td>3.0</td><td>17.1</td><td>3.0</td><td>0.2</td><td>0.0</td></tr><tr><td>DR. KERNEL-14B-STTS†</td><td>39.3</td><td>25.1</td><td>20.4</td><td>17.6</td><td>80.9</td><td>47.8</td><td>23.6</td><td>12.5</td><td>29.8</td><td>7.3</td><td>0.5</td><td>0.0</td></tr></table>

# 6 Experiments

# 6.1 Setup

We evaluate on KernelBench (Ouyang et al., 2025) across all three levels. We follow the official Torch backend in Kernelbench and their implementations of correctness and speedup measurement. We furhter conduct hacking checks when we evaluate kernels. Therefore, our evaluations are stricter than the original Kernelbench. We follow the standard metrics in Kernelbench Fast@p, $p = [ 1 , 1 . 2 , 1 . 5 , 2 ] .$ , where it is the ratio of samples are both correct and achieve $\times p$ speedup. Both the evaluations and training are conducted on NVIDIA H100. We experiment with Qwen3-8B-Base and Qwen-14B-Base. We sample each question for 8 samples with 3 max turns. We set the max tokens as 32768 and the max generated tokens per turn as 8192. To keep a fair comparison, we report results at turn 3 for our models and for most baselines, as turn 3 typically yields the best average performance. For baselines whose best average performance is achieved at an earlier turn, we instead report their best-performing turn. 

We first perform cold-start supervised fine-tuning on our collected 8K 5-turn trajectories, using a learning rate of $1 \times 1 0 ^ { - 6 }$ , batch size 256, for 4 epochs. After cold-start training, we run multi-turn RL on the RL queries from cudaLLM (CudaLLM Team, 2025). The queries used for both SFT and RL cover basic PyTorch operators, Transformer components, more complex compositions, and LLM-generated tasks. For RL, we use a learning rate of $1 \times 1 0 ^ { - 6 }$ , train for 300 rollout steps, sample 16 rollouts per prompt with max turns to 3 and a rollout batch size of 16. We implement the training pipeline with asynchronous inference. 

# 6.2 Results

Main Results. Table 1 summarizes performance across KernelBench levels under different Fast thresholds. We compare DR. KERNEL against AutoTriton (Li et al., 2025), open-source models with strong coding/reasoning abilities, and proprietary models. AutoTriton is a very relevant baseline since it is released, Triton-based, and reports Fast@p. 

Overall, DR. KERNEL achieves the strongest performance among open-source baselines and is competitive with frontier models on Level 1 and Level 2. In particular, DR. KERNEL-14B attains high Fas $@ 1 . 2$ on both Level 1 and Level 2, indicating that it improves not only any speedup (Fast@1) but also the stricter and meaningful speedup. This contrasts with prior 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-09/0aa500ad-1825-4ed8-9d27-1d5a17dca497/898b37d344a82981e58bc80cb745b6bb1921126778980916699578a59f05f9dd.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-09/0aa500ad-1825-4ed8-9d27-1d5a17dca497/c1c5ea83c4d086e1a4efd9cb862c8e205ee0a2a2c3762775a7da95c148abac64.jpg)



Figure 6: Test-time scaling with DR. KERNEL-14B. We report Fast@1.2 for the last turn (left) and best-of-history (right). Vanilla extrapolation increases the number of turns by appending all previous turns to the prompt. Context management stores the full history externally, but only includes the top- $_ w$ turns (by reward, $w { = } 4$ ) in the prompt to fit context length.


approaches such as AutoTriton, which achieves a strong Fast@1 on Level 2 but delivers substantially smaller gains under stricter thresholds (e.g., Fast@1.2). 

Comparing against our cold-start model, DR. KERNEL shows that multi-turn RL contributes materially to performance gains, especially on the stricter metrics (e.g., Fast@1.2 improves from $5 . 6 \overset { \cdot } {  } 2 \overset { \cdot } { 0 } . 0$ on Level 2). While DR. KERNEL improves Level 3 Fast@1 relative to opensource baselines, performance at stricter thresholds on Level 3 remains limited, suggesting that further scaling of training data and model capacity is likely required to close the gap to frontier models on the hardest subset. 

Finally, test-time scaling further amplifies DR. KERNEL’s performance. With sequential testtime scaling (STTS) via context management (§6.3), DR. KERNEL-14B-STTS boosts Fast $@ 1 . 2$ from $1 6 . 9 \stackrel { \smile } { \to } 1 8 . 8$ on Level 1 and from $2 5 . 6  3 1 . 6$ on Level 2; with best-turn selection across history turns (DR. KERNEL-14B-STTS†), Fast@1.2 further rises to 25.1 (Level 1) and 47.8 (Level 2), surpassing frontier models such as GPT-5 and Claude-4.5-Sonnet on these metrics. STTS also materially improves Level 3 (Fast@1.2: $1 . 2  3 . 0 / 7 . 3 )$ , narrowing the gap and partially compensating for the hardest subset; given the substantial model-size disparity to frontier models, these test-time scaled results remain informative despite leveraging TTS. 

Analysis We conduct a comprehensive analysis that includes reward hacking ratio (Appendix C) and case studies (Appendix E.2). Please refer to these sections for details. 

# 6.3 Test-time Scaling for Kernel Generation

We study sequential test-time scaling (STTS) by increasing the number of multi-turn refinement steps at inference time. We use DR. KERNEL-14B with a maximum context length of 32,768 tokens and evaluate two strategies: vanilla extrapolation and context management. We report two metrics: (i) Last-turn Fast@1.2, i.e., the Fast@1.2 achieved by the final generated kernel at turn $T _ { \cdot }$ ; and (ii) Best-of-history Fast@1.2, i.e., the best Fast@1.2 obtained among turns $\{ 1 , \ldots , T \}$ . With STTS, our model even outperforms GPT-5/Claude-4.5-Sonnet in Kernelbench level-2 subset. 

Vanilla extrapolation We directly extrapolate the number of refinement turns beyond training (trained with up to 3 turns) by appending the entire interaction history to the prompt at each turn. Figure 6 shows that increasing turns initially improves both last-turn and best-of-history performance. However, as $T$ grows, prompt length scales linearly and may approach the context limit, which can degrade performance. 

Context management To scale $T$ without unbounded prompt growth, we store all turns in an external memory and maintain a fixed in-context window. Concretely, at each turn we select the top- $_ { \cdot w }$ turns with the highest rewards from the accumulated history and only 


Table 2: Fast performance across levels under different thresholds (evaluated under torch.compile). Since torch.compile provides a strong optimized baseline, Fast@1 remains meaningful (unlike eager mode, where trivial “lazy” optimizations can inflate Fast@1).


<table><tr><td rowspan="2">Model</td><td colspan="4">LEVEL1</td><td colspan="4">LEVEL2</td><td colspan="4">LEVEL3</td></tr><tr><td>Fast1</td><td>Fast1.2</td><td>Fast1.5</td><td>Fast2</td><td>Fast1</td><td>Fast1.2</td><td>Fast1.5</td><td>Fast2</td><td>Fast1</td><td>Fast1.2</td><td>Fast1.5</td><td>Fast2</td></tr><tr><td>GPT-5</td><td>18.6</td><td>8.0</td><td>6.5</td><td>5.5</td><td>22.1</td><td>3.6</td><td>1.5</td><td>1.0</td><td>14.0</td><td>4.0</td><td>3.0</td><td>1.0</td></tr><tr><td>Claude-4.5-Sonnet</td><td>10.0</td><td>2.2</td><td>2.0</td><td>1.8</td><td>20.5</td><td>3.0</td><td>0.0</td><td>0.0</td><td>12.0</td><td>3.5</td><td>0.5</td><td>0.0</td></tr><tr><td>DR. KERNEL-8B</td><td>16</td><td>3.0</td><td>1.5</td><td>8.4</td><td>20.6</td><td>0.8</td><td>0.0</td><td>0.0</td><td>7.2</td><td>2.3</td><td>0.0</td><td>0.0</td></tr><tr><td>DR. KERNEL-14B</td><td>17.8</td><td>5</td><td>3.3</td><td>2.5</td><td>23.5</td><td>1.9</td><td>0.0</td><td>0.0</td><td>9.2</td><td>3</td><td>0.0</td><td>0.0</td></tr></table>

append these selected turns as the prompt history for generating the next turn (we use $w { = } 4 )$ ). As shown in Figure 6, context management yields consistently stronger best-ofhistory performance and continues to improve as turns scale. Last-turn performance can be slightly lower at small $T _ { \iota }$ , since vanilla extrapolation can condition on the full history, but as $T$ increases, context management becomes strictly more reliable and surpasses the best performance achievable by vanilla extrapolation. 

# 6.4 Results on torch.compile

torch.compile is an advanced PyTorch feature that captures PyTorch programs into a compiled computation graph and optimizes execution via operator fusion, code generation, and scheduling, among other compiler passes. While most prior work evaluates modelgenerated kernels only under Torch eager execution, we further validate both our models and frontier models under torch.compile, providing a substantially stronger and more practical assessment of speedups. 

As shown in Table 2, DR. KERNEL remains effective under the more challenging torch.compile setting and stays competitive with frontier models across the three levels. Because torch.compile already applies compiler optimizations, the headroom for additional gains is smaller than in eager mode; consequently, the absolute Fast@p numbers are generally lower for all models, including both ours and frontier models. Importantly, Fast@1 under torch.compile is also a stricter target: trivial “lazy” changes that may yield marginal improvements in eager execution typically do not surpass the optimized compiled baseline. These results suggest that our training methods generalize beyond eager-mode artifacts, and further scaling of data and model capacity is a promising direction to obtain larger improvements even on top of torch.compile. 

# 7 Conclusion

We investigate RL for Triton kernel generation and identify the challenges of reward hacking and lazy optimization, which are common in prior works. To address these, we develop the KERNELGYM environment with hacking checks and profiling tools, propose unbiased multiturn RL methods incorporating mismatch correction, profiling-based rewards/rejection sampling, and explore sequential test-time scaling. Our approach enhances meaningful speedup, advancing RL training for kernel generation. 

# 8 Limitations and Future Work

While our approach demonstrates progress in RL-based training for Triton kernel generation, we acknowledge several areas that warrant further investigation. 

Data Scaling and Pre-training From a data perspective, resource constraints limited our supervised fine-tuning (SFT) phase to 8,000 cold-start samples. Given the relative scarcity of high-quality kernel programming data in the pre-training corpora of current Large Language Models (LLMs), our results suggest that the ”data floor” for this domain is quite 

high. Future work could involve larger-scale data collection to facilitate domain-specific pre-training or continual pre-training (middle-training), which would also provide a more robust foundation for subsequent RL optimization. 

Model Capacity Our observations with DR. KERNEL-8B and DR. KERNEL-14B confirm that larger models possess a superior capacity for kernel generation. This scaling effect is particularly critical in Reinforcement Learning, where the model must rely on its own generations to explore the solution space and update its policy. We expect that migrating these methods to even larger parameter scales will accelerate the development. 

Path Toward Production-Ready Automation Although our methods achieve performance improvements that rival or exceed frontier models, the field remains in an exploratory stage. While current models can generate high-quality code snippets, they are not yet capable of fully autonomous, end-to-end kernel generation for production environments. We hope that our contributions, specifically the KERNELGYM environment and the DR. KERNEL training framework, serve as a catalyst for future research. 

# References



Arash Ahmadian, Chris Cremer, Matthias Galle, Marzieh Fadaee, Julia Kreutzer, Olivier ´ Pietquin, Ahmet Ust ¨ un, and Sara Hooker. Back to basics: Revisiting reinforce style ¨ optimization for learning from human feedback in llms, 2024. URL https://arxiv.org/ abs/2402.14740. 





Carlo Baronio, Pietro Marsella, Ben Pan, Simon Guo, and Silas Alberti. Kevin: Multi-turn rl for generating cuda kernels, 2025. URL https://arxiv.org/abs/2507.11948. 





CudaLLM Team. CudaLLM: Training language models to generate high-performance cuda kernels. https://github.com/ByteDance-Seed/cudaLLM, 2025. GitHub repository. Latest commit Aug 18, 2025 (commit 7d280a6). Accessed 2026-01-20. 





Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations (ICLR), 2024. 





Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning, 2025. URL https://arxiv.org/abs/2503.09516. 





Wouter Kool, Herke van Hoof, and Max Welling. Buy 4 REINFORCE samples, get a baseline for free!, 2019. URL https://openreview.net/forum?id=r1lgTGL5DE. 





Shangzhan Li, Zefan Wang, Ye He, Yuxuan Li, Qi Shi, Jianling Li, Yonggang Hu, Wanxiang Che, Xu Han, Zhiyuan Liu, et al. Autotriton: Automatic triton programming with reinforcement learning in llms. arXiv preprint arXiv:2507.05687, 2025. 





Jiacai Liu, Yingru Li, Yuqian Fu, Jiawei Wang, Qian Liu, and Yu Shen. When speed kills stability: Demystifying RL collapse from the training-inference mismatch, September 2025. URL https://richardli.xyz/rl-collapse. 





Alexander Novikov, Ngan V ˆ u, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, ˜ Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco J. R. Ruiz, Abbas Mehrabian, M. Pawan Kumar, Abigail See, Swarat Chaudhuri, George Holland, Alex Davies, Sebastian Nowozin, Pushmeet Kohli, and Matej Balog. Alphaevolve: A coding agent for scientific and algorithmic discovery, 2025. URL https: //arxiv.org/abs/2506.13131. 





Anne Ouyang, Simon Guo, Simran Arora, Alex L Zhang, William Hu, Christopher Re, and Azalia Mirhoseini. Kernelbench: Can LLMs write efficient GPU kernels? In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum? id=yeoN1iQT1x. 





Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018. URL http://incompleteideas.net/book/the-book-2nd. html. 





Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In S. Solla, T. Leen, and K. Muller (eds.), ¨ Advances in Neural Information Processing Systems, volume 12. MIT Press, 1999. URL https://proceedings.neurips.cc/paper files/paper/1999/file/ 464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf. 





Qwen Team. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388. 





Lei Wang, Yu Cheng, Yining Shi, Zhengju Tang, Zhiwen Mo, Wenhao Xie, Lingxiao Ma, Yuqing Xia, Jilong Xue, Fan Yang, and Zhi Yang. Tilelang: A composable tiled programming model for ai systems, 2025. URL https://arxiv.org/abs/2504.17577. 





Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, and Amelia Glaese. Browsecomp: A simple yet challenging benchmark for browsing agents, 2025. URL https: //arxiv.org/abs/2504.12516. 





Jiin Woo, Shaowei Zhu, Allen Nie, Zhen Jia, Yida Wang, and Youngsuk Park. Tritonrl: Training llms to think and code triton without cheating, 2025. URL https://arxiv.org/ abs/2510.17891. 





Feng Yao, Liyuan Liu, Dinghuai Zhang, Chengyu Dong, Jingbo Shang, and Jianfeng Gao. Your efficient rl framework secretly brings you off-policy rl training, August 2025. URL https://fengyao.notion.site/off-policy-rl. 





Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen, Baris Kasikci, Vinod Grover, Arvind Krishnamurthy, and Luis Ceze. Flashinfer: Efficient and customizable attention engine for llm inference serving. arXiv preprint arXiv:2501.01005, 2025. URL https://arxiv.org/abs/2501.01005. 



# A Derivation: Self-Inclusion Causes a Scaled Gradient in GRPO

Setup For a given prompt question, fix a turn group $\mathcal { G } _ { t }$ with $| \mathcal { G } _ { t } | = N$ . For each rollout $i \in \mathcal { G } _ { t } ,$ let $s _ { i , t } \triangleq ( h _ { i , : t - 1 } , x _ { i , t } )$ and $y _ { i , t } \sim \pi _ { \theta } ( \cdot \mid s _ { i , t } ) ,$ , where $h _ { i , : t - 1 }$ is all turn history before $t , x _ { i , t }$ is the prompt with environmental feedback in turn t. Let $G _ { i , t }$ be the reward-to-go return at turn $t$ and define the in-group mean $\begin{array} { r } { \bar { G } _ { t } = \frac { 1 } { N } \sum _ { j \in \mathcal { G } _ { t } } G _ { j , t } } \end{array}$ . GRPO mean-centering uses $A _ { i , t } ^ { \mathrm { G R P O } } = G _ { i , t } - \bar { G } _ { t }$ . 

For any random variable $Z$ that does not depend on the sampled action $y _ { i , t }$ (conditional on $s _ { i , t } )$ , 

$$
\mathbb {E} _ {y _ {i, t} \sim \pi_ {\theta} (\cdot | s _ {i, t})} \left[ \nabla_ {\theta} \log \pi_ {\theta} \left(y _ {i, t} \mid s _ {i, t}\right) \cdot Z \right] = Z \cdot \mathbb {E} _ {y _ {i, t}} \left[ \nabla_ {\theta} \log \pi_ {\theta} \left(y _ {i, t} \mid s _ {i, t}\right) \right] = 0, \tag {10}
$$

where we use $\begin{array} { r } { \mathbb { E } _ { y } [ \nabla _ { \theta } \log \pi _ { \theta } ( y \mid s ) ] = \nabla _ { \theta } \int \pi _ { \theta } ( y \mid s ) d y = \nabla _ { \theta } 1 = 0 . } \end{array}$ 

Policy Gradient in GRPO Consider the per-turn GRPO estimator within this group: 

$$
\hat {g} = \frac {1}{N} \sum_ {i \in \mathcal {G} _ {t}} \nabla_ {\theta} \log \pi_ {\theta} \left(y _ {i, t} \mid s _ {i, t}\right) \left(G _ {i, t} - \bar {G} _ {t}\right). \tag {11}
$$

Taking expectation and expanding the baseline term, 

$$
\begin{array}{l} \mathbb {E} [ \hat {g} ] = \frac {1}{N} \sum_ {i} \mathbb {E} [ \nabla_ {\theta} \log \pi_ {\theta} (y _ {i, t} | s _ {i, t}) G _ {i, t} ] - \frac {1}{N} \sum_ {i} \mathbb {E} [ \nabla_ {\theta} \log \pi_ {\theta} (y _ {i, t} | s _ {i, t}) \bar {G} _ {t} ] \\ = \frac {1}{N} \sum_ {i} \mathbb {E} \left[ \nabla_ {\theta} \log \pi_ {\theta} \left(y _ {i, t} \mid s _ {i, t}\right) G _ {i, t} \right] - \frac {1}{N} \sum_ {i} \mathbb {E} \left[ \nabla_ {\theta} \log \pi_ {\theta} \left(y _ {i, t} \mid s _ {i, t}\right) \frac {1}{N} \sum_ {j} G _ {j, t} \right]. \tag {12} \\ \end{array}
$$

For $j \neq i , G _ { j , t }$ is independent of $y _ { i , t }$ under independent rollouts, hence we can apply the score-function identity in Eq. (10) to obtain 

$$
\mathbb {E} \left[ \nabla_ {\theta} \log \pi_ {\theta} (y _ {i, t} \mid s _ {i, t}) G _ {j, t} \right] = 0 \qquad (j \neq i).
$$

Therefore only the $j = i$ term remains: 

$$
\begin{array}{l} \mathbb {E} [ \hat {g} ] = \frac {1}{N} \sum_ {i} \mathbb {E} \left[ \nabla_ {\theta} \log \pi_ {\theta} \left(y _ {i, t} \mid s _ {i, t}\right) G _ {i, t} \right] - \frac {1}{N} \sum_ {i} \mathbb {E} \left[ \nabla_ {\theta} \log \pi_ {\theta} \left(y _ {i, t} \mid s _ {i, t}\right) \frac {1}{N} G _ {i, t} \right] \\ = \left(1 - \frac {1}{N}\right) \cdot \frac {1}{N} \sum_ {i} \mathbb {E} \left[ \nabla_ {\theta} \log \pi_ {\theta} \left(y _ {i, t} \mid s _ {i, t}\right) G _ {i, t} \right]. \tag {13} \\ \end{array}
$$

The last term corresponds to the unbiased REINFORCE gradient for this prompt–turn group (up to the outer averaging convention), hence the GRPO mean baseline induces a shrinkage factor $\begin{array} { r } { ( 1 - \frac { 1 } { N } ) } \end{array}$ due to self-inclusion. 

Leave-one-out removes self-inclusion. Using the leave-one-out baseline G¯ (−i)t $\bar { G } _ { t } ^ { ( - i ) } ~ =$ $\begin{array} { r } { \frac { 1 } { N - 1 } \sum _ { j \neq i } G _ { j , t } , } \end{array}$ the baseline term no longer contains $G _ { i , t }$ and is independent of $y _ { i , t }$ under independent rollouts. By Eq. (10), the expected contribution of the baseline term becomes zero, yielding an unbiased estimator. 

# B Training Dynamics

In this section, we analyze the training dynamics. As shown in Figure 7, multi-turn RL for kernel generation exhibits significant instability, characterized by elevated entropy, perplexity, and gradient norms, even when using unbiased advantage estimation. Incorporating mismatch correction (i.e., Mismatch Rejection Sampling) effectively stabilizes the training process. Furthermore, the introduction of PR and PRS provides additional stability, leading to smoother training. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-09/0aa500ad-1825-4ed8-9d27-1d5a17dca497/ff7fbe4767233746ea3bea1879dc377f71035c9e1d2faed03f6c9871ad3a6dd8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-09/0aa500ad-1825-4ed8-9d27-1d5a17dca497/fc7c877e41e4a583051726086d6edf8b3147e8514556f75eb1f1274ed859c2db.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-09/0aa500ad-1825-4ed8-9d27-1d5a17dca497/76c7a49acd79ebc3964528f4fde62c18b5bab0d36b1b7720f9da6c4955a408c1.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-09/0aa500ad-1825-4ed8-9d27-1d5a17dca497/b315145c210fee7ff8e0f68d9403e271c2bb7de259d8e276974dbe15ed46395c.jpg)



Figure 7: The training dynamics of TRLOO, TRLOO $^ +$ Mismatch Rejection Sampling (MRS), TRLOO + MRS + Profiling-based Reward (PR) and $\mathrm { T R L O O } + \mathrm { M R S } + \mathrm { P R } + 1$ Profiling-based Rejection Sampling (PRS). We analyze the training dynamics via the lens of entropy, gradient norm, VLLM-PPL, and FSDP-PPL, which are also monitored by Liu et al. (2025).


# C Hacking Ratio

We analyze the changes in the hacking ratio during training for DR. KERNEL-14B on the Kernelbench level-2 subset. With the hacking check in KERNELGYM, the hacking ratio steadily decreases from approximately $2 0 \%$ at the start to around $3 \%$ . We also examine the hacking ratio on the Kernelbench level-1 subset; compared to AutoTriton, which exhibits a hacking ratio of around $1 0 \%$ , DR. KERNEL-14B experiences hacking in only $1 . 7 \%$ of cases. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-09/0aa500ad-1825-4ed8-9d27-1d5a17dca497/61d3ffa93bb93cf41c871269424ace38516dbbc4df7de4551c3a9a74c734dc14.jpg)



Figure 8: The hacking ratio of DR. KERNEL-14B. With the hacking check in KERNELGYM, the hacking ratio decreases from approximately $2 0 \%$ at the start to only around $3 \%$ in the Kernelbench level-2 subset.


# D Ablations of PRS

We perform an ablation study to evaluate the effect of the softness sampling design choice in PRS. Our default setting for DR. KERNEL is the combination of TRLOO, MRS, PR, and 

PRS, and we compare it to a variant, DR. KERNELw/o s in PRS, where kernels with $P R \ge \tau$ are kept directly, and kernels with $P R < \tau$ are discarded outright. In contrast, in the variant with softness, kernels with $P R \in [ 0 . 3 , 0 . 4 )$ are probabilistically retained, based on PR, τ, and s. In this ablation, we set $\tau = 0 . 3$ as the fixed configuration. 

As shown in Figure 9, DR. KERNEL outperforms the variant w/o s in PRS. The variant still performs better and shows improved stability compared to the baseline w/o PR & PRS. This ablation demonstrates that the softness sampling mechanism enhances the robustness of PRS, helping it balance correctness and meaningful speedup by selectively retaining relatively low-quality samples that might otherwise be discarded. This ability to retain such samples contributes to a more effective exploration-exploitation trade-off and facilitates more stable kernel generation over time. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-09/0aa500ad-1825-4ed8-9d27-1d5a17dca497/c5c030db4bd02a5dea5d81606d17d3e341b1e8b5314ddd3a4611a51800305e76.jpg)



Figure 9: Comparison of DR. KERNEL with and without softness in PRS. DR. KERNEL outperforms the variant w/o s in PRS, while both variants show better stability compared to the baseline w/o PR & PRS.


# E Cases Studies

# E.1 Lazy Optimization vs. Better Fusion

We show the cases of profiling feedback from lazy optimization and better fusion cases. As shownin Figure 10, in the lazy optimization case, where only a trivial summation operation is replaced, the model-generated kernel accounts for only $0 . 0 1 4 \%$ of the total CUDA execution time. In contrast, with better fusion, the model generates more meaningful kernels, achieving significantly better speedup and increasing the CUDA runtime fraction to $8 6 . 1 5 \%$ of the total runtime. 

# E.2 Trajectories from DR. KERNEL

We conduct qualitative studies on DR. KERNEL-14B using multi-turn inference across three turns. As shown in Figure 11, in the first turn, DR. KERNEL identifies the LayerNorm operation and generates a kernel by fusing different operations into a single Triton kernel. After receiving feedback from KERNELGYM, it recognizes that certain configurations, such as block size, number of wraps, and stages, are under-explored, and applies ”autoconfig” to select better configurations. By the third turn, DR. KERNEL identifies the optimal configuration based on the running hardware, further improving the performance by adjusting the configuration in ”autoconfig.” This case demonstrates that after RL training, DR. KERNEL-14B handles basic kernel writing and also adapts effectively to environment feedback, showcasing its ability to improve over time. 

# E.3 Case for Better Fusion

We further study the case where the lazy optimization issue is alleviated, as shown in the profiling summary in Figure 10 (right). In contrast to the lazy optimization case, the example generated by DR. KERNEL-8B addresses lazy optimization by converting most operations 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-09/0aa500ad-1825-4ed8-9d27-1d5a17dca497/526274e9f15cb874e62507039d9cc3503adc7e6ca81a8eef8ba80f495911d43a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-09/0aa500ad-1825-4ed8-9d27-1d5a17dca497/914a58c5deccf7c78bfb5b92da367e66935f0d6cf188e0276d0e5549ecbb750a.jpg)



Figure 10: Profiling feedback for cases with lazy optimization and better fusion. We omit some profiling items for brevity. The original code for Profiling Lazy Optimization.json is shown in Figure 2 (right), and the original code for Profiling Better Fusion.json can be found in Appendix E.3. In the lazy optimization case, where only a trivial summation operation is replaced, the model-generated kernel accounts for only $0 . 0 1 4 \%$ (i.e $P R =$ 0.00014) of the total CUDA execution time. In contrast, with better fusion, the model generates more meaningful kernels, achieving significantly better speedup and increasing the CUDA runtime fraction to $8 6 . 1 5 \%$ i.e $P R \stackrel { \sim } { = } 0 . 8 6 1 5$ of the total runtime.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-09/0aa500ad-1825-4ed8-9d27-1d5a17dca497/3a724a7602a43d577ec000cfe27afb0fb749298c8e02defe3735479e01aa9f30.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-09/0aa500ad-1825-4ed8-9d27-1d5a17dca497/44380a6304a5234cccfb49c01b7a20aa850aadc964f4e004a3a8fe02fa5ad96f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-09/0aa500ad-1825-4ed8-9d27-1d5a17dca497/43c501063257ea4811a3b60f1e0a810693e18178e0f64b2e52f4e75447d4b451.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-09/0aa500ad-1825-4ed8-9d27-1d5a17dca497/b0bf27591e6e426cf16a8db36668ad3ae0203dced71cd9827feeb374b6085ab2.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-09/0aa500ad-1825-4ed8-9d27-1d5a17dca497/be678d2f4d4310d48fbac2a5bb32a56487f85e3e868caa1ae46bb9c76ef52099.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-09/0aa500ad-1825-4ed8-9d27-1d5a17dca497/0cc8b7d612898d178b5152891b3f1742b5ec9e7c1ae725289bcf1e6bffd28b64.jpg)



Figure 11: The case study of DR. KERNEL-14B in LayerNorm operation from Kernelbench Level-1 subset. The speedup of the three turns are 1.04, 1.21, and 1.45, respectively. In this case, DR. KERNEL-14B effectively handles basic kernel writing and adapts to environmental feedback over multiple turns.


into Triton kernels, which account for $8 6 . 1 5 \%$ of the total CUDA runtime. However, we note that a convolution operation remains in the implementation. During training, we observed that the model attempted to implement this operation with Triton. Despite Triton’s potential for kernel optimization, convolution operations are difficult to implement effectively, as they are highly optimized by libraries such as cuDNN. Consequently, our small-sized models with limited data struggle to implement this operation as effectively as cuDNN, and instead, the model focuses on fusing other kernels. This behavior highlights one of the root causes of lazy optimization and emphasizes the importance of a well-established optimization objective in RL-based kernel generation, such as a bottleneck-aware profiling-based reward. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-09/0aa500ad-1825-4ed8-9d27-1d5a17dca497/3b43b9339df797500a5f980f1b37bd7f4802ad2e5a7f5b83e0ac2cc9010f32fb.jpg)



Figure 12: The case study of better fusion from DR. KERNEL-8B, which corresponds to the profiling feedback in Figure 10 (right).


# F Prompt Template

# F.1 Prompt Template for Cold-Start Data Distillation

We show the template for cold-start data distillaion in Figure 13. 

# F.2 Prompt Template for SFT and RL

We show the template for both SFT and RL in Figure 14. 

# 1st Turn Prompt Template for Training

You are looking at this PyTorch code and thinking it could be optimized with Triton. 

Here's the PyTorch code: 

```python 

{reference_code} 

You need to create a Triton version with the entry point called `{entry_point}New`. 

Please firstly analyze this code and think hard how you can optimize it. 

**Please output and show your thinking, plan, analysis etc., before your coding, which should be as more as possible.** 

# Prompt Template After 1st Turn

Server feedback from the evaluation environment for your last implementation: {kernelgym_feedbacks} 

Based on the above server feedback, please improve the implementation: 

- If there are errors/crashes/illegal memory access: identify the root cause and fix it; prevent recurrence. 

- If there is no speedup or performance regresses: optimize the bottlenecks to achieve a clear speedup. 

- If there is already a speedup: further improve performance without degrading correctness. 

- Please output your thinking, plan, analysis, and the final code. 

Figure 13: The prompt template for cold-start data distillation. 

# 1st Turn Prompt Template for Training

[user] You write custom Triton kernels to replace the pytorch operators in the given architecture to get speedups. 

You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom Triton kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination. 

Here's an example to show you the syntax of inline embedding custom Triton kernels in torch: The example given architecture is: 

{example_ref} 

The example new arch with custom Triton kernels looks like this: 

You are given the following architecture: 

{ref_code} 

Optimize the architecture named Model with custom Triton operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Let's think step by step. 

# Example Kernel Code

import torch 

import torch.nn.functional as F 

@triton.jit 

def add_kern 

x_ptr, # Pointer to first input 

n_elements, # Total number of elements 

B): 

# Each program handles a contiguous block of data of size BLOCK_SIZE 

block_start = tl.program_id(0) * BLOCK_SIZE 

# Mask to ensure we don't go out of bounds 

mask = offsets < n_elements 

# Load input valuesx = tl.load(x_ptr + offsets, mask=mask, other 

y = tl.load(y_ptr + offsets, mask=ma 

# Perform the elem entwise addition 

out = x + y 

tl.store(out_ptr + offsets, out, mask=mask) 

def triton_add(x: torch.Tensor, y: torch.Tensor) 

This function wraps the Triton kernel call. It: 

1. Ensures the inputs are contiguous on GPU 

2. Calculates the grid (blocks) 

3. Launches the Triton kernel. 

"""assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA." 

x = x.contiguous() 

# Prepare output tensor 

out = torch.empty_like(x) 

# Number of elements in the tensor 

n_elements = x.numel() 

# Determine the number of blocks needed 

# Launch the Triton kernel 

add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE) 

class ModelNew(nn.Module): 

def forward(self, a, b): 

# Example Reference Code

import torch 

import torch.nn as nn 

import torch.nn.functional as F 

dlassModelmr 

superl)- 

def forward(self, a, 

return a + b 

detsetirutsll 

def get_ 

# randomly generate input tena = torch.randn(1, 128).cuda() 

b = torch.randn(1, 128).cuda() 

return [a, b] 

def get_init_inputs(): 

de#gendiominputeratetensorsreeuiredforinitializationbasedonthe modelarchitecture 

return [] 

# Prompt Template for Training After 1st Turn

Now you have received the server feedback for your last 

implementation. Based on that and all your previous responses, 

improve the implementation. 

the implementation: 

the Impieme 

{KernelGYM Feedback} 

Return an improved Triton implementation named `ModelNew` as a 

Figure 14: The prompt template for both SFT and RL. The task instruction, example code and reference code are shown in the first-turn prompt. And for the later turns, prompt asks model to refine the kernel implementaion based on all hisotory information and KERNELGYM feedbacks. 