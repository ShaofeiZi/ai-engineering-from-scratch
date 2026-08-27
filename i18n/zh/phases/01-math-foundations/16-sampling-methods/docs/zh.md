# 采样方法

> 采样是 AI 探索可能性空间的方式。

**Type:** 构建
**Language:** Python
**Prerequisites:** 第 1 阶段，第 06–07 课（Probability, Bayes' Theorem）
**Time:** 约 2 小时

## 学习目标

- 仅使用均匀随机数，从零实现逆 CDF 采样、拒绝采样和重要性采样
- 为语言模型 token 生成实现温度采样、top-k 采样和 top-p（nucleus）采样
- 解释重参数化技巧，以及它为何能让 VAE 通过采样过程进行反向传播
- 运行 Metropolis-Hastings MCMC，从未归一化的目标分布中采样

## 问题

语言模型处理完你的提示后，会输出一个包含 50,000 个 logits 的向量，词表中的每个 token 对应一个值。接下来它必须选出一个 token。应该怎样选？

如果总是选择概率最高的 token，每次回复都会完全相同，确定但乏味；如果从词表中均匀随机选择，输出就会变成乱码。正确答案位于这两个极端之间，而这个位置正由采样控制。

采样并不只用于文本生成。强化学习通过采样轨迹来估计策略梯度；VAE 从学习到的分布中采样潜在表示，并让梯度穿过随机过程；扩散模型先采样噪声，再反复去噪生成图像；Monte Carlo 方法通过样本平均估计没有闭式解的积分；MCMC 算法则探索无法穷举的高维后验分布。

每个生成式 AI 系统都是采样系统。采样策略决定输出的质量、多样性和可控性。本课会从均匀随机数出发，逐步构建主要采样方法，最终实现现代 LLM 和生成模型所使用的技术。

## 核心概念

### 采样为何重要

采样在 AI 与机器学习中承担四类基本角色：

**生成。**语言模型、扩散模型和 GAN 都通过采样产生输出。采样算法直接控制创造性、连贯性和多样性。温度、top-k 和 nucleus sampling 是工程师每天都会调整的旋钮。

**训练。**随机梯度下降会采样 mini-batch，Dropout 会采样需要停用的神经元，数据增强会采样随机变换，重要性采样则在强化学习（PPO、TRPO）中重新加权样本以降低梯度方差。

**估计。**机器学习中的许多量没有闭式解，例如数据分布上的期望损失、能量模型的配分函数，以及 Bayesian 推断中的证据。Monte Carlo 估计通过样本平均近似这些量。

**探索。**MCMC 算法会在 Bayesian 推断中探索后验分布；进化策略会采样参数扰动；Thompson sampling 会在多臂老虎机中平衡探索与利用。

核心挑战是：你只能直接从均匀分布、正态分布等简单分布中采样。面对其他分布，需要找到一种方法，把简单样本转换成目标分布的样本。

### 均匀随机采样

所有采样方法都从这里开始。均匀随机数生成器会在 [0, 1) 中产生数值，并保证长度相同的每个子区间拥有相同概率。

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12
```

要从 n 个离散项目中均匀采样，可以生成 U 并返回 floor(n * U)。要从连续区间 [a, b] 中均匀采样，则计算 a + (b - a) * U。

关键洞见是：一个均匀随机数恰好包含从任意分布生成一个样本所需的随机性，难点只是找到正确的变换。

### 逆 CDF 方法（逆变换采样）

累积分布函数（CDF）把数值映射为概率：

```
F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]
```

逆 CDF 会把概率映射回数值。如果 U ~ Uniform(0, 1)，那么 X = F_inverse(U) 就服从目标分布。

```
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**指数分布示例：**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda
```

只要 F_inverse 有闭式表达式，这种方法就十分理想。正态分布没有闭式逆 CDF，因此需要使用其他方法，例如 Box-Muller 或数值近似。

**离散版本：**对于离散分布，可以通过累计求和构建 CDF，生成 U，再找到累计概率第一次超过 U 的索引。这就是第 06 课中 `sample_categorical` 的工作方式。

### 拒绝采样

如果无法求 CDF 的逆，但可以计算目标 PDF（即使只知道差一个常数的未归一化形式），就可以使用拒绝采样。

```
Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. If u < p(x) / (M * q(x)), accept x
  4. Otherwise, reject and go to step 1

Acceptance rate = 1/M
```

上界 M 越紧，接受率越高。在低维空间（1–3 维）中，拒绝采样通常表现很好；在高维空间中，接受率会指数下降，因为提议分布的大部分体积都会被拒绝。这就是拒绝采样面对的维度灾难。

**示例：从截断正态分布采样。**在截断区间内使用均匀提议分布，包络上界 M 是该区间中正态 PDF 的最大值。

**示例：从半圆采样。**在包围半圆的矩形中均匀提出点，点落入半圆内部时接受。Monte Carlo 计算 pi 采用的就是这种方法：接受率等于面积比 pi/4。

### 重要性采样

有时你并不需要目标分布 p(x) 的样本，而是想估计 p(x) 下的某个期望，同时手中只有另一个分布 q(x) 的样本。

```
Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)
```

这对强化学习至关重要。在 PPO（Proximal Policy Optimization）中，你使用旧策略 pi_old 收集轨迹，却希望优化新策略 pi_new；重要性权重就是 pi_new(a|s) / pi_old(a|s)。PPO 会裁剪这些权重，防止新策略偏离旧策略过远。

重要性采样估计量的方差取决于 q 与 p 有多相似。如果二者差异很大，少数样本会得到极大的权重并主导估计。自归一化重要性采样会除以权重之和，从而缓解这一问题：

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### Monte Carlo 估计

Monte Carlo 估计通过随机样本的平均值近似积分，大数定律保证它会收敛。

```
Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension
```

误差率与维度无关。这就是为什么在网格积分无法处理的高维空间中，Monte Carlo 方法占据主导地位。

**估计 pi：**

```
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```

**估计期望：**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N
```

### Markov Chain Monte Carlo（MCMC）：Metropolis-Hastings

MCMC 会构造一条平稳分布为目标分布 p(x) 的 Markov 链。运行足够多步后，链中的样本就近似来自 p(x)。

```
Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples
```

对于对称提议分布（q(x'|x) = q(x|x')），比率会简化为 p(x')/p(x)。这就是原始 Metropolis 算法。

**为什么它有效。**接受规则保证详细平衡：位于 x 并转移到 x' 的概率，等于位于 x' 并转移到 x 的概率。详细平衡意味着 p(x) 是这条链的平稳分布。

**实践注意事项：**
- Burn-in：丢弃链达到平衡之前的早期样本
- Thinning：每隔 k 个样本保留一个，以降低自相关
- 提议尺度：太小会让链移动缓慢（接受率高，但探索慢）；太大则会拒绝大多数提议（接受率低，停在原地）
- 在高维空间中，Gaussian 提议的最佳接受率约为 0.234

### Gibbs 采样

Gibbs 采样是适用于多元分布的一种特殊 MCMC。它不会同时提出所有维度的新值，而是每次根据条件分布更新一个变量。

```
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Gibbs 采样要求能够从每个条件分布 p(x_i | x_{-i}) 中采样。许多模型都满足这一条件：
- Bayesian 网络：条件分布可以从图结构得到
- Gaussian 混合模型：条件分布是 Gaussian
- Ising 模型：每个自旋的条件分布只依赖其邻居

由于每次都从精确条件分布中采样，自动满足详细平衡，因此接受率始终为 1。

**局限性。**变量高度相关时，Gibbs 采样混合很慢，因为逐个更新变量无法沿分布的对角方向大步移动。

### 温度采样（用于 LLM）

语言模型会为词表中的每个 token 输出 logits z_1, ..., z_V，softmax 再把它们转换成概率。温度会在 softmax 之前缩放 logits：

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)
```

**为什么它有效。**用 T < 1 除 logits 会放大差异。如果 z_1 = 2、z_2 = 1，除以 T = 0.5 后得到 z_1/T = 4、z_2/T = 2，差距变大；softmax 后，最高 logit 的 token 会获得更多概率质量。

**实践建议：**
- T = 0.0：贪心解码，适合事实型问答
- T = 0.3–0.7：略有创造性，适合代码生成
- T = 0.7–1.0：平衡，适合一般对话
- T = 1.0–1.5：创意写作、头脑风暴
- T > 1.5：随机性越来越强，很少有实用价值

温度不会改变哪些 token 有可能被选择，只会改变分配给各 token 的概率质量。

### Top-k 采样

Top-k 采样把候选集合限制为概率最高的 k 个 token，然后重新归一化，并从受限集合中采样。

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  greedy decoding
k = V:  no filtering (standard sampling)
k = 40: typical setting, removes long tail of unlikely tokens
```

Top-k 能防止模型选择词表长尾中概率极低的 token，例如拼写错误或无意义内容。问题在于，无论上下文如何，k 始终固定。模型很自信时，一个 token 可能拥有 95% 概率，但 k=40 仍允许其他 39 个候选；模型很不确定时，概率可能分散在 1,000 个 token 上，k=40 又会切掉许多合理选项。

### Top-p（Nucleus）采样

Top-p 采样会动态调整候选集合大小。它不保留固定数量的 token，而是保留累计概率超过 p 的最小 token 集合。

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy
```

模型很自信时，nucleus sampling 只会保留少数 token，可能只有 2–3 个；模型不确定时，它会保留许多 token，可能达到 200 个。这种自适应行为，使 nucleus sampling 通常能生成比 top-k 更好的文本。

**常见组合：**
- 温度 0.7 + top-p 0.9：通用设置
- 温度 0.0（贪心）：适合确定性任务
- 温度 1.0 + top-k 50：Fan 等人（2018）原始论文的设置

Top-k 与 top-p 可以结合使用：先应用 top-k，再在剩余集合上应用 top-p。

### 重参数化技巧（用于 VAE）

变分自编码器（VAE）会把输入编码为潜在空间中的一个分布，从该分布采样，再解码样本以重建输入。问题在于：采样运算本身不可微，无法直接通过它反向传播。

```
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

重参数化技巧会把随机性与参数分离：

```
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```

这种方法之所以成立，是因为 N(mu, sigma^2) 与 mu + sigma * N(0, 1) 具有相同分布。关键洞见是：把随机性移到一个不带参数的来源 epsilon，再把样本写成参数的可微确定性变换。

**VAE 训练循环中的步骤：**
1. 编码器为每个输入输出 mu 和 log(sigma^2)
2. 采样 epsilon ~ N(0, 1)
3. 计算 z = mu + sigma * epsilon
4. 解码 z 以重建输入
5. 通过第 4、3、2、1 步反向传播；第 3 步可微，因此这一过程可行

没有重参数化技巧，VAE 就无法使用标准反向传播训练。正是这个洞见让 VAE 变得实用。

### Gumbel-Softmax（可微类别采样）

重参数化技巧适用于 Gaussian 等连续分布。对于离散类别分布，需要另一种方法。Gumbel-Softmax 提供了类别采样的可微近似。

**Gumbel-Max 技巧（不可微）：**

```
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```

**Gumbel-Softmax（可微近似）：**

```
Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation
```

Gumbel-Softmax 会生成离散样本的连续松弛形式。输出是概率向量（软 one-hot），而不是硬 one-hot，梯度能够流经 softmax。训练的前向传播中，还可以使用 straight-through 估计器：前向传播使用硬 argmax，反向传播则使用软 Gumbel-Softmax 梯度。

**应用：**
- VAE 中的离散潜变量
- 神经架构搜索（选择离散运算）
- 硬注意力机制
- 具有离散动作的强化学习

### 分层采样

标准 Monte Carlo 采样可能因为随机性，在样本空间中留下空白区域。分层采样会把空间划分成多个层，并从每一层采样，强制实现均匀覆盖。

```
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

分层采样的方差总是小于或等于标准 Monte Carlo：

```
Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.
```

**应用：**
- 数值积分（准 Monte Carlo）
- 训练数据划分（确保每个 fold 的类别平衡）
- 与重要性采样结合的分层重要性采样
- NeRF（Neural Radiance Fields）沿相机光线使用分层采样

### 与扩散模型的联系

扩散模型通过采样过程生成图像。前向过程在 T 个步骤中不断给图像添加 Gaussian 噪声，直到它变成纯噪声；反向过程则学习逐步去噪，恢复原始图像。

```
Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.
```

它与本课方法的联系包括：
- 每个去噪步骤都会使用重参数化技巧，即采样噪声后应用确定性变换
- 噪声调度 {alpha_t} 控制一种温度退火过程
- 训练使用 Monte Carlo 估计近似 ELBO（证据下界）
- 扩散模型中的 ancestral sampling 是一条 Markov 链，每一步只依赖当前状态

整个图像生成过程就是迭代采样：从噪声开始，每一步都在学习到的去噪模型条件下，采样一个噪声更少的版本。

```figure
monte-carlo-pi
```

## 动手构建

### 第 1 步：均匀采样与逆 CDF 采样

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

生成 10,000 个指数分布样本，验证其均值为 1/lambda。

### 第 2 步：拒绝采样

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

使用拒绝采样从截断正态分布生成样本，再绘制样本直方图验证形状。

### 第 3 步：重要性采样

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

使用均匀提议分布估计正态分布下的 E[X^2]，并与已知答案（mu^2 + sigma^2）比较。

### 第 4 步：使用 Monte Carlo 估计 pi

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### 第 5 步：Metropolis-Hastings MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

从双峰分布（两个 Gaussian 的混合）中采样，并可视化链的轨迹。

### 第 6 步：Gibbs 采样

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### 第 7 步：温度采样

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

展示温度如何改变一组 token logits 的输出分布。

### 第 8 步：Top-k 与 top-p 采样

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### 第 9 步：重参数化技巧

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

演示梯度能够通过重参数化样本传播，却无法通过直接采样传播。

### 第 10 步：Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

展示温度逐渐降低时，输出如何趋近 one-hot 向量。

包含全部可视化的完整实现位于 `code/sampling.py`。

## 实际使用

下面是使用 NumPy 和 SciPy 的生产版本：

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

大规模 MCMC 应使用专用库：
- PyMC：使用 NUTS（自适应 HMC）进行完整 Bayesian 建模
- emcee：ensemble MCMC 采样器
- NumPyro/JAX：GPU 加速 MCMC

你已经从零实现了这些方法，现在知道库函数调用在底层做了什么。

## 练习

1. 为 Cauchy 分布实现逆 CDF 采样。其 CDF 为 F(x) = 0.5 + arctan(x)/pi。生成 10,000 个样本，把直方图与真实 PDF 进行比较，并观察重尾现象，也就是远离中心的极端值。

2. 使用 Uniform(0, 1) 提议分布，通过拒绝采样从 Beta(2, 5) 分布生成样本。将接受的样本与真实 Beta PDF 画在一起。理论接受率是多少？

3. 分别使用 1,000、10,000 和 100,000 个样本，通过 Monte Carlo 估计 sin(x) 在 0 到 pi 之间的积分。比较各级别误差，并验证误差按 O(1/sqrt(N)) 缩放。

4. 实现 Metropolis-Hastings，从与 exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2) 成正比的二维分布 p(x, y) 中采样。绘制样本和链轨迹，并尝试不同的提议标准差。

5. 构建完整文本生成演示：给定 10 个词及对应 logits，分别使用（a）贪心、（b）temperature=0.7、（c）top-k=3、（d）top-p=0.9，生成长度为 20 个 token 的序列。每种方法运行 5 次，比较输出多样性。

## 关键术语

| 术语 | 人们常说 | 准确含义 |
|------|----------------|----------------------|
| Sampling | “抽取随机值” | 按概率分布生成数值，也是所有生成式 AI 背后的机制 |
| Uniform distribution | “全部等概率” | [a, b] 中每个值的概率密度均为 1/(b-a)，是所有采样方法的起点 |
| Inverse CDF | “概率变换” | F_inverse(U) 把均匀样本转换为已知 CDF 分布的样本，准确而高效 |
| Rejection sampling | “提出后接受或拒绝” | 从简单提议分布生成样本，再按目标/提议比率对应的概率接受；结果精确，但会浪费样本 |
| Importance sampling | “重新加权样本” | 使用 q(x) 的样本估计 p(x) 下的期望，并按 p(x)/q(x) 加权；是强化学习 PPO 的核心机制 |
| Monte Carlo | “对随机样本求平均” | 使用样本平均近似积分，无论维度如何，误差都是 O(1/sqrt(N)) |
| MCMC | “最终会收敛的随机游走” | 构造平稳分布为目标分布的 Markov 链；Metropolis-Hastings 是基础算法 |
| Metropolis-Hastings | “接受上坡，有时也接受下坡” | 提出移动并根据密度比接受；详细平衡保证收敛到目标分布 |
| Gibbs sampling | “每次更新一个变量” | 固定其他变量，根据条件分布依次更新每个变量；接受率为 100% |
| Temperature | “置信度旋钮” | softmax 前用 T 除 logits；T<1 让分布更尖锐、更自信，T>1 让分布更平坦、更多样 |
| Top-k sampling | “保留最好的 k 个” | 把概率最高的 k 个 token 以外的概率置零，重新归一化后采样；候选集合大小固定 |
| Nucleus sampling (top-p) | “保留高概率部分” | 保留累计概率超过 p 的最小 token 集合；候选集合大小自适应 |
| Reparameterization trick | “把随机性移到外部” | 写成 z = mu + sigma * epsilon，其中 epsilon ~ N(0,1)，使采样过程可微，是 VAE 训练的关键 |
| Gumbel-Softmax | “软类别采样” | 使用 Gumbel 噪声和带温度的 softmax，对类别采样进行可微近似 |
| Stratified sampling | “强制覆盖” | 把样本空间划分成多个层，并从每层采样；方差始终低于或等于朴素 Monte Carlo |
| Burn-in | “预热期” | 链达到平稳分布前丢弃的早期 MCMC 样本 |
| Detailed balance | “可逆条件” | p(x) * T(x->y) = p(y) * T(y->x)，是 p 成为 Markov 链平稳分布的充分条件 |
| Diffusion sampling | “迭代去噪” | 从噪声开始，反复应用学习到的去噪步骤生成数据；每一步都是条件采样操作 |

## 延伸阅读

- [Holbrook（2023）：Metropolis-Hastings 算法](https://arxiv.org/abs/2304.07010)——MCMC 基础的详细教程
- [Jang、Gu、Poole（2017）：使用 Gumbel-Softmax 进行类别重参数化](https://arxiv.org/abs/1611.01144)——Gumbel-Softmax 原始论文
- [Holtzman 等（2020）：神经文本退化的奇特案例](https://arxiv.org/abs/1904.09751)——nucleus（top-p）采样论文
- [Kingma 与 Welling（2014）：自动编码变分 Bayes](https://arxiv.org/abs/1312.6114)——引入重参数化技巧的 VAE 论文
- [Ho、Jain、Abbeel（2020）：去噪扩散概率模型](https://arxiv.org/abs/2006.11239)——将采样与图像生成联系起来的 DDPM 论文
