# JAX 入门

> PyTorch 会修改张量，TensorFlow 会构建计算图，JAX 则会编译纯函数。最后这一点会改变你思考深度学习的方式。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 第 03 阶段第 01–10 课、NumPy 基础
**Time:** 约 90 分钟

## 学习目标

- 使用 JAX 的函数式 API（jax.numpy、jax.grad、jax.jit、jax.vmap）编写纯函数式神经网络代码
- 解释 PyTorch 的即时执行与可变状态设计，同 JAX 函数式编译模型之间的关键差异
- 应用 jit 编译和 vmap 向量化，使训练循环比朴素 Python 实现更快
- 使用 JAX 训练一个简单网络，并把显式状态管理与 PyTorch 的面向对象方式进行对比

## 问题

你已经知道如何用 PyTorch 构建神经网络：定义一个 `nn.Module`，调用 `.backward()`，再让优化器执行一步更新。这套方式确实有效，也有数百万人在使用。

但 PyTorch 的基因中带有一项限制：它会在 Python 中逐项即时追踪操作。每一次 `tensor + tensor` 都要单独启动一个内核，每一步训练都会重新解释同一段 Python 代码。一般情况下这没有问题，直到你需要在 2,048 个 TPU 上训练一个拥有 5400 亿参数的模型——此时这些开销足以拖垮整个任务。

Google DeepMind 使用 JAX 训练 Gemini，Anthropic 也使用 JAX 训练 Claude。这些绝非小规模任务，而是地球上规模最大的神经网络训练之一。他们选择 JAX，是因为它把训练循环视为可编译程序，而不是一串 Python 调用。

JAX 就是拥有三种超能力的 NumPy：自动微分、面向 XLA 的即时编译，以及自动向量化。你只需编写处理单个样本的函数，JAX 就能为你得到处理整个批次、计算梯度、编译为机器码并跨多个设备运行的函数，而且无需修改原始函数。

## 核心概念

### JAX 的理念

JAX 是一个函数式框架。没有类，没有可变状态，也没有 `.backward()` 方法，而是采用以下方式：

| PyTorch | JAX |
|---------|-----|
| 带状态的 `nn.Module` 类 | 纯函数：`f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| 即时执行 | 通过 XLA 进行 JIT 编译 |
| `for x in batch:` 手工循环 | `jax.vmap(f)` 自动向量化 |
| `DataParallel` / `FSDP` | `jax.pmap(f)` 自动并行 |
| 可变的 `model.parameters()` | 不可变的数组 pytree |

这并非风格偏好，而是编译器约束。JIT 编译要求纯函数：相同输入始终产生相同输出，而且不能有副作用。正是这种限制使 100 倍加速成为可能。

### jax.numpy：熟悉的表层接口

JAX 在加速器上重新实现了 NumPy API：

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

函数名称相同，广播规则相同，切片语义也相同。但数组位于 GPU/TPU 上，而且每一项操作都可以被编译器追踪。

一个关键区别是：JAX 数组不可变，不能写 `a[0] = 5`，而要写成 `a = a.at[0].set(5)`。一开始可能会觉得别扭，但适应一周左右后通常就会豁然开朗：正是不可变性让 `grad`、`jit` 和 `vmap` 这样的变换可以自由组合。

### jax.grad：函数式自动微分

PyTorch 把梯度附着在张量上，也就是 `.grad`；JAX 则把梯度附着在函数上。

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` 接收一个函数，返回另一个用于计算梯度的函数。不需要调用 `.backward()`，也没有保存在张量上的计算图。梯度只是另一个可以调用、组合或 JIT 编译的函数。

这种机制可以任意组合：

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

二阶导数、三阶导数、Jacobian、Hessian，都可以通过组合 `grad` 得到。PyTorch 也能做到，例如使用 `torch.autograd.functional.hessian`，但那是后来附加的能力；在 JAX 中，它就是整个系统的基础。

约束在于，`grad` 只能处理纯函数。函数内部不能使用普通打印语句，因为它只会在追踪而非执行时运行；不能修改外部状态；也不能在没有显式管理 key 的情况下生成随机数。

### jit：编译到 XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

首次调用时，JAX 会追踪函数，也就是记录发生了哪些操作，而不执行真实计算。随后它把追踪结果交给 XLA（Accelerated Linear Algebra），这是 Google 为 TPU 和 GPU 开发的编译器。XLA 会融合操作、消除多余的内存复制，并生成经过优化的机器码。

后续调用会完全绕过 Python，直接在加速器上以 C++ 级速度运行编译后的代码。

JIT 适合以下场景：
- 训练步骤，也就是成千上万次重复相同计算
- 推理，也就是使用同一个模型处理不同输入
- 任何会以相似形状输入调用不止一次的函数

JIT 不适合以下场景：
- 包含依赖具体数值的 Python 控制流，例如 `if x > 0` 且 x 是被追踪数组
- 只执行一次的计算，因为编译开销会超过运行时间
- 调试，因为追踪会隐藏真实执行过程

控制流限制是真实存在的。必须用 `jax.lax.cond` 取代 `if/else`，用 `jax.lax.scan` 取代 `for` 循环。这些并不是可选风格，而是为获得编译能力付出的代价。

### vmap：自动向量化

先编写一个处理单个样本的函数：

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` 可以把它提升为处理整个批次的函数：

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` 表示：不要沿 `params` 的任何轴分批，因为参数由所有样本共享；沿 `x` 的第 0 轴分批。不需要手写 `for` 循环，不需要重塑形状，也不需要让批次维度贯穿整个实现，JAX 会自行识别批次维度并向量化全部计算。

这并非语法糖。`vmap` 会生成融合后的向量化代码，比 Python 循环快 10–100 倍，而且可以与 `jit` 和 `grad` 组合：

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

只需一行，就能得到逐样本梯度。在 PyTorch 中，如果不采用复杂变通方式，这几乎无法实现。

### pmap：跨设备数据并行

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` 会把函数复制到所有可用设备（GPU/TPU）上，并拆分批次。在函数内部，`jax.lax.pmean` 和 `jax.lax.psum` 用于跨设备同步梯度。

Google 使用 `pmap` 及其后继 `shard_map`，在数千颗 TPU v5e 芯片上训练 Gemini。其编程模型是：先编写单设备版本，再用 `pmap` 包装，就完成了。

### Pytrees：通用数据结构

JAX 操作的是“pytrees”，也就是列表、元组、字典和数组任意嵌套形成的结构。模型参数可以表示成一棵 pytree：

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

JAX 的每种变换，包括 `grad`、`jit`、`vmap`，都知道如何遍历 pytrees。`jax.tree.map(f, tree)` 会把 `f` 应用到每个叶节点。优化器就是以这种方式一次更新所有参数：

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

不需要 `.parameters()` 方法，也不需要参数注册；树结构本身就是模型。

### 函数式与面向对象

PyTorch 把状态保存在对象内部：

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX 使用显式状态的纯函数：

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

params 由调用方传入，不保存任何东西，也不修改任何东西。因此每个函数都可测试、可组合、可编译。代价是需要自己管理 params，或者使用 Flax、Equinox 等库。

### JAX 生态

JAX 提供基础原语，其他库则提供更方便的开发体验：

| 库 | 作用 | 风格 |
|---------|------|-------|
| **Flax**（Google） | 神经网络层 | 带显式状态的 `nn.Module` |
| **Equinox**（Patrick Kidger） | 神经网络层 | 基于 Pytree，符合 Python 习惯 |
| **Optax**（DeepMind） | 优化器 + LR 调度 | 可组合的梯度变换 |
| **Orbax**（Google） | 检查点 | 保存/恢复 pytrees |
| **CLU**（Google） | 指标 + 日志 | 训练循环工具 |

Optax 是标准优化器库。它把梯度变换，例如 Adam、SGD 和裁剪，与参数更新分离，因此组合起来非常简单：

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### 何时选择 JAX，何时选择 PyTorch

| 因素 | JAX | PyTorch |
|--------|-----|---------|
| TPU 支持 | 一等支持（Google 同时开发两者） | 社区维护（torch_xla） |
| GPU 支持 | 良好（通过 XLA 使用 CUDA） | 业界最佳（原生 CUDA） |
| 调试 | 较难（追踪 + 编译） | 简单（即时、逐行执行） |
| 生态系统 | 以研究为主（Flax、Equinox） | 庞大（HuggingFace、torchvision 等） |
| 招聘市场 | 小众（Google/DeepMind/Anthropic） | 主流（随处可见） |
| 大规模训练 | 更优（XLA、pmap、mesh） | 良好（FSDP、DeepSpeed） |
| 原型速度 | 较慢（函数式开销） | 较快（直接修改并运行） |
| 生产推理 | TensorFlow Serving、Vertex AI | TorchServe、Triton、ONNX |
| 使用者 | DeepMind（Gemini）、Anthropic（Claude） | Meta（Llama）、OpenAI（GPT）、Stability AI |

坦率的答案是：除非有使用 JAX 的具体理由，否则选择 PyTorch。这些理由包括拥有 TPU、需要逐样本梯度、需要进行超大规模多设备训练，或者你在 Google、DeepMind、Anthropic 工作。

### JAX 中的随机数

JAX 没有全局随机状态。每次随机操作都需要显式传入 PRNG key：

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

一开始这很烦人，但它保证了跨设备、跨编译过程的可复现性，而 PyTorch 的 `torch.manual_seed` 在多 GPU 场景中无法提供同等保证。

```figure
batchnorm-effect
```

## 动手构建

### 第 1 步：环境与数据

我们会使用 JAX 和 Optax，在 MNIST 上训练一个三层 MLP：784 个输入，两层隐藏层分别有 256 和 128 个神经元，输出 10 个类别。

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### 第 2 步：初始化参数

不需要类，只需一个返回 pytree 的函数：

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

这就是手工完成的 He 初始化。一个种子拆分成三个 PRNG key，每个权重都是嵌套字典中的不可变数组。

### 第 3 步：前向传播

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

这些都是纯函数：传入 params，返回预测，不使用 `self`，也不保存状态。`loss_fn` 从零计算交叉熵，也就是 Softmax、取对数、取负平均值。

### 第 4 步：JIT 编译的训练步骤

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` 会在一次传播中同时返回损失值和梯度。`@jax.jit` 装饰器把两个函数都编译为 XLA。第一次调用后，每一步训练都不再经过 Python。

### 第 5 步：训练循环

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

训练 10 个 epoch，测试准确率约为 97%。第一个 epoch 较慢，因为需要 JIT 编译；第 2–10 个 epoch 则会很快。

注意这里少了哪些步骤：没有 `.zero_grad()`，没有 `.backward()`，也没有 `.step()`。整个更新只是一项组合函数调用。梯度计算、Adam 变换和参数更新全部发生在 `train_step` 内。

## 实际应用

### Flax：Google 的标准选择

Flax 是最常用的 JAX 神经网络库。它重新引入了 `nn.Module`，但仍要求显式管理状态：

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

结构与 PyTorch 相同，但 `params` 和模型相互分离。`model.init()` 创建 params，`model.apply(params, x)` 执行前向传播，模型对象本身不包含状态。

### Equinox：更符合 Python 习惯的选择

Patrick Kidger 开发的 Equinox 把模型表示成 pytrees：

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

模型本身就是一棵 pytree，不需要 `.apply()`。参数就是模型的叶节点。这种设计更贴近 JAX 的思维方式。

### Optax：可组合优化器

Optax 把梯度变换与参数更新解耦：

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

梯度裁剪、学习率预热、权重衰减都被组合成一条变换链。每个变换接收梯度，修改后再传给下一个变换，不需要一个庞大的优化器类。

## 交付成果

**安装：**

```bash
pip install jax jaxlib optax flax
```

启用 GPU 支持：

```bash
pip install jax[cuda12]
```

使用 TPU（Google Cloud）：

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**性能陷阱：**

- 第一次 JIT 调用较慢，因为需要编译。基准测试前应先预热。
- 避免在 JIT 内部用 Python 循环遍历 JAX 数组，应使用 `jax.lax.scan` 或 `jax.lax.fori_loop`。
- `jax.debug.print()` 可以在 JIT 内部使用，普通 `print()` 不可以。
- 使用 `jax.profiler` 或 TensorBoard 分析性能；XLA 编译可能会隐藏瓶颈。
- JAX 默认预分配 75% 的 GPU 内存。设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 可以禁用这一行为。

**检查点：**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**本课会产出：**
- `outputs/prompt-jax-optimizer.md`——用于选择正确 JAX 优化器配置的提示词
- `outputs/skill-jax-patterns.md`——介绍 JAX 函数式模式的技能

## 练习

1. 为 MLP 添加 Dropout。在 JAX 中，Dropout 需要 PRNG key，因此要让 key 穿过前向传播，并为每个 Dropout 层分别拆分。比较采用与不采用 Dropout 时的测试准确率。

2. 使用 `jax.vmap` 为一批 32 张 MNIST 图像计算逐样本梯度，再计算每个样本的梯度范数。哪些样本的梯度最大？为什么？

3. 用适用于任意层数的通用 `mlp_forward(params, x)` 替换手工前向函数。使用 `jax.tree.leaves` 自动判断深度。

4. 对采用和不采用 `@jax.jit` 的训练步骤进行基准测试，各计时 100 步。你的硬件上能获得多大加速？第一次调用的编译开销是多少？

5. 通过组合 `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))` 实现梯度裁剪。分别采用和不采用裁剪进行训练，绘制训练过程中的梯度范数以观察效果。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| XLA | “让 JAX 变快的东西” | Accelerated Linear Algebra，一种从计算图融合操作并生成优化 GPU/TPU 内核的编译器 |
| JIT | “即时编译” | JAX 在首次调用时追踪函数并编译到 XLA，后续调用直接运行编译版本 |
| 纯函数 | “没有副作用” | 输出只取决于输入的函数；没有全局状态和修改操作，随机数也必须通过显式 key 生成 |
| vmap | “自动分批” | 无需改写代码，就把处理单个样本的函数变换成处理整个批次的函数 |
| pmap | “自动并行” | 把函数复制到多个设备上，并拆分输入批次 |
| Pytree | “嵌套数组字典” | JAX 可以遍历和变换的列表、元组、字典与数组的任意嵌套结构 |
| 追踪 | “记录计算过程” | JAX 使用抽象值执行函数以构建计算图，而不计算真实结果 |
| 函数式自动微分 | “对函数求梯度” | 通过变换函数来计算导数，而不是把梯度存储附着到张量上 |
| Optax | “JAX 的优化器库” | 由 Adam、SGD、裁剪、调度等可组合梯度变换构成的库 |
| Flax | “JAX 的 nn.Module” | Google 为 JAX 开发的神经网络库，在保持状态显式的同时提供层抽象 |

## 延伸阅读

- JAX 文档：https://jax.readthedocs.io/ —— 官方文档，提供关于 grad、jit 和 vmap 的优秀教程
- 《JAX: composable transformations of Python+NumPy programs》（Bradbury 等，2018）——解释设计理念的原始论文
- Flax 文档：https://flax.readthedocs.io/ —— Google 为 JAX 开发的神经网络库
- Patrick Kidger，《Equinox: neural networks in JAX via callable PyTrees and filtered transformations》（2021）——比 Flax 更符合 Python 习惯的替代方案
- DeepMind，“Optax: composable gradient transformation and optimisation”——标准优化器库
- 《You Don't Know JAX》（Colin Raffel，2020）——由 T5 作者之一撰写的 JAX 陷阱与模式实用指南
