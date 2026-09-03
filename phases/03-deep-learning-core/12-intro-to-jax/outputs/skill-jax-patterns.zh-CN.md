---
name: skill-jax-patterns
description: JAX 中的函数式编程模式：何时以及如何使用 grad、jit、vmap 和 pmap
version: 1.0.0
phase: 3
lesson: 12
tags: [jax, functional-programming, autodiff, compilation, vectorization]
---

# JAX 功能模式

JAX 转换了纯函数。下面的每个模式都遵循一个规则：编写一个接受输入并返回输出的函数，没有副作用。然后改造它。

## 四种转变

### grad -- 函数微分
```python
grads = jax.grad(loss_fn)(params, x, y)
loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
```
何时使用：您需要梯度进行优化。
约束：函数必须返回标量。对于非标量输出，请使用 `jax.jacobian` 。

### jit -- 编译函数
```python
fast_fn = jax.jit(f)
```
使用时间：该函数将使用相同形状的输入被调用多次。
约束：没有依赖于跟踪值的 Python 控制流。使用 `jax.lax.cond` 作为条件，使用 `jax.lax.scan` 作为循环。

### vmap -- 向量化函数
```python
batch_fn = jax.vmap(f, in_axes=(None, 0))
```
使用时间：您为一个示例编写了一个函数，并且需要它批量工作。
 `in_axes` 指定要批处理的参数轴。  `None` 表示不批量（广播）。

### pmap -- 跨设备并行化
```python
parallel_fn = jax.pmap(f, axis_name='devices')
```
使用时机：您有多个 GPU/TPU 并且需要数据并行性。
在函数内部，`jax.lax.pmean(x, 'devices')` 跨设备进行平均。

## 构图规则

变换组成。顺序很重要：
```python
per_example_grads = jax.jit(jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0)))
```
从右到左阅读：获取loss_fn的梯度，对示例进行向量化，编译结果。

有效成分：
- `jit(grad(f))` -- 编译梯度计算
- `jit(vmap(f))` -- 编译的批量计算
- `vmap(grad(f))` -- 每个示例的梯度
- `pmap(jit(f))` -- 并行编译计算
- `grad(jit(f))` -- 编译函数的梯度（与 jit(grad(f)) 相同）

## 参数管理模式

JAX 参数是 pytree（数组的嵌套字典）：
```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 10)),  'b': jnp.zeros(10)},
}
```
一次性更新所有参数：
```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```
计数参数：
```python
n_params = sum(p.size for p in jax.tree.leaves(params))
```
## PRNG 密钥管理

JAX 需要显式随机密钥：
```python
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
noise = jax.random.normal(subkey, shape)
```
对于多个随机操作，拆分一次：
```python
keys = jax.random.split(key, n)
```
切勿重复使用密钥。使用前务必分开。

## 常见错误

1. **jit 内的可变数组**：JAX 数组是不可变的。使用 `x.at[i].set(v)` 而不是 `x[i] = v` 。

2. **在 jit 中使用 Python print**：`print` 在跟踪期间运行，而不是在执行期间运行。使用 `jax.debug.print("{}", x)` 。

3. **Python if/for 跟踪值上的内部 jit**：使用 `jax.lax.cond` 、 `jax.lax.switch` 、 `jax.lax.scan` 、 `jax.lax.fori_loop` 。

4. **忘记 `.block_until_ready()` **：JAX 使用异步调度。对于基准测试，请调用 `.block_until_ready()` 等待实际完成。

5. **重用 PRNG 密钥**：具有相同密钥的两个操作会产生相同的“随机”值。总是分裂。

6. **即时函数中的全局状态**：全局变量在跟踪时捕获。追踪后的变化是不可见的。将所有内容作为参数传递。

## 决策清单

1. 该函数是否被多次调用？添加 `@jax.jit` 。
2. 需要渐变吗？用 `jax.grad` 或 `jax.value_and_grad` 包裹。
3. 它是否处理一个示例，但您有一批？用 `jax.vmap` 包裹。
4. 您有多个设备吗？用 `jax.pmap` 包裹。
5.它使用随机性吗？显式地通过 PRNG 键。
6. 它对数组值有Python控制流吗？替换为 `jax.lax` 原语。

## 何时使用 JAX

在以下情况下使用 JAX：
- 您需要每个示例的梯度（差分隐私、Fisher 信息）
- 您正在 TPU 上进行训练（JAX 是本机框架）
- 你需要高阶导数（Hessians、Jacobians）
- 您想要将整个训练步骤编译为单个内核
- 您的团队在 Google DeepMind 或 Anthropic

在以下情况下使用 PyTorch：
- 您想要最大的生态系统（HuggingFace、torchvision、Lightning）
- 您优先考虑调试的简易性而不是原始速度
- 您正在使用 TorchServe/Triton 部署到 NVIDIA GPU
- 您正在招聘（存在更多 PyTorch 开发人员）
- 您想要快速迭代新架构
