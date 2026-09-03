# 视觉编码器图块

> 读取像素的视觉模型，同样需要一个“像素 tokenizer”。patch embedding 就是这个 tokenizer。先把图像切成规则方格，再把每个方格展平，通过一个线性层投影，最后补上一份 2D 位置信号，让 transformer 知道每个方格原本位于整张图中的什么位置。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 第 19 阶段第 30 到 37 课（Track B 基础）
**Time:** 约 90 分钟

## 学习目标

- 把一张图像 token 化成固定长度的 patch embedding 序列。
- 实现一个基于 `Conv2d` 的 patch projection，并让它与 unfold-then-linear 的数学形式一致。
- 构造一个确定性的 2D sinusoidal position embedding，使 token 顺序能够编码空间位置。
- 在一个合成 fixture 上验证 patch 数量、embedding 形状，以及 `Conv2d` / unfold 等价性。

## 问题

transformer 吃的是向量序列，图像则是一个 3 通道网格。如果把每个像素都当成一个 token，序列长度会直接爆炸：一张 224x224 的 RGB 图像会变成 150,528 个 tokens，12 层 transformer 根本负担不起这样的 attention 代价。如果反过来把整张图像直接读成一个巨大的平坦向量，又会把局部结构全部抹掉，而 attention 层后面也补不回来。编码器前端的任务，就是把像素网格压缩成几百个 token，让每个 token 都概括一个方形区域。

patch embedding 用一层线性投影就能做到这一点。把一张 224x224 图像切成 16x16 patches，会得到一个 14x14 的网格，也就是 196 个 patches。每个 patch 会从 `(3, 16, 16) = 768` 个像素值展平成一个向量，再通过线性层映射到模型的 hidden dimension。这样 transformer 看到的就是 196 个维度为 `hidden` 的 tokens，再额外加上一个 CLS token。这已经是后续网络可以真正处理的序列长度了。

## 概念

```mermaid
flowchart LR
  Image[224x224x3 image] --> Cut[cut into 16x16 patches]
  Cut --> Grid[14x14 grid of patches]
  Grid --> Flatten[flatten each patch]
  Flatten --> Proj[linear projection]
  Proj --> Tokens[196 tokens of dim hidden]
  Tokens --> Pos[add 2D sinusoidal position]
  Pos --> Out[final token sequence]
```

### 为什么用 patch，而不是 pixel

attention 对序列长度是平方复杂度。一个 196-token 的序列，每个头、每一层需要计算 `196 * 196 = 38,416` 个 attention scores；一个 150,528-token 的序列，则需要 `150,528 * 150,528 = 22.6 billion`。patch 让 attention 计算量减少了大约 590,000 倍，而一个 16x16 区域通常已经足够携带高层视觉任务需要的信号。代价是：单个 patch 内部会丢失一部分细粒度空间细节。这也是为什么下游多模态系统在需要精确定位时，常常还会额外跑一条高分辨率分支。

### 为什么一层线性投影就够了

每个 patch 都被当作一个独立向量。投影层学习到的是一个基底，例如边缘检测器、颜色滤波器和简单纹理。对 ViT-Base 来说，这一层只有 `768 * 768 = 589,824` 个参数，规模很小，也很好训练。更深的卷积 stem 当然存在，也就是所谓的“hybrid” ViT，但平坦的线性投影仍然是标准做法，大多数现代开源视觉编码器都采用这个形状。

### `Conv2d` 这个技巧

一个 `Conv2d(in_channels=3, out_channels=hidden, kernel_size=patch_size, stride=patch_size)`，在没有 padding 的情况下，数值上等价于 unfold-then-linear。原因是：每个输出位置，本质上都在用一组卷积核权重对对应 patch 像素做点积。换句话说，这个 convolution 本身就是 patch projection。大多数生产代码库都用这种写法，因为它在 GPU 上更快，而且能少做一次 reshape。

### 位置嵌入

投影后的 tokens 本身不携带顺序信息。2D sinusoidal embedding 会给每个 token 加上一段固定信号，用来编码它的 `(row, col)` 位置。embedding 维度的一半用多频率 sin/cos 编码行位置，另一半编码列位置。这种编码是确定性的，因此在换分辨率时不需要重新训练，而且能平滑插值到训练时从未出现过的网格上。

| 组件 | 形状 | 参数量 |
|-----------|-------|------------|
| Patch 投影（`Conv2d`） | `(hidden, 3, patch, patch)` | `3 * P * P * hidden + hidden` |
| 位置嵌入（固定） | `(num_patches, hidden)` | 0（计算得到，不参与学习） |
| CLS token（可学习） | `(1, hidden)` | `hidden` |

对于 224 分辨率下的 ViT-Base/16，projection 一共有 590,592 个参数，CLS token 有 768 个参数，而 sinusoidal position 没有任何可学习参数。下一课（59）会在这个前端之上再堆叠一个 12 层 transformer。

### 用等价性做 sanity check

patch 这一步有两种写法：一种是 `Conv2d` projection，另一种是显式的 unfold-then-linear。对于同一组权重，这两种写法必须输出完全一样的结果。如果不一样，说明 unfold 的数学实现有问题，而整个编码器后续部分都会建立在错误基础上。本课测试专门会验证这件事。

```figure
ch-patch-tokenizer
```

## 动手实现

`code/main.py` 实现了：

- `PatchEmbed`，一个用 `nn.Module` 包装 `Conv2d` patch projection 的模块。
- `sinusoidal_2d(grid_h, grid_w, dim)`，一个无状态函数，用来构建 2D position table。
- `VisionFrontEnd`，把 patch embedding、CLS prepend 和 position addition 组合成一次 forward pass。
- `synthesize_image(seed)` helper，使用 `numpy.random` 构造一个确定性的 224x224x3 fixture。
- 一个 demo：把一张 fixture image 跑过前端，并打印输出形状、CLS token 的 norm，以及 position embedding 中的一行。

运行它：

```bash
python3 code/main.py
```

输出：224x224 的 fixture 会被 token 化成一个形状为 `(1, 197, 768)` 的序列。第一个 token 是 CLS，后面的 196 个是 patch tokens。position embedding 的 norm 在同一行内保持一致，这正是 sinusoidal 编码的典型特征。

## 实际使用

相同的 patch front end 出现在所有现代 vision-language model 中：CLIP ViT-L/14、SigLIP、DINOv2、Qwen-VL 系列，以及 InternVL 栈，都是从一个 `Conv2d` patch projection 加上 position signal 开始的。不同模型家族之间的差异主要出现在下游：CLS pooling 或 no-CLS pooling、register tokens、patch size 是 14 还是 16、以及通过插值位置编码支持动态分辨率。本课实现的 frontend，就是所有这些模型共同站立的底板。

## 测试

`code/test_main.py` 覆盖：

- patch 数量是否等于 `(image_size / patch_size) ** 2`
- 输出形状是否等于 `(batch, num_patches + 1, hidden)`
- `Conv2d` projection 是否等于在小 fixture 上手写的 unfold-then-linear
- sinusoidal position table 在多次调用之间是否保持确定性
- CLS token 是否能沿 batch 维正确 broadcast，且没有泄漏

运行它们：

```bash
python3 -m unittest code/test_main.py
```

## 练习

1. 把 sinusoidal position 换成一个可学习的 `nn.Parameter`，并在一个很小的合成分类任务上比较第一轮训练损失。固定分辨率下，learned positions 往往更强；而当训练后改变分辨率时，sinusoidal 往往更稳。

2. 把 `Conv2d` 换成显式的 `nn.Unfold` 加 `nn.Linear`，并断言两者输出在浮点误差范围内一致。同一套数学，两种写法。

3. 增加对非方形 patch 尺寸的支持，例如宽屏输入里的 32x16，并验证位置表能够正确处理非方形网格。

4. profile patch 这一步在 batch size 为 1、8、64 时的耗时。patch projection 几乎从来不是瓶颈，真正主导时间的是后面的 attention 层。

5. 把前端作为冻结特征提取器，用在一个 4 类合成形状数据集上，例如 circles、squares、triangles、stars。CLS token 输出应该能够被线性分开。

## 关键术语

| 术语 | 含义 |
|------|---------------|
| Patch | 图像中的方形子区域，常见大小是 14x14 或 16x16 |
| Patch embedding | 将一个展平的 patch 线性投影到隐藏维度后得到的向量 |
| Sequence length | 完成 patch token 化后的 token 数量，通常还包括 CLS token |
| Sinusoidal position | 用于编码 2D 网格坐标的固定 sin/cos 信号 |
| CLS token | 添加在序列开头、用于池化的可学习向量 |

## 延伸阅读

- An Image is Worth 16x16 Words (ViT, 2021) 介绍了原始的 patch-embed 叙述方式。
- Attention Is All You Need (2017) 提供了这里改写为 2D 的 sinusoidal position 公式。
- DINOv2 论文可用于继续做 register tokens，这可以作为第 6 个练习扩展。
