// 课程：实时视觉边缘部署（阶段 04 / 课程 15）
// 主题：用 Rust 实现边缘推理循环。构建微型深度可分离卷积块（MobileNet
// 基本单元），在 160x160x3 输入张量上运行，并按设备端分析器的方式报告
// p50/p95/p99 延迟。仅使用标准库。
// 参考资料：
//   https://doc.rust-lang.org/std/time/struct.Instant.html
//   https://arxiv.org/abs/1704.04861  （MobileNetV1：深度可分离卷积）
//   https://pytorch.org/docs/stable/quantization.html  （边缘设备测量规范）
// 构建：rustc --edition 2021 -O code/main.rs -o /tmp/lesson_edge && /tmp/lesson_edge

use std::time::Instant;

const H: usize = 160;
const W: usize = 160;
const C_IN: usize = 3;
const C_OUT: usize = 16;
const K: usize = 3;
const WARMUP: usize = 3;
const ITERS: usize = 20;

#[derive(Clone)]
struct Tensor {
    data: Vec<f32>,
    h: usize,
    w: usize,
    c: usize,
}

impl Tensor {
    fn zeros(h: usize, w: usize, c: usize) -> Self {
        Self { data: vec![0.0; h * w * c], h, w, c }
    }

    fn idx(&self, y: usize, x: usize, c: usize) -> usize {
        (y * self.w + x) * self.c + c
    }
}

// 低成本的确定性 PRNG，避免在仅使用标准库的课程中引入 rand。
fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let bits = (*seed >> 33) as u32;
    (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn fill_random(t: &mut Tensor, seed: &mut u64) {
    for v in t.data.iter_mut() {
        *v = lcg(seed) * 0.5;
    }
}

// 深度卷积：每个输入通道使用一个 3x3 卷积核，不进行跨通道混合。
// MobileNet 借此将 FLOPs 降至稠密卷积的约 1/9。
fn depthwise_conv(input: &Tensor, weights: &[f32]) -> Tensor {
    let mut out = Tensor::zeros(input.h, input.w, input.c);
    let pad = K / 2;
    for y in 0..input.h {
        for x in 0..input.w {
            for c in 0..input.c {
                let mut acc = 0.0;
                for ky in 0..K {
                    for kx in 0..K {
                        let iy = y as isize + ky as isize - pad as isize;
                        let ix = x as isize + kx as isize - pad as isize;
                        if iy < 0 || ix < 0 || iy >= input.h as isize || ix >= input.w as isize {
                            continue;
                        }
                        let pixel = input.data[input.idx(iy as usize, ix as usize, c)];
                        let w_idx = c * K * K + ky * K + kx;
                        acc += pixel * weights[w_idx];
                    }
                }
                let oi = out.idx(y, x, c);
                out.data[oi] = acc.max(0.0);
            }
        }
    }
    out
}

// 逐点 1x1 卷积：混合通道。它与上面的深度卷积共同组成一个 MobileNet 块，
// 计算成本约为完整 HxWxC_in x C_out 3x3 稠密卷积的 1/8 至 1/9。
fn pointwise_conv(input: &Tensor, weights: &[f32], c_out: usize) -> Tensor {
    let mut out = Tensor::zeros(input.h, input.w, c_out);
    for y in 0..input.h {
        for x in 0..input.w {
            for co in 0..c_out {
                let mut acc = 0.0;
                for ci in 0..input.c {
                    let pixel = input.data[input.idx(y, x, ci)];
                    let w_idx = co * input.c + ci;
                    acc += pixel * weights[w_idx];
                }
                let oi = out.idx(y, x, co);
                out.data[oi] = acc.max(0.0);
            }
        }
    }
    out
}

fn forward(input: &Tensor, dw_w: &[f32], pw_w: &[f32]) -> Tensor {
    let dw = depthwise_conv(input, dw_w);
    pointwise_conv(&dw, pw_w, C_OUT)
}

fn flops_per_pass() -> u64 {
    let dw = (H * W * C_IN * K * K * 2) as u64;
    let pw = (H * W * C_IN * C_OUT * 2) as u64;
    dw + pw
}

fn percentile(sorted_ms: &[f64], pct: f64) -> f64 {
    if sorted_ms.is_empty() {
        return 0.0;
    }
    let idx = ((sorted_ms.len() as f64 - 1.0) * pct).round() as usize;
    sorted_ms[idx]
}

fn main() {
    let mut seed: u64 = 0xa1b2_c3d4_e5f6_0708;

    let mut input = Tensor::zeros(H, W, C_IN);
    fill_random(&mut input, &mut seed);

    let mut dw_weights = vec![0.0f32; C_IN * K * K];
    let mut pw_weights = vec![0.0f32; C_OUT * C_IN];
    for w in dw_weights.iter_mut() { *w = lcg(&mut seed) * 0.1; }
    for w in pw_weights.iter_mut() { *w = lcg(&mut seed) * 0.1; }

    println!();
    println!("=== 边缘推理基准测试（Rust，单线程）===");
    println!();
    println!("模型：深度 3x3 + 逐点 1x1（一个 MobileNet 块）");
    println!("输入形状：{}x{}x{}", H, W, C_IN);
    println!("输出通道：{}", C_OUT);
    let flops = flops_per_pass();
    println!("每次前向 FLOPs：{:.2} M", flops as f64 / 1e6);
    println!();

    println!("预热（{} 次迭代，不计入结果）...", WARMUP);
    for _ in 0..WARMUP {
        let _ = forward(&input, &dw_weights, &pw_weights);
    }

    println!("测量中（{} 次迭代）...", ITERS);
    let mut times_ms = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        let out = forward(&input, &dw_weights, &pw_weights);
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        times_ms.push(dt);
        std::hint::black_box(out);
    }

    let mut sorted = times_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = percentile(&sorted, 0.50);
    let p95 = percentile(&sorted, 0.95);
    let p99 = percentile(&sorted, 0.99);
    let mean: f64 = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    let min = sorted[0];
    let max = *sorted.last().unwrap();

    println!();
    println!("延迟（ms）：");
    println!("  p50   {:>8.2}", p50);
    println!("  p95   {:>8.2}", p95);
    println!("  p99   {:>8.2}", p99);
    println!("  均值  {:>8.2}", mean);
    println!("  最小  {:>8.2}", min);
    println!("  最大  {:>8.2}", max);

    let throughput_fps = 1000.0 / p50;
    let gflops_s = (flops as f64) / (p50 / 1000.0) / 1e9;
    println!();
    println!("吞吐量（根据 p50 计算）：");
    println!("  {:>5.1} fps   {:>5.2} GFLOPs/s", throughput_fps, gflops_s);

    println!();
    println!("边缘端测量规范（本程序也会执行）：");
    println!("  - 忽略 {} 次预热，以避免冷缓存偏差", WARMUP);
    println!("  - 固定输入分辨率（生产环境的分辨率必须一致）");
    println!("  - 同时报告 p50 和 p99，以呈现尾延迟");
    println!();
}
