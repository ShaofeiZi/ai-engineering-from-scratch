// 位置编码：正弦编码、RoPE、ALiBi。仅使用标准库。
// 主题：将 token 位置编码到查询、键或注意力偏置中。
// 参考资料（仅作为引用，不作为依赖）：
//   - Vaswani 2017（正弦编码）：https://arxiv.org/abs/1706.03762
//   - Su 等（2021，RoPE）：https://arxiv.org/abs/2104.09864
//   - Press 等（2021，ALiBi）：https://arxiv.org/abs/2108.12409
//   - candle 的 RoPE 实现：https://github.com/huggingface/candle/blob/main/candle-nn/src/rotary_emb.rs
//
// 编译并运行：rustc --edition 2021 main.rs -o /tmp/pe && /tmp/pe

use std::f32::consts::PI;

// 正弦位置编码表 [n, d]。
fn sinusoidal_pe(n: usize, d: usize, base: f32) -> Vec<Vec<f32>> {
    let mut pe = vec![vec![0.0f32; d]; n];
    for pos in 0..n {
        for i in 0..(d / 2) {
            let theta = (pos as f32) / base.powf(2.0 * i as f32 / d as f32);
            pe[pos][2 * i] = theta.sin();
            pe[pos][2 * i + 1] = theta.cos();
        }
    }
    pe
}

// 将 x 的奇偶元素对旋转 pos * theta_i 角度，返回新的 Vec。
fn apply_rope(x: &[f32], pos: usize, base: f32) -> Vec<f32> {
    let d = x.len();
    let mut out = x.to_vec();
    for i in 0..(d / 2) {
        let theta = (pos as f32) / base.powf(2.0 * i as f32 / d as f32);
        let c = theta.cos();
        let s = theta.sin();
        let a = x[2 * i];
        let b = x[2 * i + 1];
        out[2 * i] = a * c - b * s;
        out[2 * i + 1] = a * s + b * c;
    }
    out
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ALiBi 斜率：当 h 位于 0..n_heads 时为 2^(-8*(h+1)/n_heads)。
fn alibi_slopes(n_heads: usize) -> Vec<f32> {
    (0..n_heads)
        .map(|h| 2.0f32.powf(-8.0 * (h + 1) as f32 / n_heads as f32))
        .collect()
}

// 各个头的 ALiBi 偏置矩阵：-slope * |i - j|，可选因果掩码。
fn alibi_bias(n_heads: usize, seq_len: usize, causal: bool) -> Vec<Vec<Vec<f32>>> {
    let slopes = alibi_slopes(n_heads);
    let mut out = Vec::with_capacity(n_heads);
    for &m in &slopes {
        let mut head = vec![vec![0.0f32; seq_len]; seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                head[i][j] = if causal && j > i {
                    f32::NEG_INFINITY
                } else {
                    -m * (i as i64 - j as i64).abs() as f32
                };
            }
        }
        out.push(head);
    }
    out
}

// 用于生成确定性高斯样本的微型 LCG。
struct Rng { state: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { state: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1 } }
    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.state >> 33) as u32
    }
    fn uniform(&mut self) -> f32 { (self.next_u32() as f32 + 1.0) / (u32::MAX as f32 + 2.0) }
    fn gauss(&mut self) -> f32 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

fn demo_sinusoidal() {
    println!("=== 正弦位置编码 ===");
    let pe = sinusoidal_pe(8, 8, 10000.0);
    println!("前 4 个位置、前 4 个维度：");
    for pos in 0..4 {
        print!("  pos={}: ", pos);
        for i in 0..4 {
            print!(" {:+.3}", pe[pos][i]);
        }
        println!();
    }
    println!();
}

fn demo_rope_relative() {
    println!("=== RoPE：点积仅取决于相对距离 ===");
    let mut rng = Rng::new(0);
    let d = 16usize;
    let q: Vec<f32> = (0..d).map(|_| rng.gauss()).collect();
    let k: Vec<f32> = (0..d).map(|_| rng.gauss()).collect();

    let pairs = [(3usize, 5usize), (7, 9), (100, 102), (1024, 1026)];
    println!(" {:>6}  {:>6}  {:>4}  {:>18}", "pos_q", "pos_k", "间隔", "<q_rot, k_rot>");
    for (pq, pk) in pairs {
        let q_rot = apply_rope(&q, pq, 10000.0);
        let k_rot = apply_rope(&k, pk, 10000.0);
        let d_prod = dot(&q_rot, &k_rot);
        println!(" {:>6}  {:>6}  {:>4}  {:>18.6}", pq, pk, (pk as i64) - (pq as i64), d_prod);
    }
    println!("间隔为 2 的所有行都具有相同点积。");
    println!();
}

fn demo_rope_base_scaling() {
    println!("=== RoPE 基数缩放（面向长上下文的 NTK-aware 方法）===");
    let mut rng = Rng::new(1);
    let d = 8usize;
    let q: Vec<f32> = (0..d).map(|_| rng.gauss()).collect();
    let k: Vec<f32> = (0..d).map(|_| rng.gauss()).collect();

    for base in [10_000.0f32, 100_000.0, 1_000_000.0] {
        let q_rot = apply_rope(&q, 4096, base);
        let k_rot = apply_rope(&k, 4098, base);
        println!("  基数={:>9}  得分={:+.6}", base as u64, dot(&q_rot, &k_rot));
    }
    println!("基数越大 = 旋转越慢 = 不发生相位环绕的上下文越长。");
    println!();
}

fn demo_alibi() {
    println!("=== ALiBi 偏置矩阵 ===");
    let n_heads = 4usize;
    let slopes = alibi_slopes(n_heads);
    print!("{} 个头的斜率：", n_heads);
    for s in &slopes { print!(" {:.4}", s); }
    println!();
    let bias = alibi_bias(n_heads, 6, false);
    println!("第 0 个头的偏置（token 越近，惩罚越小）：");
    for row in &bias[0] {
        print!(" ");
        for v in row { print!(" {:+6.2}", v); }
        println!();
    }
    println!();
}

fn demo_rope_microbench() {
    println!("=== 微基准测试：5 万次 RoPE 旋转（d=128）===");
    let mut rng = Rng::new(2);
    let d = 128usize;
    let q: Vec<f32> = (0..d).map(|_| rng.gauss()).collect();
    let start = std::time::Instant::now();
    let mut sink = 0.0f32;
    for pos in 0..50_000usize {
        let r = apply_rope(&q, pos, 10_000.0);
        sink += r[0];
    }
    let elapsed = start.elapsed();
    println!("5 万次旋转耗时 {:.2}ms（{:.0}/秒）  累加值={:.4}",
        elapsed.as_secs_f64() * 1000.0,
        50_000.0 / elapsed.as_secs_f64(),
        sink,
    );
}

fn main() {
    demo_sinusoidal();
    demo_rope_relative();
    demo_rope_base_scaling();
    demo_alibi();
    demo_rope_microbench();
    println!();
    println!("要点：RoPE 在点积本身中编码相对位置。");
    println!("ALiBi 完全跳过嵌入。如今正弦位置编码大多只具有历史意义。");
}
