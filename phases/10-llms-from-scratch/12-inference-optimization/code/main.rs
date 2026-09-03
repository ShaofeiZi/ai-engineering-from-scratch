// 推理优化：KV cache 与投机解码示例，仅使用 Rust 标准库。
// 主题：prefill 与 decode、KV cache 内存布局、前缀缓存 trie、draft-verify 循环。
// 参考资料（仅引用思路，不作为依赖）：
//   - vLLM PagedAttention（Kwon 2023）：https://arxiv.org/abs/2309.06180
//   - 投机解码（Leviathan）：https://arxiv.org/abs/2211.17192
//   - candle KV cache：https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/llama.rs
//   - llm.c 推理说明：https://github.com/karpathy/llm.c
//
// 编译并运行：rustc --edition 2021 main.rs -o /tmp/inf && /tmp/inf

use std::collections::HashMap;
use std::f32::consts::PI;

// ---------- xorshift64 RNG（确定性，低位分布良好）----------
struct Rng { state: u64 }
impl Rng {
    fn new(seed: u64) -> Self {
        let mut s = seed;
        if s == 0 { s = 0xdead_beef_cafe_babe; }
        Rng { state: s }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    fn next_u32(&mut self) -> u32 { (self.next_u64() >> 32) as u32 }
    fn uniform(&mut self) -> f32 { (self.next_u32() as f32 + 1.0) / (u32::MAX as f32 + 2.0) }
    fn gauss(&mut self) -> f32 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
    fn range(&mut self, hi: usize) -> usize { (self.next_u32() as usize) % hi }
    fn choice(&mut self, probs: &[f32]) -> usize {
        let r = self.uniform();
        let mut acc = 0.0;
        for (i, p) in probs.iter().enumerate() {
            acc += *p;
            if r <= acc { return i; }
        }
        probs.len() - 1
    }
}

// ---------- KVCache：按 [num_layers, num_heads, max_seq, head_dim] 分层 ----------
struct KVCache {
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    bytes_per_element: usize,
    k: Vec<f32>,
    v: Vec<f32>,
    seq_len: usize,
}

impl KVCache {
    fn new(num_layers: usize, num_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let total = num_layers * num_heads * max_seq_len * head_dim;
        KVCache {
            num_layers, num_heads, head_dim, max_seq_len,
            bytes_per_element: 2, // 模拟 fp16
            k: vec![0.0; total],
            v: vec![0.0; total],
            seq_len: 0,
        }
    }

    fn idx(&self, layer: usize, head: usize, pos: usize, dim: usize) -> usize {
        ((layer * self.num_heads + head) * self.max_seq_len + pos) * self.head_dim + dim
    }

    // 为一层写入形状为 [n_new, num_heads, head_dim] 的新 K/V 切片。
    fn update(&mut self, layer: usize, new_k: &[f32], new_v: &[f32], n_new: usize) {
        assert_eq!(new_k.len(), n_new * self.num_heads * self.head_dim);
        assert_eq!(new_v.len(), n_new * self.num_heads * self.head_dim);
        assert!(layer < self.num_layers, "层索引超出范围");
        let start = self.seq_len;
        assert!(start + n_new <= self.max_seq_len, "超出 KV cache 容量");
        for t in 0..n_new {
            for h in 0..self.num_heads {
                for d in 0..self.head_dim {
                    let src = (t * self.num_heads + h) * self.head_dim + d;
                    let dst = self.idx(layer, h, start + t, d);
                    self.k[dst] = new_k[src];
                    self.v[dst] = new_v[src];
                }
            }
        }
    }

    fn advance(&mut self, n: usize) { self.seq_len += n; }

    fn capacity_bytes(&self) -> usize {
        2 * self.k.len() * self.bytes_per_element
    }
    fn used_bytes(&self) -> usize {
        let per_tok = 2 * self.num_layers * self.num_heads * self.head_dim * self.bytes_per_element;
        per_tok * self.seq_len
    }
}

// ---------- 前缀缓存 trie（PagedAttention 风格的前缀共享）----------
struct TrieNode {
    children: HashMap<usize, usize>, // token -> 节点索引
    hit_count: usize,
}

struct PrefixCache {
    nodes: Vec<TrieNode>,
    max_entries: usize,
    hits: usize,
    misses: usize,
}

impl PrefixCache {
    fn new(max_entries: usize) -> Self {
        PrefixCache {
            nodes: vec![TrieNode { children: HashMap::new(), hit_count: 0 }],
            max_entries,
            hits: 0,
            misses: 0,
        }
    }

    fn walk(&self, tokens: &[usize]) -> usize {
        let mut node = 0usize;
        let mut depth = 0usize;
        for &t in tokens {
            match self.nodes[node].children.get(&t) {
                Some(&next) => { node = next; depth += 1; }
                None => break,
            }
        }
        depth
    }

    fn lookup(&mut self, tokens: &[usize]) -> usize {
        let depth = self.walk(tokens);
        if depth > 0 {
            self.hits += 1;
            let mut node = 0usize;
            for &t in tokens.iter().take(depth) {
                node = *self.nodes[node].children.get(&t).unwrap();
                self.nodes[node].hit_count += 1;
            }
        } else {
            self.misses += 1;
        }
        depth
    }

    fn insert(&mut self, tokens: &[usize]) -> usize {
        let mut node = 0usize;
        for (i, &t) in tokens.iter().enumerate() {
            if !self.nodes[node].children.contains_key(&t) {
                if self.nodes.len() >= self.max_entries { return i; }
                let new_idx = self.nodes.len();
                self.nodes.push(TrieNode { children: HashMap::new(), hit_count: 0 });
                self.nodes[node].children.insert(t, new_idx);
            }
            node = *self.nodes[node].children.get(&t).unwrap();
        }
        tokens.len()
    }

    fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f32 / total as f32 }
    }
}

// ---------- 批处理模拟器 ----------
#[derive(Clone)]
struct Request {
    arrival: usize,
    output_tokens: usize,
    tokens_generated: usize,
    start: usize,
    end: usize,
}
impl Request {
    fn new(arrival: usize, output_tokens: usize) -> Self {
        Request { arrival, output_tokens, tokens_generated: 0, start: 0, end: 0 }
    }
    fn done(&self) -> bool { self.tokens_generated >= self.output_tokens }
}

fn simulate_static_batching(mut reqs: Vec<Request>, batch_size: usize) -> Vec<Request> {
    reqs.sort_by_key(|r| r.arrival);
    let mut step = 0;
    let mut completed = Vec::new();
    let mut idx = 0;
    while idx < reqs.len() {
        let mut batch: Vec<Request> = Vec::new();
        while idx < reqs.len() && batch.len() < batch_size {
            let mut r = reqs[idx].clone();
            r.start = step.max(r.arrival);
            batch.push(r);
            idx += 1;
        }
        if !batch.is_empty() {
            step = step.max(batch.iter().map(|r| r.start).max().unwrap());
            let max_out = batch.iter().map(|r| r.output_tokens).max().unwrap();
            for mut r in batch.into_iter() {
                r.tokens_generated = r.output_tokens;
                r.end = step + max_out;
                completed.push(r);
            }
            step += max_out;
        }
    }
    completed
}

fn simulate_continuous_batching(mut reqs: Vec<Request>, batch_size: usize) -> Vec<Request> {
    reqs.sort_by_key(|r| r.arrival);
    let mut step = 0usize;
    let mut completed = Vec::new();
    let mut waiting: Vec<Request> = Vec::new();
    let mut active: Vec<Request> = Vec::new();
    let mut idx = 0;

    while idx < reqs.len() || !active.is_empty() || !waiting.is_empty() {
        while idx < reqs.len() && reqs[idx].arrival <= step {
            waiting.push(reqs[idx].clone());
            idx += 1;
        }
        while !waiting.is_empty() && active.len() < batch_size {
            let mut r = waiting.remove(0);
            r.start = step;
            active.push(r);
        }
        if active.is_empty() {
            if !waiting.is_empty() { step += 1; continue; }
            if idx < reqs.len() { step = reqs[idx].arrival; continue; }
            break;
        }
        for r in active.iter_mut() { r.tokens_generated += 1; }
        let mut still: Vec<Request> = Vec::new();
        for mut r in active.drain(..) {
            if r.done() {
                r.end = step + 1;
                completed.push(r);
            } else {
                still.push(r);
            }
        }
        active = still;
        step += 1;
    }
    completed
}

struct BatchStats {
    avg_latency: f32,
    p50: f32,
    p99: f32,
    total_time: f32,
    throughput: f32,
}

fn batch_stats(completed: &[Request]) -> BatchStats {
    let mut lats: Vec<f32> = completed.iter().map(|r| (r.end - r.arrival) as f32).collect();
    lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg = lats.iter().sum::<f32>() / lats.len() as f32;
    let p50 = lats[lats.len() / 2];
    let p99 = lats[((lats.len() as f32 * 0.99) as usize).min(lats.len() - 1)];
    let total = completed.iter().map(|r| r.end).max().unwrap() as f32
        - completed.iter().map(|r| r.arrival).min().unwrap() as f32;
    let total_tokens: usize = completed.iter().map(|r| r.output_tokens).sum();
    let thr = if total > 0.0 { total_tokens as f32 / total } else { 0.0 };
    BatchStats { avg_latency: avg, p50, p99, total_time: total, throughput: thr }
}

// ---------- 投机解码示例 ----------
struct DraftModel { vocab: usize, acceptance_rate: f32 }
struct TargetModel { vocab: usize }

impl DraftModel {
    fn generate(&self, k: usize, rng: &mut Rng) -> Vec<usize> {
        (0..k).map(|_| rng.range(self.vocab)).collect()
    }
}

impl TargetModel {
    // 返回均匀概率向量；真实的目标模型会从自身的实际分布中采样。
    fn uniform_probs(&self) -> Vec<f32> { vec![1.0 / self.vocab as f32; self.vocab] }
}

#[allow(dead_code)]
struct SpecResult {
    total_tokens: usize,
    spec_cost: f32,
    seq_cost: f32,
    speedup: f32,
    avg_accepted: f32,
}

fn speculative_decode(
    draft: &DraftModel, target: &TargetModel,
    context: &[usize], num_spec: usize,
    draft_cost: f32, target_cost: f32, verify_cost: f32,
    max_tokens: usize,
    rng: &mut Rng,
) -> SpecResult {
    let mut ctx: Vec<usize> = context.to_vec();
    let mut total_tokens = 0usize;
    let mut total_cost = 0.0f32;
    let mut accepted_counts: Vec<usize> = Vec::new();

    while total_tokens < max_tokens {
        let draft_tokens = draft.generate(num_spec, rng);
        total_cost += draft_cost * num_spec as f32;

        // 一次验证前向传播同时为全部 k 个 token 打分。
        total_cost += verify_cost;

        let mut accepted = 0usize;
        for &tok in &draft_tokens {
            if total_tokens >= max_tokens { break; }
            let r = rng.uniform();
            if r < draft.acceptance_rate {
                accepted += 1;
                ctx.push(tok);
                total_tokens += 1;
            } else {
                let probs = target.uniform_probs();
                let resampled = rng.choice(&probs);
                ctx.push(resampled);
                total_tokens += 1;
                break;
            }
        }
        accepted_counts.push(accepted);

        if accepted == num_spec && total_tokens < max_tokens {
            // 使用目标模型的独立预测补充一个 token。
            let probs = target.uniform_probs();
            let bonus = rng.choice(&probs);
            ctx.push(bonus);
            total_tokens += 1;
        }
    }
    let seq_cost = total_tokens as f32 * target_cost;
    let avg_accept = accepted_counts.iter().sum::<usize>() as f32 / accepted_counts.len() as f32;
    SpecResult {
        total_tokens,
        spec_cost: total_cost,
        seq_cost,
        speedup: if total_cost > 0.0 { seq_cost / total_cost } else { 1.0 },
        avg_accepted: avg_accept,
    }
}

// ---------- KV cache 内存分析 ----------
#[allow(dead_code)]
struct ModelCfg {
    name: &'static str,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    params_b: f64,
}

fn kv_cache_mem(cfg: &ModelCfg, seq_len: usize, bytes: usize) -> (usize, f64) {
    let per_token = 2 * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim * bytes;
    let total = per_token * seq_len;
    (per_token, total as f64 / (1024.0 * 1024.0 * 1024.0))
}

fn main() {
    let mut rng = Rng::new(42);

    // 第 1 步：KV cache 内存分析
    println!("{}", "=".repeat(70));
    println!("步骤 1：各模型的 KV cache 内存");
    println!("{}", "=".repeat(70));
    let configs: [ModelCfg; 5] = [
        ModelCfg { name: "Llama-3-8B",   num_layers: 32, num_kv_heads: 8, head_dim: 128, params_b: 8.0 },
        ModelCfg { name: "Llama-3-70B",  num_layers: 80, num_kv_heads: 8, head_dim: 128, params_b: 70.0 },
        ModelCfg { name: "Llama-3-405B", num_layers: 126, num_kv_heads: 8, head_dim: 128, params_b: 405.0 },
        ModelCfg { name: "Mistral-7B",   num_layers: 32, num_kv_heads: 8, head_dim: 128, params_b: 7.0 },
        ModelCfg { name: "GPT-4-est",    num_layers: 120, num_kv_heads: 96, head_dim: 128, params_b: 1800.0 },
    ];
    println!("  {:<20} {:>12} {:>12} {:>12} {:>12}", "模型", "每个 token", "@ 4K 上下文", "@ 32K 上下文", "@ 128K 上下文");
    println!("  {}", "-".repeat(70));
    for c in &configs {
        let (pt, _) = kv_cache_mem(c, 1, 2);
        let (_, g4) = kv_cache_mem(c, 4096, 2);
        let (_, g32) = kv_cache_mem(c, 32768, 2);
        let (_, g128) = kv_cache_mem(c, 131072, 2);
        println!("{:<20} {:>10}KB {:>10.2}GB {:>10.2}GB {:>10.2}GB",
            c.name, pt / 1024, g4, g32, g128);
    }

    // 第 2 步：模拟 Attention 写入 KV cache
    println!("\n{}", "=".repeat(70));
    println!("步骤 2：KV cache 的 prefill 与 decode");
    println!("{}", "=".repeat(70));
    let num_heads = 4usize;
    let head_dim = 16usize;
    let seq_len = 8usize;
    let mut cache = KVCache::new(1, num_heads, head_dim, 128);

    // 构造用于 prefill 的模拟 K/V 张量。
    let n_prefill = seq_len;
    let kv_size = n_prefill * num_heads * head_dim;
    let k: Vec<f32> = (0..kv_size).map(|_| rng.gauss()).collect();
    let v: Vec<f32> = (0..kv_size).map(|_| rng.gauss()).collect();
    cache.update(0, &k, &v, n_prefill);
    cache.advance(n_prefill);
    println!("预填充：缓存 {} 个 token，已用 {} 字节（容量 {} 字节）",
        cache.seq_len, cache.used_bytes(), cache.capacity_bytes());

    // 解码 4 步，每步附加 1 个 token 的 K/V。
    for step in 0..4 {
        let kv_size = num_heads * head_dim;
        let k_new: Vec<f32> = (0..kv_size).map(|_| rng.gauss()).collect();
        let v_new: Vec<f32> = (0..kv_size).map(|_| rng.gauss()).collect();
        cache.update(0, &k_new, &v_new, 1);
        cache.advance(1);
        println!("解码步骤 {}：缓存 {} 个 token，已用 {} 字节",
            step + 1, cache.seq_len, cache.used_bytes());
    }

    // 第 3 步：静态批处理与连续批处理
    println!("\n{}", "=".repeat(70));
    println!("步骤 3：静态批处理与连续批处理");
    println!("{}", "=".repeat(70));

    let make_reqs = |seed: u64, n: usize| -> Vec<Request> {
        let mut r = Rng::new(seed);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let arrival = r.range(20);
            // 通过逆变换采样构造 Pareto 式长尾输出长度。
            let u = r.uniform().max(1e-3);
            let out_len = ((1.0 / u.powf(1.0 / 1.5)) * 15.0) as usize + 5;
            let out_len = out_len.min(200);
            out.push(Request::new(arrival, out_len));
        }
        out
    };
    let batch_size = 8usize;
    let s = simulate_static_batching(make_reqs(42, 30), batch_size);
    let c = simulate_continuous_batching(make_reqs(42, 30), batch_size);
    let ss = batch_stats(&s);
    let cs = batch_stats(&c);
    println!("30 个请求，batch_size={}", batch_size);
    println!("  {:<14} {:>12} {:>12} {:>12}", "指标", "静态", "连续", "变化");
    println!("  {}", "-".repeat(54));
    let print_delta = |name: &str, sv: f32, cv: f32, smaller_better: bool| {
        let delta = if smaller_better {
            if sv > 0.0 { format!("{:+.1}%", (sv - cv) / sv * 100.0) } else { "n/a".to_string() }
        } else {
            if sv > 0.0 { format!("{:.2}x", cv / sv) } else { "n/a".to_string() }
        };
        println!("  {:<14} {:>12.1} {:>12.1} {:>12}", name, sv, cv, delta);
    };
    print_delta("avg_latency", ss.avg_latency, cs.avg_latency, true);
    print_delta("p50_latency", ss.p50, cs.p50, true);
    print_delta("p99_latency", ss.p99, cs.p99, true);
    print_delta("total_time",  ss.total_time, cs.total_time, true);
    print_delta("throughput",  ss.throughput, cs.throughput, false);

    // 第 4 步：前缀缓存
    println!("\n{}", "=".repeat(70));
    println!("步骤 4：共享系统提示的前缀缓存");
    println!("{}", "=".repeat(70));
    let mut pc = PrefixCache::new(5000);
    let prompts: Vec<Vec<usize>> = vec![
        (100..200).collect(),
        (200..350).collect(),
        (400..480).collect(),
    ];
    for (i, p) in prompts.iter().enumerate() {
        let inserted = pc.insert(p);
        println!("已缓存系统提示 {}：{} 个 token，插入 {} 个新节点", i + 1, p.len(), inserted);
    }

    let mut hit_count = 0usize;
    let mut tokens_saved = 0usize;
    for _ in 0..100 {
        let idx = rng.range(prompts.len());
        let sys = &prompts[idx];
        let user_len = 20 + rng.range(30);
        let mut full = sys.clone();
        full.extend((0..user_len).map(|_| 500 + rng.range(500)));
        let depth = pc.lookup(&full);
        if depth > 0 { hit_count += 1; tokens_saved += depth; }
    }
    println!("  命中率：{:.1}%", pc.hit_rate() * 100.0);
    println!("节省的 token 数（复用前缀）：{}", tokens_saved);
    println!("每次命中平均节省的 token 数：{:.1}", tokens_saved as f32 / hit_count.max(1) as f32);

    // 第 5 步：投机解码
    println!("\n{}", "=".repeat(70));
    println!("步骤 5：投机解码速度（简化模型）");
    println!("{}", "=".repeat(70));
    let vocab = 500usize;
    let trials = 10usize;
    let strategies: [(&str, f32, usize); 3] = [
        ("draft-target (8B->70B)", 0.78, 5),
        ("EAGLE",                  0.85, 6),
        ("n-gram 查找",            0.50, 4),
    ];
    println!("  {:<24} {:>14} {:>12} {:>10}", "策略", "接受率", "平均接受数", "加速比");
    println!("  {}", "-".repeat(64));
    for (name, acc, k) in strategies {
        let mut speedups = 0.0f32;
        let mut accept_rates = 0.0f32;
        let mut avg_accepts = 0.0f32;
        for _ in 0..trials {
            let draft = DraftModel { vocab, acceptance_rate: acc };
            let target = TargetModel { vocab };
            let ctx: Vec<usize> = (0..10).map(|_| rng.range(vocab)).collect();
            let r = speculative_decode(&draft, &target, &ctx, k, 1.0, 10.0, 12.0, 100, &mut rng);
            speedups += r.speedup;
            accept_rates += r.avg_accepted / k as f32;
            avg_accepts += r.avg_accepted;
        }
        println!("  {:<24} {:>13.1}% {:>12.2} {:>9.2}x",
            name,
            accept_rates / trials as f32 * 100.0,
            avg_accepts / trials as f32,
            speedups / trials as f32,
        );
    }

    // 第 6 步：Ops:Byte 分析
    println!("\n{}", "=".repeat(70));
    println!("步骤 6：Ops:Byte 与内存/计算瓶颈");
    println!("{}", "=".repeat(70));
    let a100_tflops = 312.0f32;
    let a100_bandwidth_tbs = 2.0f32;
    let crossover = a100_tflops / a100_bandwidth_tbs;
    println!("A100 规格：{} TFLOPS，{} TB/s 带宽，Ops:Byte 交叉点={:.0}",
        a100_tflops, a100_bandwidth_tbs, crossover);
    let scenarios: [(&str, usize); 7] = [
        ("Prefill，batch=1，seq=4096", 4096),
        ("解码，batch=1",   1),
        ("解码，batch=8",   8),
        ("解码，batch=32",  32),
        ("解码，batch=128", 128),
        ("解码，batch=256", 256),
        ("解码，batch=512", 512),
    ];
    println!("  {:<32} {:>10} {:>12} {:>12}", "场景", "Ops:Byte", "瓶颈", "利用率");
    println!("  {}", "-".repeat(70));
    for (name, opb) in scenarios {
        let bound = if opb as f32 >= crossover { "Compute" } else { "Memory" };
        let util = if bound == "Memory" { opb as f32 / crossover * 100.0 } else { 100.0 };
        let bound_label = if bound == "Compute" { "计算" } else { "内存" };
        println!("  {:<32} {:>10} {:>12} {:>11.1}%", name, opb, bound_label, util);
    }

    println!("\n{}", "=".repeat(70));
    println!("总结");
    println!("{}", "=".repeat(70));
    println!("  1. KV cache 以空间换计算；每个 token 的成本随 layers × kv_heads × head_dim 增长。");
    println!("  2. 请求在批次中途完成后，连续批处理仍能让 GPU 保持忙碌。");
    println!("  3. 前缀缓存可在共享系统提示之间复用 KV 条目。");
    println!("  4. 投机解码把验证成本摊到 k 个 draft token 上。");
    println!("  5. 小批次解码受内存带宽限制；应扩大批次，直到 Ops:Byte 超过交叉点。");
}
