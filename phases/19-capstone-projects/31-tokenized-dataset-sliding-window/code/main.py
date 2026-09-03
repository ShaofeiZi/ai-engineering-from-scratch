"""用于下一 token 训练的滑动窗口分词数据集。

把 tokenizer 编码后的 ID 流封装为 PyTorch Dataset 和 DataLoader，
让训练循环能够取得形状为 (B, T) 的输入与目标批次。

tokenizer 是第 30 课的小型字节级 BPE；这里将其内联，使本课无需跨课程导入即可运行。

运行：python3 code/main.py
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset


BYTE_ALPHABET_SIZE = 256
DEFAULT_SPECIALS = ("<|endoftext|>", "<|pad|>")
WORD_SPLIT_RE = re.compile(r"\S+|\s+")


@dataclass
class MiniBPE:
    """内联的字节级 BPE tokenizer（契约与第 30 课相同）。"""

    vocab: dict[int, bytes] = field(default_factory=dict)
    inv_vocab: dict[bytes, int] = field(default_factory=dict)
    merges: dict[tuple[int, int], int] = field(default_factory=dict)
    special_to_id: dict[str, int] = field(default_factory=dict)
    id_to_special: dict[int, str] = field(default_factory=dict)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def initialize(self, specials: Iterable[str] = DEFAULT_SPECIALS) -> None:
        self.vocab.clear()
        self.inv_vocab.clear()
        self.merges.clear()
        self.special_to_id.clear()
        self.id_to_special.clear()
        for i in range(BYTE_ALPHABET_SIZE):
            self.vocab[i] = bytes([i])
            self.inv_vocab[bytes([i])] = i
        for s in specials:
            token_id = len(self.vocab)
            self.vocab[token_id] = s.encode("utf-8")
            self.inv_vocab[s.encode("utf-8")] = token_id
            self.special_to_id[s] = token_id
            self.id_to_special[token_id] = s


def _pretokenize(text: str) -> list[str]:
    return WORD_SPLIT_RE.findall(text)


def _count_pairs(units: dict[tuple[int, ...], int]) -> Counter:
    pairs: Counter = Counter()
    for symbols, count in units.items():
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += count
    return pairs


def _apply_merge_to_corpus(
    units: dict[tuple[int, ...], int],
    pair: tuple[int, int],
    new_id: int,
) -> dict[tuple[int, ...], int]:
    new_units: dict[tuple[int, ...], int] = {}
    for symbols, count in units.items():
        if len(symbols) < 2:
            new_units[symbols] = new_units.get(symbols, 0) + count
            continue
        out: list[int] = []
        i = 0
        a, b = pair
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        merged = tuple(out)
        new_units[merged] = new_units.get(merged, 0) + count
    return new_units


def train_bpe(tokenizer: MiniBPE, corpus: str, target_vocab_size: int) -> None:
    min_vocab_size = BYTE_ALPHABET_SIZE + len(DEFAULT_SPECIALS)
    if target_vocab_size < min_vocab_size:
        raise ValueError(
            f"target_vocab_size must be >= {min_vocab_size}, got {target_vocab_size}"
        )
    tokenizer.initialize(DEFAULT_SPECIALS)
    chunks = _pretokenize(corpus)
    units: dict[tuple[int, ...], int] = {}
    for chunk in chunks:
        symbols = tuple(chunk.encode("utf-8"))
        units[symbols] = units.get(symbols, 0) + 1
    while tokenizer.vocab_size < target_vocab_size:
        pairs = _count_pairs(units)
        if not pairs:
            break
        max_count = max(pairs.values())
        candidates = sorted(p for p, c in pairs.items() if c == max_count)
        best = candidates[0]
        if pairs[best] < 2:
            break
        new_id = len(tokenizer.vocab)
        merged_bytes = tokenizer.vocab[best[0]] + tokenizer.vocab[best[1]]
        tokenizer.vocab[new_id] = merged_bytes
        tokenizer.inv_vocab[merged_bytes] = new_id
        tokenizer.merges[best] = new_id
        units = _apply_merge_to_corpus(units, best, new_id)


def encode_text(tokenizer: MiniBPE, text: str) -> list[int]:
    ranked = {pair: rank for rank, pair in enumerate(tokenizer.merges.keys())}
    out: list[int] = []
    for chunk in _pretokenize(text):
        symbols: list[int] = list(chunk.encode("utf-8"))
        while len(symbols) >= 2:
            best_rank = None
            best_index = -1
            best_pair: tuple[int, int] | None = None
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = ranked.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_index = i
                    best_pair = pair
            if best_pair is None:
                break
            new_id = tokenizer.merges[best_pair]
            symbols = symbols[:best_index] + [new_id] + symbols[best_index + 2:]
        out.extend(symbols)
    return out


class SlidingWindowDataset(Dataset):
    """基于扁平 ID 流的 PyTorch Dataset。

    每个样本都是大小为 T+1 的窗口。__getitem__ 返回
    (input_ids, target_ids)，其中 target 是左移一位的 input。
    """

    def __init__(
        self,
        ids: list[int],
        context_length: int,
        stride: int | None = None,
    ) -> None:
        if context_length < 1:
            raise ValueError(f"context_length 必须 >= 1，实际为 {context_length}")
        if not ids:
            raise ValueError("ids 不能为空")
        if stride is None:
            stride = context_length
        if stride < 1:
            raise ValueError(f"stride 必须 >= 1，实际为 {stride}")
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.context_length = context_length
        self.stride = stride

    @staticmethod
    def count_windows(num_ids: int, context_length: int, stride: int) -> int:
        usable = num_ids - (context_length + 1)
        if usable < 0:
            return 0
        return 1 + usable // stride

    def __len__(self) -> int:
        return self.count_windows(self.ids.numel(), self.context_length, self.stride)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"窗口索引 {index} 越界")
        start = index * self.stride
        end = start + self.context_length + 1
        window = self.ids[start:end]
        return window[:-1].clone(), window[1:].clone()


def make_dataloader(
    dataset: SlidingWindowDataset,
    batch_size: int,
    shuffle: bool = True,
    base_seed: int = 0,
    epoch: int = 0,
    drop_last: bool = True,
) -> DataLoader:
    """构建每轮确定性打乱的 DataLoader。"""
    generator = torch.Generator()
    generator.manual_seed(base_seed + epoch)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        generator=generator if shuffle else None,
        num_workers=0,
    )


def _encode_corpus_to_ids(tokenizer: MiniBPE, corpus: str, target_vocab: int) -> list[int]:
    train_bpe(tokenizer, corpus, target_vocab_size=target_vocab)
    return encode_text(tokenizer, corpus)


DEMO_CORPUS = """\
the quick brown fox jumps over the lazy dog
a journey of a thousand miles begins with a single step
the only way to do great work is to love what you do
the best time to plant a tree was twenty years ago
the second best time is now
practice is the bridge between intention and skill
small daily actions compound into large outcomes
read more than you write write more than you talk
the map is not the territory and the menu is not the meal
what gets measured gets managed if the measurement is honest
the quick brown fox runs across the meadow at dawn
a small step today is better than a perfect plan tomorrow
courage is not the absence of fear it is action despite fear
the lazy dog sleeps under the old oak tree
every expert was once a beginner who refused to quit
focus is saying no to a hundred good ideas
the river that you cannot cross today will be easier tomorrow
practice the basics until the basics become invisible
""" * 8


def _print_section(title: str) -> None:
    bar = "-" * len(title)
    print(f"\n{title}\n{bar}")


def main() -> int:
    target_vocab = 320
    context_length = 16
    stride = 8
    batch_size = 4
    base_seed = 7

    tokenizer = MiniBPE()
    ids = _encode_corpus_to_ids(tokenizer, DEMO_CORPUS, target_vocab)

    _print_section("语料库与分词器")
    print(f"语料字符数        : {len(DEMO_CORPUS)}")
    print(f"词表大小          : {tokenizer.vocab_size}")
    print(f"ID 总数           : {len(ids)}")

    dataset = SlidingWindowDataset(ids, context_length=context_length, stride=stride)
    print(f"上下文长度        : {context_length}")
    print(f"步幅              : {stride}")
    print(f"窗口数量          : {len(dataset)}")
    expected = SlidingWindowDataset.count_windows(len(ids), context_length, stride)
    assert len(dataset) == expected, "len(dataset) 必须等于 count_windows"

    _print_section("检查单个样本")
    input_ids, target_ids = dataset[0]
    print(f"输入形状          : {tuple(input_ids.shape)}")
    print(f"目标形状          : {tuple(target_ids.shape)}")
    assert input_ids.shape == target_ids.shape, "形状必须一致"
    assert torch.equal(input_ids[1:], target_ids[:-1]), "目标必须是向后平移一位的输入"

    _print_section("从 DataLoader 取出一个批次")
    loader = make_dataloader(dataset, batch_size=batch_size, base_seed=base_seed, epoch=0)
    inputs, targets = next(iter(loader))
    print(f"输入批次形状      : {tuple(inputs.shape)}")
    print(f"目标批次形状      : {tuple(targets.shape)}")
    print(f"首行输入          : {inputs[0].tolist()}")
    print(f"首行目标          : {targets[0].tolist()}")
    assert inputs.shape == (batch_size, context_length)
    assert targets.shape == (batch_size, context_length)

    _print_section("带种子的随机打乱")
    loader_a = make_dataloader(dataset, batch_size=batch_size, base_seed=base_seed, epoch=0)
    loader_b = make_dataloader(dataset, batch_size=batch_size, base_seed=base_seed, epoch=0)
    batch_a = next(iter(loader_a))
    batch_b = next(iter(loader_b))
    assert torch.equal(batch_a[0], batch_b[0]), "相同种子必须产生相同的首批数据"
    print("相同种子 -> 相同首批数据：成功")

    loader_c = make_dataloader(dataset, batch_size=batch_size, base_seed=base_seed, epoch=1)
    batch_c = next(iter(loader_c))
    assert not torch.equal(batch_a[0], batch_c[0]), "不同轮次必须改变顺序"
    print("不同轮次 -> 不同顺序：成功")

    _print_section("步幅权衡")
    for s in (4, 8, 16):
        ds = SlidingWindowDataset(ids, context_length=context_length, stride=s)
        print(f"  步幅 {s:>2}: {len(ds):>4} 个窗口")

    print("\n演示成功。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
