"""流式分词并写入可调整大小、可分片且支持 mmap 读取的 HDF5 数据集。

实现内容：
- 字节级确定性 Tokenizer。
- HDF5ShardWriter：把 token 缓冲到分块大小，按固定步长调整数据集大小，
  并将 token_count 和 sha256 记录为数据集属性。
- ShardedTokenizationPipeline：每个源分片生成一个 HDF5，并写入 shards.json 索引。
- MmapTokenStore：以 swmr 模式打开分片文件供读取。
- SlidingWindowDataloader：生成定长 (input, target) 对。

文件末尾的演示会构建内存语料库，将其分词到多个分片，通过内存映射打开分片，
让 dataloader 运行少量批次，并打印每批形状和校验和。运行：python3 code/main.py
"""

from __future__ import annotations

import hashlib
import json
import random
import struct
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

try:
    import h5py
except ImportError as exc:
    raise SystemExit(
        "本课需要 h5py。安装命令：pip install h5py"
    ) from exc


DEFAULT_CHUNK_SIZE = 8192
DEFAULT_WINDOW_SIZE = 64
BOUNDARY_TOKEN_ID = 0
TOKEN_DTYPE = np.uint16


@dataclass
class ShardWriteResult:
    """每个分片的写入结果。"""

    shard_id: str
    path: str
    token_count: int
    document_count: int
    chunk_size: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ShardIndexEntry:
    """读取器用于定位分片的索引行。"""

    shard_id: str
    path: str
    token_count: int
    document_count: int
    sha256: str
    global_start: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class Tokenizer:
    """字节级确定性 tokenizer。

    词表：
        0      边界 token（由 dataloader 注入的分隔符）
        1..256 原始字节 token（偏移一位以预留 0）

    真实 tokenizer 使用 BPE 或 SentencePiece；此实现足以演示流式写入，
    无需引入第三方 tokenizer。
    """

    BOUNDARY_TOKEN = BOUNDARY_TOKEN_ID
    BYTE_OFFSET = 1

    def __init__(self) -> None:
        self.vocab_size = 257

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        data = text.encode("utf-8")
        return [self.BYTE_OFFSET + b for b in data]

    def decode(self, ids: Iterable[int]) -> str:
        byte_ids = [int(i) - self.BYTE_OFFSET for i in ids if int(i) >= self.BYTE_OFFSET]
        return bytes(byte_ids).decode("utf-8", errors="replace")


class HDF5ShardWriter:
    """通过分块大小的缓冲，把 token 流式写入可调整大小的 HDF5 数据集。

    在 `with` 块中打开，以保证剩余缓冲区得到刷新，并写入结束属性
    （token_count、sha256）。
    """

    def __init__(
        self,
        path: Path,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        dataset_name: str = "tokens",
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须为正数")
        self.path = Path(path)
        self.chunk_size = chunk_size
        self.dataset_name = dataset_name
        self._buffer: list[int] = []
        self._token_count = 0
        self._document_count = 0
        self._hasher = hashlib.sha256()
        self._file: h5py.File | None = None
        self._dataset: h5py.Dataset | None = None

    def __enter__(self) -> "HDF5ShardWriter":
        self._file = h5py.File(self.path, "w", libver="latest")
        self._dataset = self._file.create_dataset(
            self.dataset_name,
            shape=(0,),
            maxshape=(None,),
            chunks=(self.chunk_size,),
            dtype=TOKEN_DTYPE,
        )
        self._file.swmr_mode = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self._dataset is not None and self._file is not None:
                if self._buffer:
                    self._flush_buffer(final=True)
                self._dataset.attrs["token_count"] = self._token_count
                self._dataset.attrs["document_count"] = self._document_count
                self._dataset.attrs["sha256"] = self._hasher.hexdigest()
        finally:
            if self._file is not None:
                self._file.close()
                self._file = None
                self._dataset = None

    def add_document(self, token_ids: Iterable[int]) -> None:
        self._document_count += 1
        for token in token_ids:
            self._buffer.append(int(token))
            if len(self._buffer) >= self.chunk_size:
                self._flush_buffer(final=False)

    def add_boundary(self) -> None:
        """在文档之间插入分隔 token。"""

        self._buffer.append(BOUNDARY_TOKEN_ID)
        if len(self._buffer) >= self.chunk_size:
            self._flush_buffer(final=False)

    def _flush_buffer(self, final: bool) -> None:
        if self._dataset is None:
            raise RuntimeError("writer 尚未打开")
        if not self._buffer:
            return
        size = len(self._buffer) if final else self.chunk_size
        chunk = np.asarray(self._buffer[:size], dtype=TOKEN_DTYPE)
        new_total = self._token_count + size
        self._dataset.resize((new_total,))
        self._dataset[self._token_count : new_total] = chunk
        self._dataset.flush()
        self._hasher.update(chunk.tobytes())
        self._token_count = new_total
        self._buffer = self._buffer[size:]
        if not final and len(self._buffer) >= self.chunk_size:
            self._flush_buffer(final=False)

    @property
    def token_count(self) -> int:
        return self._token_count

    @property
    def document_count(self) -> int:
        return self._document_count

    def result(self, shard_id: str) -> ShardWriteResult:
        return ShardWriteResult(
            shard_id=shard_id,
            path=str(self.path),
            token_count=self._token_count,
            document_count=self._document_count,
            chunk_size=self.chunk_size,
            sha256=self._hasher.hexdigest(),
        )


class ShardedTokenizationPipeline:
    """把可迭代分片输入分词到 HDF5 文件，并写入 shards.json。"""

    def __init__(
        self,
        tokenizer: Tokenizer,
        output_dir: Path,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size

    def write_shard(self, shard_id: str, documents: Iterable[str]) -> ShardWriteResult:
        shard_path = self.output_dir / f"{shard_id}.h5"
        writer = HDF5ShardWriter(shard_path, chunk_size=self.chunk_size)
        with writer:
            for text in documents:
                writer.add_document(self.tokenizer.encode(text))
                writer.add_boundary()
        return writer.result(shard_id)

    def write_corpus(self, shards: dict[str, Iterable[str]]) -> list[ShardIndexEntry]:
        entries: list[ShardIndexEntry] = []
        running_offset = 0
        for shard_id, documents in shards.items():
            result = self.write_shard(shard_id, documents)
            entries.append(
                ShardIndexEntry(
                    shard_id=result.shard_id,
                    path=result.path,
                    token_count=result.token_count,
                    document_count=result.document_count,
                    sha256=result.sha256,
                    global_start=running_offset,
                )
            )
            running_offset += result.token_count
        index_path = self.output_dir / "shards.json"
        body = {
            "version": 1,
            "chunk_size": self.chunk_size,
            "total_tokens": running_offset,
            "shards": [entry.to_dict() for entry in entries],
        }
        index_path.write_text(json.dumps(body, sort_keys=True, indent=2), encoding="utf-8")
        return entries


class MmapTokenStore:
    """通过内存映射读取分片 HDF5 token 语料库。

    存储以 SWMR 模式打开每个分片文件一次。`get_slice(start, stop)` 请求会跨分片
    路由，结果以扁平 NumPy uint16 数组返回。读取会进入页缓存；转换为训练张量时，
    dataloader 产生一次复制。
    """

    def __init__(self, shard_entries: list[ShardIndexEntry]) -> None:
        if not shard_entries:
            raise ValueError("至少需要一个分片条目")
        self._entries = shard_entries
        self._files: list[h5py.File] = []
        self._datasets: list[h5py.Dataset] = []
        try:
            for entry in shard_entries:
                self._files.append(h5py.File(entry.path, "r", swmr=True))
            self._datasets = [f["tokens"] for f in self._files]
        except Exception:
            for opened in self._files:
                try:
                    opened.close()
                except Exception:
                    pass
            self._files = []
            self._datasets = []
            raise
        self._total_tokens = sum(entry.token_count for entry in shard_entries)

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    def close(self) -> None:
        for file in self._files:
            try:
                file.close()
            except Exception:
                pass
        self._files = []
        self._datasets = []

    def __enter__(self) -> "MmapTokenStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def get_slice(self, start: int, stop: int) -> np.ndarray:
        if start < 0 or stop < 0 or stop < start:
            raise ValueError(f"切片无效：start={start} stop={stop}")
        if stop > self._total_tokens:
            raise ValueError(f"stop ({stop}) exceeds total tokens ({self._total_tokens})")
        if stop == start:
            return np.empty((0,), dtype=TOKEN_DTYPE)
        out = np.empty((stop - start,), dtype=TOKEN_DTYPE)
        cursor = 0
        for entry, dataset in zip(self._entries, self._datasets):
            shard_start = entry.global_start
            shard_stop = shard_start + entry.token_count
            if stop <= shard_start:
                break
            if start >= shard_stop:
                continue
            local_start = max(0, start - shard_start)
            local_stop = min(entry.token_count, stop - shard_start)
            length = local_stop - local_start
            if length <= 0:
                continue
            out[cursor : cursor + length] = dataset[local_start:local_stop]
            cursor += length
        if cursor != stop - start:
            raise RuntimeError(
                f"slice read produced {cursor} tokens, expected {stop - start}"
            )
        return out


class SlidingWindowDataloader:
    """基于扁平 token 流的随机滑动窗口采样器。"""

    def __init__(
        self,
        store: MmapTokenStore,
        window_size: int = DEFAULT_WINDOW_SIZE,
        batch_size: int = 4,
        seed: int = 0,
    ) -> None:
        if window_size <= 1:
            raise ValueError("window_size 必须大于 1")
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正数")
        if store.total_tokens <= window_size:
            raise ValueError(
                f"store has only {store.total_tokens} tokens; need more than {window_size}"
            )
        self.store = store
        self.window_size = window_size
        self.batch_size = batch_size
        self._random = random.Random(seed)
        self._max_start = store.total_tokens - window_size - 1

    def _sample_window(self) -> tuple[np.ndarray, np.ndarray]:
        start = self._random.randint(0, self._max_start)
        chunk = self.store.get_slice(start, start + self.window_size + 1)
        return chunk[:-1], chunk[1:]

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        while True:
            inputs = np.empty((self.batch_size, self.window_size), dtype=TOKEN_DTYPE)
            targets = np.empty((self.batch_size, self.window_size), dtype=TOKEN_DTYPE)
            for row in range(self.batch_size):
                inputs[row], targets[row] = self._sample_window()
            yield inputs, targets

    def take(self, num_batches: int) -> list[tuple[np.ndarray, np.ndarray]]:
        iterator = iter(self)
        return [next(iterator) for _ in range(num_batches)]


class JSONLSource:
    """使用可配置键从 JSONL 文件生成文档的适配器。

    下载器（阶段 19 · 42）生成的 JSONL 每行都是带 `text` 字段的 JSON 对象。
    此适配器取出文本，跳过格式错误或缺少该字段的行。真实管线会记录丢弃行；
    此适配器会计数，以便调用方审计丢弃率。
    """

    def __init__(self, path: Path, text_field: str = "text") -> None:
        self.path = Path(path)
        self.text_field = text_field
        self.dropped_lines = 0

    def __iter__(self) -> Iterator[str]:
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    self.dropped_lines += 1
                    continue
                if not isinstance(record, dict):
                    self.dropped_lines += 1
                    continue
                value = record.get(self.text_field)
                if not isinstance(value, str) or not value:
                    self.dropped_lines += 1
                    continue
                yield value


def pack_documents(
    tokenizer: Tokenizer,
    documents: Iterable[str],
    max_tokens: int,
) -> Iterator[list[int]]:
    """使用边界 token 把已分词文档打包成定长组。

    生成恰含 max_tokens 个 token ID 的列表。长文档会跨组拆分；短文档共享一组，
    并由 BOUNDARY_TOKEN_ID 分隔。最后一组可能短于 max_tokens，按原样生成。
    """

    if max_tokens <= 1:
        raise ValueError("max_tokens 必须大于 1")
    buffer: list[int] = []
    for text in documents:
        token_ids = tokenizer.encode(text)
        if buffer:
            buffer.append(BOUNDARY_TOKEN_ID)
        buffer.extend(token_ids)
        while len(buffer) >= max_tokens:
            yield buffer[:max_tokens]
            buffer = buffer[max_tokens:]
    if buffer:
        yield buffer


def tokenize_jsonl_path(
    jsonl_path: Path,
    output_dir: Path,
    shard_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    text_field: str = "text",
) -> ShardWriteResult:
    """便捷包装器：把一个 JSONL 文件分词到一个 HDF5 分片。"""

    tokenizer = Tokenizer()
    pipeline = ShardedTokenizationPipeline(tokenizer, output_dir=output_dir, chunk_size=chunk_size)
    source = JSONLSource(jsonl_path, text_field=text_field)
    return pipeline.write_shard(shard_id, source)


def load_index(index_path: Path) -> list[ShardIndexEntry]:
    """读取 shards.json 并返回 ShardIndexEntry 行。"""

    data = json.loads(Path(index_path).read_text("utf-8"))
    entries: list[ShardIndexEntry] = []
    for row in data["shards"]:
        entries.append(
            ShardIndexEntry(
                shard_id=str(row["shard_id"]),
                path=str(row["path"]),
                token_count=int(row["token_count"]),
                document_count=int(row.get("document_count", 0)),
                sha256=str(row["sha256"]),
                global_start=int(row["global_start"]),
            )
        )
    return entries


def validate_corpus(index_entries: list[ShardIndexEntry]) -> list[str]:
    """根据磁盘 token 重新计算各分片 sha256，并报告不匹配项。"""

    failures: list[str] = []
    for entry in index_entries:
        with h5py.File(entry.path, "r", swmr=True) as fh:
            dataset = fh["tokens"]
            recorded_count = int(dataset.attrs.get("token_count", entry.token_count))
            tokens = np.asarray(dataset[:recorded_count], dtype=TOKEN_DTYPE)
            recomputed = hashlib.sha256(tokens.tobytes()).hexdigest()
            if recomputed != entry.sha256:
                failures.append(entry.shard_id)
    return failures


def build_demo_corpus() -> dict[str, list[str]]:
    """两份足够长、可练习 mmap 读取的合成文档分片。"""

    base = [
        "the alignment problem is a story about reward functions and the things they fail to write down",
        "attention scales better with sequence length so transformers replaced recurrent networks during the language modeling era",
        "an evaluation harness keeps training honest by treating the test corpus as a contract that cannot drift",
        "deduplication is upstream of tokenization because every duplicate token costs the trainer twice in compute",
        "checkpoints record the optimizer state and the random seed so that a restart resumes exactly where it stopped",
    ]
    long_repeat = " ".join(base * 4)
    shards: dict[str, list[str]] = {
        "shard-0000": [long_repeat, long_repeat, long_repeat],
        "shard-0001": [long_repeat, long_repeat, long_repeat],
    }
    return shards


def run_demo() -> int:
    """构建演示语料库，对其分词、验证，并运行 dataloader。

    设计为自行终止：管线写入临时目录，dataloader 只获取固定的少量批次，
    因此脚本无需外部输入即可退出。
    """

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        tokenizer = Tokenizer()
        pipeline = ShardedTokenizationPipeline(tokenizer, output_dir=out, chunk_size=512)
        shards = build_demo_corpus()
        entries = pipeline.write_corpus(shards)
        for entry in entries:
            print(
                f"[shard] {entry.shard_id} tokens={entry.token_count} "
                f"sha256={entry.sha256[:12]} global_start={entry.global_start}"
            )
        validation_failures = validate_corpus(entries)
        if validation_failures:
            print(f"[验证] 失败：{validation_failures}")
            return 1
        print(f"[验证] 全部 {len(entries)} 个分片均与记录的 sha256 一致")
        with MmapTokenStore(entries) as store:
            loader = SlidingWindowDataloader(store, window_size=64, batch_size=4, seed=7)
            for batch_index, (inputs, targets) in enumerate(loader.take(10)):
                checksum = int(hashlib.blake2b(inputs.tobytes(), digest_size=4).hexdigest(), 16)
                print(
                    f"[batch] step={batch_index} shape={tuple(inputs.shape)} "
                    f"checksum={checksum:08x}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(run_demo())
