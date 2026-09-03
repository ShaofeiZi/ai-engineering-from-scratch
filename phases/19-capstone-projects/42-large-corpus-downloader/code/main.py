"""支持续传、MinHash + LSH 去重和分片清单的流式语料下载器。

从 URL 列表拉取压缩分片，以流式方式通过 Zstandard 解压器，迭代 JSONL 文档，
使用 MinHash 为每篇文档生成指纹，通过局部敏感哈希为签名分桶，丢弃近似重复项，
并写入语料库清单。

文件末尾的演示会在磁盘上构建小型合成语料库，用 Zstandard 压缩，通过文件 URL
公开，再经本模块下载并打印清单。运行：python3 code/main.py
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import struct
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

try:
    import zstandard as zstd
except ImportError as exc:
    raise SystemExit(
        "本课需要 zstandard。安装命令：pip install zstandard"
    ) from exc


CHUNK_BYTES = 1 << 16
DEFAULT_NUM_HASHES = 128
DEFAULT_BANDS = 32
DEFAULT_SHINGLE_WIDTH = 5
MAX_UINT64 = (1 << 64) - 1
MERSENNE_PRIME = (1 << 61) - 1


@dataclass
class ShardPlan:
    """计划分片列表中的一行。"""

    shard_id: str
    url: str
    expected_size: int | None = None


@dataclass
class ShardResult:
    """每个分片的下载与去重结果。"""

    shard_id: str
    url: str
    raw_bytes: int
    decompressed_bytes: int
    document_count: int
    kept_count: int
    duplicate_count: int
    sha256: str

    def to_manifest_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class DocVerdict:
    """单篇文档的去重判定。"""

    shard_id: str
    doc_index: int
    verdict: str  # "keep" or "near_duplicate"
    collided_with: str | None = None  # 保留项的 "shard:doc"。


@dataclass
class CheckpointState:
    """持久化在分片旁的续传检查点。"""

    url: str
    verified_bytes: int
    expected_size: int | None
    sha256_prefix_hex: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "CheckpointState":
        data = json.loads(text)
        return cls(
            url=str(data["url"]),
            verified_bytes=int(data["verified_bytes"]),
            expected_size=(int(data["expected_size"]) if data.get("expected_size") is not None else None),
            sha256_prefix_hex=str(data["sha256_prefix_hex"]),
        )


def _hash_seed_pair(seed: int) -> tuple[int, int]:
    """从种子派生两个 64 位系数 (a, b)。

    签名使用形式为 ((a * x + b) mod p) mod 2^64 的通用哈希。
    两个系数由种子确定性派生，使哈希函数族可跨运行、跨机器复现。
    """

    digest = hashlib.blake2b(seed.to_bytes(8, "little"), digest_size=16).digest()
    a = int.from_bytes(digest[:8], "little") | 1  # 确保 a 非零。
    b = int.from_bytes(digest[8:], "little")
    return a, b


class MinHasher:
    """使用固定哈希种子族的 MinHash 签名构建器。"""

    def __init__(self, num_hashes: int = DEFAULT_NUM_HASHES, shingle_width: int = DEFAULT_SHINGLE_WIDTH) -> None:
        if num_hashes <= 0:
            raise ValueError("num_hashes must be positive")
        if shingle_width <= 0:
            raise ValueError("shingle_width must be positive")
        self.num_hashes = num_hashes
        self.shingle_width = shingle_width
        self._coefficients: list[tuple[int, int]] = [_hash_seed_pair(i) for i in range(num_hashes)]

    def shingles(self, text: str) -> list[str]:
        """返回相互重叠的空白分词 shingle。"""

        tokens = text.split()
        if len(tokens) < self.shingle_width:
            return [" ".join(tokens)] if tokens else []
        shingles: list[str] = []
        for start in range(len(tokens) - self.shingle_width + 1):
            shingles.append(" ".join(tokens[start : start + self.shingle_width]))
        return shingles

    @staticmethod
    def _hash_shingle(shingle: str) -> int:
        digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little")

    def signature(self, text: str) -> list[int]:
        """以 num_hashes 个 64 位整数的列表形式返回 MinHash 签名。"""

        shingles = self.shingles(text)
        if not shingles:
            return [MAX_UINT64] * self.num_hashes
        shingle_hashes = [self._hash_shingle(s) for s in shingles]
        sig: list[int] = []
        for a, b in self._coefficients:
            best = MAX_UINT64
            for h in shingle_hashes:
                candidate = ((a * h + b) % MERSENNE_PRIME) & MAX_UINT64
                if candidate < best:
                    best = candidate
            sig.append(best)
        return sig


class LSHIndex:
    """基于 MinHash 签名的局部敏感哈希索引。

    将每个签名拆为 ``bands`` 个 band，每个 band 有
    ``rows = num_hashes / bands`` 行。两个签名只要在至少一个 band 上一致就会
    碰撞。碰撞概率为 ``1 - (1 - s^r)^b``，其中 s 是 Jaccard 相似度，因此在
    ``s = (1/b)^(1/r)`` 附近形成明显阈值。``(b=32, r=4)`` 时阈值约为
    ``s = 0.42``；``(b=20, r=5)`` 时约为 ``s = 0.55``。
    """

    def __init__(self, num_hashes: int, bands: int = DEFAULT_BANDS) -> None:
        if bands <= 0 or num_hashes % bands != 0:
            raise ValueError(f"bands ({bands}) must divide num_hashes ({num_hashes})")
        self.num_hashes = num_hashes
        self.bands = bands
        self.rows = num_hashes // bands
        self._buckets: list[dict[bytes, list[str]]] = [{} for _ in range(bands)]
        self._signatures: dict[str, list[int]] = {}

    @staticmethod
    def _band_key(band: list[int]) -> bytes:
        return hashlib.blake2b(b"".join(struct.pack("<Q", v) for v in band), digest_size=16).digest()

    def query(self, signature: list[int]) -> str | None:
        """返回近重复保留文档的 doc id；若无则返回 None。"""

        for i in range(self.bands):
            band = signature[i * self.rows : (i + 1) * self.rows]
            key = self._band_key(band)
            bucket = self._buckets[i].get(key)
            if bucket:
                return bucket[0]
        return None

    def insert(self, doc_id: str, signature: list[int]) -> None:
        self._signatures[doc_id] = signature
        for i in range(self.bands):
            band = signature[i * self.rows : (i + 1) * self.rows]
            key = self._band_key(band)
            self._buckets[i].setdefault(key, []).append(doc_id)

    def jaccard_estimate(self, doc_a: str, doc_b: str) -> float:
        """返回两个已索引文档之间无偏的 Jaccard 估计值。"""

        sig_a = self._signatures[doc_a]
        sig_b = self._signatures[doc_b]
        agree = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
        return agree / self.num_hashes


class Dedup:
    """把 MinHasher 与 LSHIndex 组合为流式去重器。"""

    def __init__(self, hasher: MinHasher, index: LSHIndex) -> None:
        self.hasher = hasher
        self.index = index

    def evaluate(self, shard_id: str, doc_index: int, text: str) -> DocVerdict:
        sig = self.hasher.signature(text)
        keeper = self.index.query(sig)
        if keeper is not None:
            return DocVerdict(
                shard_id=shard_id,
                doc_index=doc_index,
                verdict="near_duplicate",
                collided_with=keeper,
            )
        doc_id = f"{shard_id}:{doc_index}"
        self.index.insert(doc_id, sig)
        return DocVerdict(shard_id=shard_id, doc_index=doc_index, verdict="keep")


class ZstdDocIterator:
    """从 Zstandard 压缩字节流中迭代 JSONL 文档。

    用 Zstandard 流读取器包装上游读取器，再逐行迭代文档。解压器不会缓冲整个分片，
    而是增量消费上游数据。
    """

    def __init__(self, raw_reader: io.RawIOBase | io.BufferedIOBase) -> None:
        self._dctx = zstd.ZstdDecompressor()
        self._stream = self._dctx.stream_reader(raw_reader)
        self._text = io.TextIOWrapper(self._stream, encoding="utf-8", newline="")

    def __iter__(self) -> Iterator[str]:
        for line in self._text:
            line = line.rstrip("\n")
            if line:
                yield line


class StreamingDownloader:
    """使用 Range 续传与检查点，把远程 URL 流式写入本地路径。

    每个数据块都会推进已验证哈希与字节计数，并以原子方式重写检查点。检查点记录
    已验证字节的 sha256 前缀，因此损坏的部分文件无法被静默续传。
    """

    def __init__(
        self,
        cache_dir: Path,
        opener: Callable[[urllib.request.Request], object] | None = None,
        chunk_bytes: int = CHUNK_BYTES,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_bytes = chunk_bytes
        self._opener = opener or urllib.request.urlopen

    def _paths_for(self, shard_id: str) -> tuple[Path, Path]:
        shard_path = self.cache_dir / f"{shard_id}.zst"
        checkpoint_path = self.cache_dir / f"{shard_id}.partial.json"
        return shard_path, checkpoint_path

    def _read_checkpoint(self, checkpoint_path: Path) -> CheckpointState | None:
        if not checkpoint_path.exists():
            return None
        try:
            return CheckpointState.from_json(checkpoint_path.read_text("utf-8"))
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def _write_checkpoint(self, checkpoint_path: Path, state: CheckpointState) -> None:
        tmp = checkpoint_path.with_suffix(".json.tmp")
        tmp.write_text(state.to_json(), encoding="utf-8")
        os.replace(tmp, checkpoint_path)

    def _verify_partial(self, shard_path: Path, state: CheckpointState) -> bool:
        if not shard_path.exists():
            return False
        actual_size = shard_path.stat().st_size
        if actual_size != state.verified_bytes:
            return False
        hasher = hashlib.sha256()
        with shard_path.open("rb") as fh:
            remaining = state.verified_bytes
            while remaining > 0:
                buf = fh.read(min(self.chunk_bytes, remaining))
                if not buf:
                    return False
                hasher.update(buf)
                remaining -= len(buf)
        return hasher.hexdigest() == state.sha256_prefix_hex

    def download(self, plan: ShardPlan) -> ShardResult:
        parsed = urllib.parse.urlparse(plan.url)
        if parsed.scheme not in {"http", "https", "file"}:
            raise ValueError(
                f"unsupported URL scheme {parsed.scheme!r} for shard {plan.shard_id}"
            )
        shard_path, checkpoint_path = self._paths_for(plan.shard_id)
        state = self._read_checkpoint(checkpoint_path)
        resume_from = 0
        rolling = hashlib.sha256()
        if state is not None and state.url == plan.url and self._verify_partial(shard_path, state):
            resume_from = state.verified_bytes
            with shard_path.open("rb") as fh:
                remaining = resume_from
                while remaining > 0:
                    buf = fh.read(min(self.chunk_bytes, remaining))
                    if not buf:
                        break
                    rolling.update(buf)
                    remaining -= len(buf)
        else:
            if shard_path.exists():
                shard_path.unlink()
            if checkpoint_path.exists():
                checkpoint_path.unlink()

        request = urllib.request.Request(plan.url)
        if resume_from > 0:
            request.add_header("Range", f"bytes={resume_from}-")
        response = self._opener(request)
        try:
            if resume_from > 0:
                status = int(getattr(response, "status", 0) or 0)
                headers = getattr(response, "headers", None)
                content_range = ""
                if headers is not None:
                    content_range = str(headers.get("Content-Range", "") or "")
                if status != 206 or not content_range.startswith(f"bytes {resume_from}-"):
            # 服务器忽略或错误报告了 Range 标头。
            # 在修改分片或读取响应体之前，关闭部分响应并重新发起完整 GET。
                    try:
                        response.close()
                    except Exception:
                        pass
                    resume_from = 0
                    rolling = hashlib.sha256()
                    if shard_path.exists():
                        shard_path.unlink()
                    if checkpoint_path.exists():
                        checkpoint_path.unlink()
                    response = self._opener(urllib.request.Request(plan.url))
            mode = "ab" if resume_from > 0 else "wb"
            with shard_path.open(mode) as out:
                while True:
                    buf = response.read(self.chunk_bytes)
                    if not buf:
                        break
                    rolling.update(buf)
                    next_verified = resume_from + len(buf)
                    new_state = CheckpointState(
                        url=plan.url,
                        verified_bytes=next_verified,
                        expected_size=plan.expected_size,
                        sha256_prefix_hex=rolling.hexdigest(),
                    )
                    self._write_checkpoint(checkpoint_path, new_state)
                    out.write(buf)
                    out.flush()
                    resume_from = next_verified
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

        decompressed_bytes = 0
        document_count = 0
        with shard_path.open("rb") as fh:
            dctx = zstd.ZstdDecompressor()
            reader = dctx.stream_reader(fh)
            while True:
                chunk = reader.read(self.chunk_bytes)
                if not chunk:
                    break
                decompressed_bytes += len(chunk)
                document_count += chunk.count(b"\n")

        return ShardResult(
            shard_id=plan.shard_id,
            url=plan.url,
            raw_bytes=resume_from,
            decompressed_bytes=decompressed_bytes,
            document_count=document_count,
            kept_count=0,
            duplicate_count=0,
            sha256=rolling.hexdigest(),
        )


class ShardPlanner:
    """把 URL 列表转换为计划分片列表。"""

    @staticmethod
    def from_urls(urls: Iterable[str]) -> list[ShardPlan]:
        plans: list[ShardPlan] = []
        for index, url in enumerate(urls):
            shard_id = f"shard-{index:04d}"
            plans.append(ShardPlan(shard_id=shard_id, url=url))
        return plans


class ManifestWriter:
    """把分片结果收集到带自身内容哈希的清单中。"""

    def __init__(self) -> None:
        self._rows: list[dict[str, object]] = []
        self._verdicts: list[dict[str, object]] = []

    def add_shard(self, result: ShardResult) -> None:
        self._rows.append(result.to_manifest_row())

    def add_verdict(self, verdict: DocVerdict) -> None:
        self._verdicts.append(asdict(verdict))

    def write(self, manifest_path: Path) -> str:
        body = {
            "version": 1,
            "generated_at": int(time.time()),
            "shards": self._rows,
            "verdicts": self._verdicts,
        }
        text = json.dumps(body, sort_keys=True, indent=2)
        manifest_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        manifest_path.write_text(text, encoding="utf-8")
        lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")
        lock_path.write_text(json.dumps({"manifest_sha256": manifest_sha}), encoding="utf-8")
        return manifest_sha

    @property
    def shards(self) -> list[dict[str, object]]:
        return list(self._rows)

    @property
    def verdicts(self) -> list[dict[str, object]]:
        return list(self._verdicts)


def process_shard(
    plan: ShardPlan,
    downloader: StreamingDownloader,
    dedup: Dedup,
    manifest: ManifestWriter,
) -> ShardResult:
    """下载、解压、去重并统计单个分片。"""

    result = downloader.download(plan)
    kept = 0
    duplicates = 0
    shard_path = downloader.cache_dir / f"{plan.shard_id}.zst"
    with shard_path.open("rb") as fh:
        for doc_index, line in enumerate(ZstdDocIterator(fh)):
            verdict = dedup.evaluate(plan.shard_id, doc_index, line)
            manifest.add_verdict(verdict)
            if verdict.verdict == "keep":
                kept += 1
            else:
                duplicates += 1
    result = dataclasses.replace(result, kept_count=kept, duplicate_count=duplicates)
    manifest.add_shard(result)
    return result


def build_demo_corpus(directory: Path) -> list[str]:
    """构建包含重复项的微型合成语料库，并写入 zst 分片。

    返回下载器应拉取的文件 URL 列表。
    """

    directory.mkdir(parents=True, exist_ok=True)
    base = [
        "the alignment problem is a story about reward functions and what we miss when we write them",
        "the alignment problem is a story about reward functions and the things we forget to write down",
        "transformers replaced recurrent networks because attention scales better with sequence length",
        "attention scales better with sequence length so transformers replaced recurrent networks",
        "evaluation harnesses keep training honest by treating the test corpus as a contract",
        "a contract between training and evaluation is what an eval harness ultimately enforces",
        "deduplication is upstream of tokenization so duplicates do not pay tokenization cost twice",
        "the tokenizer is a vocabulary contract between the model and the corpus",
        "checkpointing the verified bytes before writing the buffer is the only safe resume order",
        "the manifest is the deciding edge between data is downloaded and data is verifiable",
    ]
    shards = [base[:5], base[3:9], base[6:]]
    urls: list[str] = []
    for i, group in enumerate(shards):
        payload = ("\n".join(group) + "\n").encode("utf-8")
        compressed = zstd.ZstdCompressor(level=10).compress(payload)
        path = directory / f"corpus-{i:02d}.zst"
        path.write_bytes(compressed)
        urls.append(path.as_uri())
    return urls


def run_demo() -> int:
    with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as cache_dir:
        corpus_dir = Path(raw_dir)
        cache_path = Path(cache_dir)
        urls = build_demo_corpus(corpus_dir)
        plans = ShardPlanner.from_urls(urls)
        downloader = StreamingDownloader(cache_dir=cache_path)
        hasher = MinHasher(num_hashes=128, shingle_width=3)
        index = LSHIndex(num_hashes=128, bands=32)
        dedup = Dedup(hasher=hasher, index=index)
        manifest = ManifestWriter()
        for plan in plans:
            result = process_shard(plan, downloader, dedup, manifest)
            print(
                f"[分片] {result.shard_id} 文档={result.document_count} "
                f"保留={result.kept_count} 重复={result.duplicate_count} "
                f"sha256={result.sha256[:12]}"
            )
        manifest_path = cache_path / "manifest.json"
        manifest_sha = manifest.write(manifest_path)
        kept = sum(int(row["kept_count"]) for row in manifest.shards)
        dup = sum(int(row["duplicate_count"]) for row in manifest.shards)
        print(f"[清单] sha256={manifest_sha[:12]} 保留={kept} 重复={dup}")
    return 0


if __name__ == "__main__":
    sys.exit(run_demo())
