"""
智能体运行框架的可观测性：GenAI span 与 Prometheus 指标。

See: phases/19-capstone-projects/28-observability-otel-traces/docs/en.md
概念参考：
  - OpenTelemetry GenAI 语义约定（gen_ai.* 属性键）。
  - Prometheus 文本展示格式（计数器和直方图）。
  - W3C Trace Context（16 字节 trace_id、8 字节 span_id）。
文件末尾的演示会把 span 发送到临时 jsonl 文件，打印 Prometheus 展示文本，
并以状态码 0 退出。
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# OTel 语义约定键
# ---------------------------------------------------------------------------

# 标准 GenAI 属性（OpenTelemetry GenAI 语义约定）。
# 这些键保持稳定；只会添加新键，不会重命名已有键。
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_RESPONSE_ID = "gen_ai.response.id"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
GEN_AI_TOOL_RESULT_BYTES = "gen_ai.tool.result.bytes"

# 运行框架专用属性（使用不会冲突的前缀）。
HARNESS_GATE_DECISION = "agent.harness.gate.decision"
HARNESS_GATE_REASON = "agent.harness.gate.reason"

STATUS_UNSET = "UNSET"
STATUS_OK = "OK"
STATUS_ERROR = "ERROR"


# ---------------------------------------------------------------------------
# Span 数据结构
# ---------------------------------------------------------------------------


@dataclass
class SpanEvent:
    """记录在 span 内的离散事件（遵循 OTel）。"""

    name: str
    timestamp_unix_nano: int
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "timestamp_unix_nano": self.timestamp_unix_nano,
            "attributes": dict(self.attributes),
        }


@dataclass
class GenAISpan:
    """符合 OTel GenAI 约定的单个 span。"""

    trace_id: str
    span_id: str
    name: str
    start_unix_nano: int
    end_unix_nano: int = 0
    parent_span_id: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    status: str = STATUS_UNSET
    status_message: str = ""
    kind: str = "INTERNAL"

    @property
    def duration_ms(self) -> float:
        if self.end_unix_nano <= 0:
            return 0.0
        return (self.end_unix_nano - self.start_unix_nano) / 1_000_000.0

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "kind": self.kind,
            "start_unix_nano": self.start_unix_nano,
            "end_unix_nano": self.end_unix_nano,
            "duration_ms": round(self.duration_ms, 4),
            "attributes": dict(self.attributes),
            "events": [e.to_dict() for e in self.events],
            "status": {"code": self.status, "message": self.status_message},
        }


# ---------------------------------------------------------------------------
# ID 生成
# ---------------------------------------------------------------------------


def new_trace_id() -> str:
    """随机 16 字节十六进制字符串，符合 W3C Trace Context。"""

    return uuid.uuid4().hex + uuid.uuid4().hex[:0]  # uuid4 的十六进制形式已经是 32 个字符


def new_span_id() -> str:
    """随机 8 字节十六进制字符串，符合 W3C Trace Context span ID。"""

    return uuid.uuid4().hex[:16]


def now_unix_nano() -> int:
    return time.time_ns()


# ---------------------------------------------------------------------------
# 导出器
# ---------------------------------------------------------------------------


@dataclass
class JSONLExporter:
    """只追加的导出器：每行 JSON 记录一个 span。"""

    path: str
    fh: Any = None
    closed: bool = False

    def _ensure_open(self) -> None:
        if self.fh is None:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            self.fh = open(self.path, "a", encoding="utf-8")

    def export(self, span: GenAISpan) -> None:
        self._ensure_open()
        line = json.dumps(span.to_dict(), separators=(",", ":"))
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self) -> None:
        if self.fh is not None and not self.closed:
            self.fh.close()
            self.closed = True


class InMemoryExporter:
    """便于测试的导出器，把 span 保存在列表中。"""

    def __init__(self) -> None:
        self.spans: list[GenAISpan] = []

    def export(self, span: GenAISpan) -> None:
        self.spans.append(span)

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# 指标基本组件
# ---------------------------------------------------------------------------


def _label_key(labels: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(labels.items()))


def _format_labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    parts = []
    for k, v in sorted(labels.items()):
        escaped = str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        parts.append(f'{k}="{escaped}"')
    return "{" + ",".join(parts) + "}"


@dataclass
class Counter:
    """简单的带标签计数器。"""

    name: str
    help: str = ""
    values: dict[tuple[tuple[str, str], ...], float] = field(default_factory=dict)

    def inc(self, labels: dict[str, str] | None = None, by: float = 1.0) -> None:
        key = _label_key(labels or {})
        self.values[key] = self.values.get(key, 0.0) + by

    def get(self, labels: dict[str, str] | None = None) -> float:
        return self.values.get(_label_key(labels or {}), 0.0)


@dataclass
class Histogram:
    """使用显式桶的直方图，与 OTel 默认毫秒桶集合一致。"""

    name: str
    help: str = ""
    buckets: tuple[float, ...] = (
        5.0,
        10.0,
        25.0,
        50.0,
        100.0,
        250.0,
        500.0,
        1000.0,
        2500.0,
        5000.0,
        10000.0,
    )
    samples: dict[tuple[tuple[str, str], ...], list[float]] = field(
        default_factory=dict
    )

    def observe(self, value: float, labels: dict[str, str] | None = None) -> None:
        key = _label_key(labels or {})
        self.samples.setdefault(key, []).append(float(value))

    def bucket_counts(
        self, labels: dict[str, str] | None = None
    ) -> dict[float, int]:
        key = _label_key(labels or {})
        values = self.samples.get(key, [])
        counts: dict[float, int] = {}
        for bound in self.buckets:
            counts[bound] = sum(1 for v in values if v <= bound)
        counts[math.inf] = len(values)
        return counts

    def total_count(self, labels: dict[str, str] | None = None) -> int:
        return len(self.samples.get(_label_key(labels or {}), []))

    def total_sum(self, labels: dict[str, str] | None = None) -> float:
        return float(sum(self.samples.get(_label_key(labels or {}), [])))


@dataclass
class MetricsRegistry:
    counters: dict[str, Counter] = field(default_factory=dict)
    histograms: dict[str, Histogram] = field(default_factory=dict)

    def counter(self, name: str, help: str = "") -> Counter:
        if name not in self.counters:
            self.counters[name] = Counter(name=name, help=help)
        return self.counters[name]

    def histogram(self, name: str, help: str = "") -> Histogram:
        if name not in self.histograms:
            self.histograms[name] = Histogram(name=name, help=help)
        return self.histograms[name]


def prometheus_exposition(registry: MetricsRegistry) -> str:
    """把注册表渲染为 Prometheus 文本展示格式。"""

    lines: list[str] = []
    for name in sorted(registry.counters):
        counter = registry.counters[name]
        if counter.help:
            lines.append(f"# HELP {counter.name} {counter.help}")
        lines.append(f"# TYPE {counter.name} counter")
        if not counter.values:
            lines.append(f"{counter.name} 0")
        for key, value in sorted(counter.values.items()):
            label_str = _format_labels(dict(key))
            lines.append(f"{counter.name}{label_str} {value}")
    for name in sorted(registry.histograms):
        hist = registry.histograms[name]
        if hist.help:
            lines.append(f"# HELP {hist.name} {hist.help}")
        lines.append(f"# TYPE {hist.name} histogram")
        keys = list(hist.samples.keys()) or [tuple()]
        for key in sorted(keys):
            label_dict = dict(key)
            counts = hist.bucket_counts(label_dict)
            for bound in hist.buckets:
                bucket_labels = {**label_dict, "le": _format_le(bound)}
                lines.append(
                    f"{hist.name}_bucket{_format_labels(bucket_labels)} {counts[bound]}"
                )
            inf_labels = {**label_dict, "le": "+Inf"}
            lines.append(
                f"{hist.name}_bucket{_format_labels(inf_labels)} {counts[math.inf]}"
            )
            lines.append(
                f"{hist.name}_sum{_format_labels(label_dict)} "
                f"{hist.total_sum(label_dict)}"
            )
            lines.append(
                f"{hist.name}_count{_format_labels(label_dict)} "
                f"{hist.total_count(label_dict)}"
            )
    return "\n".join(lines) + "\n"


def _format_le(bound: float) -> str:
    if bound == int(bound):
        return str(int(bound))
    return repr(bound)


# ---------------------------------------------------------------------------
# Span 构建器
# ---------------------------------------------------------------------------


@dataclass
class SpanBuilder:
    """持有 trace ID，并通过一个或多个导出器发送 span。"""

    trace_id: str = field(default_factory=new_trace_id)
    exporters: list[Any] = field(default_factory=list)
    metrics: MetricsRegistry | None = None

    @contextmanager
    def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        parent: GenAISpan | None = None,
        kind: str = "INTERNAL",
    ) -> Iterator[GenAISpan]:
        span = GenAISpan(
            trace_id=self.trace_id,
            span_id=new_span_id(),
            parent_span_id=parent.span_id if parent else "",
            name=name,
            kind=kind,
            start_unix_nano=now_unix_nano(),
            attributes=dict(attributes or {}),
        )
        try:
            yield span
            span.status = STATUS_OK
        except BaseException as exc:  # noqa: BLE001
            span.status = STATUS_ERROR
            span.status_message = f"{type(exc).__name__}: {exc}"
            span.events.append(
                SpanEvent(
                    name="exception",
                    timestamp_unix_nano=now_unix_nano(),
                    attributes={
                        "exception.type": type(exc).__name__,
                        "exception.message": str(exc),
                    },
                )
            )
            raise
        finally:
            span.end_unix_nano = now_unix_nano()
            for exporter in self.exporters:
                exporter.export(span)
            if self.metrics is not None:
                tool = span.attributes.get(GEN_AI_TOOL_NAME)
                if tool is not None:
                    self.metrics.counter(
                        "tools_called_total", help="工具调用总数"
                    ).inc({"tool": str(tool)})
                    self.metrics.histogram(
                        "tool_latency_ms",
                        help="工具调用延迟（毫秒）",
                    ).observe(span.duration_ms, {"tool": str(tool)})


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------


def run_demo() -> int:
    tmp_dir = tempfile.mkdtemp(prefix="otel-demo-")
    trace_path = os.path.join(tmp_dir, "traces.jsonl")

    jsonl = JSONLExporter(path=trace_path)
    metrics = MetricsRegistry()
    in_mem = InMemoryExporter()
    builder = SpanBuilder(exporters=[jsonl, in_mem], metrics=metrics)

    print("可观测性演示")
    print(f"trace_id={builder.trace_id}")
    print(f"正在把追踪写入 {trace_path}")

    # 合成一轮智能体交互：用一个 gen_ai.chat span 包住两个工具 span。
    with builder.span(
        "gen_ai.chat",
        attributes={
            GEN_AI_SYSTEM: "anthropic",
            GEN_AI_REQUEST_MODEL: "claude-track-a",
            GEN_AI_REQUEST_MAX_TOKENS: 1024,
            GEN_AI_REQUEST_TEMPERATURE: 0.0,
            GEN_AI_OPERATION_NAME: "chat",
        },
        kind="CLIENT",
    ) as chat_span:
        time.sleep(0.01)
        chat_span.attributes[GEN_AI_USAGE_INPUT_TOKENS] = 412
        chat_span.attributes[GEN_AI_USAGE_OUTPUT_TOKENS] = 96
        chat_span.attributes[GEN_AI_RESPONSE_MODEL] = "claude-track-a-2026-05-25"
        chat_span.attributes[GEN_AI_RESPONSE_ID] = "msg_" + uuid.uuid4().hex[:8]

        with builder.span(
            "gen_ai.tool.execution",
            parent=chat_span,
            attributes={
                GEN_AI_TOOL_NAME: "read_file",
                GEN_AI_TOOL_CALL_ID: "call_001",
            },
        ) as tool1:
            time.sleep(0.005)
            tool1.attributes[GEN_AI_TOOL_RESULT_BYTES] = 1024
            tool1.events.append(
                SpanEvent(
                    name="agent.harness.gate.decision",
                    timestamp_unix_nano=now_unix_nano(),
                    attributes={
                        HARNESS_GATE_DECISION: "ALLOW",
                        HARNESS_GATE_REASON: "已通过门禁链",
                    },
                )
            )

        with builder.span(
            "gen_ai.tool.execution",
            parent=chat_span,
            attributes={
                GEN_AI_TOOL_NAME: "run_tests",
                GEN_AI_TOOL_CALL_ID: "call_002",
            },
        ) as tool2:
            time.sleep(0.003)
            tool2.attributes[GEN_AI_TOOL_RESULT_BYTES] = 256

    # 第二轮故意让门禁拒绝，以覆盖错误 span。
    try:
        with builder.span(
            "gen_ai.tool.execution",
            attributes={
                GEN_AI_TOOL_NAME: "shell",
                GEN_AI_TOOL_CALL_ID: "call_003",
            },
        ) as bad_tool:
            bad_tool.events.append(
                SpanEvent(
                    name="agent.harness.gate.decision",
                    timestamp_unix_nano=now_unix_nano(),
                    attributes={
                        HARNESS_GATE_DECISION: "DENY",
                        HARNESS_GATE_REASON: "工具不在允许集合中",
                    },
                )
            )
            raise PermissionError("工具 'shell' 不在允许集合中")
    except PermissionError:
        pass

    print("")
    print(f"已发送 {len(in_mem.spans)} 个 span：")
    for span in in_mem.spans:
        attrs_compact = {
            k: v
            for k, v in span.attributes.items()
            if k.startswith("gen_ai.")
        }
        print(
            f"  - {span.name:32s} dur={span.duration_ms:7.2f}ms "
            f"status={span.status} keys={sorted(attrs_compact)[:3]}"
        )

    print("")
    print("--- Prometheus 展示文本 ---")
    print(prometheus_exposition(metrics))

    jsonl.close()

    # 健全性检查：往返读取 jsonl 文件。
    with open(trace_path, "r", encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh if line.strip()]
    print(f"从 {trace_path} 往返解析了 {len(lines)} 个 span")

    return 0


if __name__ == "__main__":
    sys.exit(run_demo())
