"""模拟实验：以非零状态码退出且不写入任何指标。

供运行器测试用来校验崩溃的终端标签。
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    print(json.dumps({"step": 0, "note": "即将失败"}))
    print("trace: simulated failure", file=sys.stderr)
    return 3


if __name__ == "__main__":
    sys.exit(main())
