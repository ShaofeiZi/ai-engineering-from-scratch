# 在 ../docs/en.md. 中可运行的课程演示
# 少射和CoT:Wei等人,2022年,https://arxiv.org/abs/2201.11903
# 自我一致:王等,2023年,https://arxiv.org/abs/2203.11171
# 默认路径是决定性的, 不执行网络请求 。
# 直播聊天-完成访问是明确的,只使用Python的stdlib.

import argparse
import json
import math
import os
import sys
from http.client import IncompleteRead
from urllib import error, parse, request

from advanced_prompting import (
    build_cot_prompt,
    few_shot_cot_solve,
    select_examples,
    vote_reasoning_paths,
)


DEMO_EXAMPLES = [
    {
        "question": (
            "A fruit stand has 36 oranges. It sells one third in the morning "
            "and one quarter of the remainder later. How many oranges remain?"
        ),
        "reasoning": (
            "One third of 36 is 12, leaving 24. One quarter of 24 is 6, "
            "so 24 - 6 = 18 oranges remain."
        ),
        "answer": "18",
    },
    {
        "question": (
            "A writer drafts 4 pages on each of 3 days. How many pages are drafted?"
        ),
        "reasoning": "There are 3 groups of 4 pages, so 3 * 4 = 12 pages.",
        "answer": "12",
    },
    {
        "question": (
            "A ticket costs $8 and a snack costs $3. What is the total cost?"
        ),
        "reasoning": "Add the two prices: 8 + 3 = 11 dollars.",
        "answer": "11",
    },
]

DEMO_QUESTION = (
    "A grocer has 48 apples. The grocer sells one third in the morning and "
    "one quarter of the remainder in the afternoon. How many apples remain?"
)

DEMO_REASONING_PATHS = [
    "One third of 48 is 16, leaving 32. One quarter of 32 is 8. "
    "Then 32 - 8 = 24. The answer is 24.",
    "After the first sale, 48 * 2/3 = 32. Keeping three quarters gives "
    "32 * 3/4 = 24. The answer is 24.",
    "The sold fractions are 1/3 + 1/4 = 7/12, so 48 * 5/12 = 20. "
    "The answer is 20.",
    "Sell 16 first and 8 next: 48 - 16 - 8 = 24. The answer is 24.",
    "The remaining fraction is (2/3) * (3/4) = 1/2; 48 / 2 = 24. "
    "The answer is 24.",
]


def positive_timeout(value):
    """分析一个大于零的限时值 。"""
    try:
        timeout = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "timeout must be a positive finite number"
        ) from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise argparse.ArgumentTypeError(
            "timeout must be a positive finite number"
        )
    return timeout


class _RejectRedirectHandler(request.HTTPRedirectHandler):
    """在urlib将请求头复制到新的 URL 之前停止重定向 。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise error.HTTPError(
            req.full_url, code, "redirect responses are not allowed", headers, fp
        )


class OpenAICompatibleHTTPClient:
    """从stdlib 创建的最小选择聊天- 补全客户端 。"""

    def __init__(self, api_key, base_url, timeout=30.0):
        try:
            parsed_base_url = parse.urlsplit(base_url)
            hostname = parsed_base_url.hostname
            parsed_base_url.port  # 访问可验证文本和数字端口值。
        except (TypeError, ValueError) as exc:
            raise ValueError("base_url must be an absolute HTTPS URL") from exc
        if parsed_base_url.scheme.lower() != "https" or not hostname:
            raise ValueError("base_url must be an absolute HTTPS URL")
        self.api_key = api_key
        self.endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self.timeout = timeout
        self._opener = request.build_opener(_RejectRedirectHandler())

    def complete(self, model, system, user, temperature, max_tokens):
        payload = json.dumps(
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
        http_request = request.Request(
            self.endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )
        # urlib的股票将处理器复制普通头到下一个URL 。
        # 即使拒绝重定向,也要保留证书不可转发。
        http_request.add_unredirected_header(
            "Authorization", f"Bearer {self.api_key}"
        )
        try:
            response = self._opener.open(http_request, timeout=self.timeout)
        except error.HTTPError as exc:
            if 300 <= exc.code < 400:
                raise RuntimeError(
                    f"provider redirect rejected (HTTP {exc.code})"
                ) from exc
            raise RuntimeError(f"provider returned HTTP {exc.code}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"provider request failed: {exc.reason}") from exc
        except OSError as exc:
            raise RuntimeError(f"provider request failed: {exc}") from exc

        try:
            with response:
                body = json.load(response)
        except (OSError, IncompleteRead) as exc:
            raise RuntimeError(f"provider response read failed: {exc}") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError("provider returned invalid JSON") from exc

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("provider response has no assistant content") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("provider response has no assistant content")
        return content


def build_offline_demo():
    selected = select_examples(DEMO_QUESTION, DEMO_EXAMPLES, num_examples=2)
    system, user = build_cot_prompt(DEMO_QUESTION, selected, num_examples=2)
    answer, confidence, votes = vote_reasoning_paths(DEMO_REASONING_PATHS)
    return {
        "question": DEMO_QUESTION,
        "selected_questions": [example["question"] for example in selected],
        "system": system,
        "user": user,
        "answer": answer,
        "confidence": confidence,
        "votes": dict(votes),
    }


def run_offline_demo(stream=None):
    stream = sys.stdout if stream is None else stream
    result = build_offline_demo()
    print("Few-shot + 思维链：离线演示", file=stream)
    print(f"问题：{result['question']}", file=stream)
    print("选定的演示:", file=stream)
    for question in result["selected_questions"]:
        print(f"  - {question}", file=stream)
    print("提示契约：先推理，再输出“The answer is <number>.”", file=stream)
    print(f"自洽性投票：{result['votes']}", file=stream)
    print(
        f"获胜答案：{result['answer']} "
        f"（置信度 {result['confidence']:.0%}）",
        file=stream,
    )
    print("没有提出网络要求。 使用 -- 在线选择加入。", file=stream)
    return result


def run_online_demo(args, environ, stream=None, error_stream=None):
    stream = sys.stdout if stream is None else stream
    error_stream = sys.stderr if error_stream is None else error_stream
    api_key = environ.get("OPENAI_API_KEY")
    model = args.model or environ.get("OPENAI_MODEL") or environ.get("LLM_MODEL")
    if not api_key or not model:
        print(
            "在线模式需要 OPENAI_API_KEY 和 --model（或 OPENAI_MODEL）。",
            file=error_stream,
        )
        return 2

    base_url = environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    try:
        client = OpenAICompatibleHTTPClient(api_key, base_url, timeout=args.timeout)
    except ValueError as exc:
        print(f"在线配置无效：{exc}", file=error_stream)
        return 2
    selected = select_examples(DEMO_QUESTION, DEMO_EXAMPLES, num_examples=2)
    try:
        answer, reasoning = few_shot_cot_solve(
            DEMO_QUESTION, selected, client, model, num_examples=2
        )
    except RuntimeError as exc:
        print(f"在线请求失败：{exc}", file=error_stream)
        return 1
    if answer is None:
        print(
            "在线回复未包含 parseable numeric answer。",
            file=error_stream,
        )
        return 1
    print(f"模型：{model}", file=stream)
    print(reasoning, file=stream)
    print(f"解析出的答案：{answer}", file=stream)
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Demonstrate few-shot CoT and self-consistency offline."
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="explicitly call an OpenAI-compatible chat-completions endpoint",
    )
    parser.add_argument("--model", help="provider model ID for --online")
    parser.add_argument(
        "--timeout",
        type=positive_timeout,
        default=30.0,
        help="positive online request timeout in seconds",
    )
    return parser.parse_args(argv)


def main(argv=None, environ=None):
    args = parse_args(argv)
    if not args.online:
        run_offline_demo()
        return 0
    return run_online_demo(args, os.environ if environ is None else environ)


if __name__ == "__main__":
    raise SystemExit(main())
