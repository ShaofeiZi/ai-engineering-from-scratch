import math
import random
import re


PHONE_REGEX = r"^\d{3}-\d{3}-\d{4}$"


class PhoneFSM:
    def __init__(self):
        self.accept_state = 12

    def valid_next(self, state):
        if state in (0, 1, 2, 4, 5, 6, 8, 9, 10, 11):
            return list("0123456789")
        if state in (3, 7):
            return ["-"]
        if state == 12:
            return []
        raise ValueError(f"未知状态 {state}")

    def transition(self, state, ch):
        if ch not in self.valid_next(state):
            return None
        return state + 1

    def is_accept(self, state):
        return state == self.accept_state


def softmax(xs):
    finite = [x for x in xs if x != float("-inf")]
    if not finite:
        return [0.0] * len(xs)
    m = max(finite)
    exps = [math.exp(x - m) if x != float("-inf") else 0.0 for x in xs]
    z = sum(exps)
    return [e / z for e in exps]


def sample(probs, rng):
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r <= acc:
            return i
    return len(probs) - 1


def mask_logits(logits, valid_indices):
    return [logits[i] if i in valid_indices else float("-inf") for i in range(len(logits))]


def fake_llm_logits(alphabet, rng):
    return [rng.gauss(0.0, 1.5) for _ in alphabet]


def generate_constrained(alphabet, fsm, seed):
    rng = random.Random(seed)
    alphabet_idx = {ch: i for i, ch in enumerate(alphabet)}
    state = 0
    out = ""
    while not fsm.is_accept(state):
        logits = fake_llm_logits(alphabet, rng)
        valid_chars = fsm.valid_next(state)
        if not valid_chars:
            break
        valid_ids = {alphabet_idx[ch] for ch in valid_chars}
        masked = mask_logits(logits, valid_ids)
        probs = softmax(masked)
        pick = sample(probs, rng)
        ch = alphabet[pick]
        out += ch
        state = fsm.transition(state, ch)
        if state is None:
            break
    return out


def generate_unconstrained(alphabet, max_len, seed):
    rng = random.Random(seed)
    out = ""
    for _ in range(max_len):
        logits = fake_llm_logits(alphabet, rng)
        probs = softmax(logits)
        pick = sample(probs, rng)
        out += alphabet[pick]
    return out


def main():
    alphabet = list("0123456789-")
    fsm = PhoneFSM()

    print("=== 电话号码生成：20 个样本 ===")
    print(f"目标模式：{PHONE_REGEX}")
    print()

    print("无约束（随机 logit，不应用掩码）：")
    unc_valid = 0
    for seed in range(20):
        s = generate_unconstrained(alphabet, max_len=12, seed=seed)
        ok = bool(re.fullmatch(PHONE_REGEX, s))
        unc_valid += int(ok)
        tag = "  OK" if ok else "FAIL"
        print(f"  [{tag}] {s}")
    print(f"  => 有效：{unc_valid} / 20")

    print()
    print("有约束（用 FSM 遮蔽 logits）：")
    con_valid = 0
    for seed in range(20):
        s = generate_constrained(alphabet, fsm, seed=seed)
        ok = bool(re.fullmatch(PHONE_REGEX, s))
        con_valid += int(ok)
        tag = "  OK" if ok else "FAIL"
        print(f"  [{tag}] {s}")
    print(f"  => 有效：{con_valid} / 20")

    print()
    print("注意：简化 LLM 会生成均匀随机的 logits。")
    print("唯一的区别是每一步都遮蔽无效 token。")
    print("真正的约束解码会在十万余 token 的词表上使用相同掩码。")


if __name__ == "__main__":
    main()
