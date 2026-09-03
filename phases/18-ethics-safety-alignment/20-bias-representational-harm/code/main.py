"""基于 embedding 的偏见玩具探针（WEAT 风格）——仅使用 Python 标准库。

构建简单的四维 embedding，每个轴对应一个语义维度。两个身份组为
A = {'he', 'his', 'man'} 和 B = {'she', 'her', 'woman'}；两个属性集为
X = {'engineer', 'programmer', 'scientist'} 和
Y = {'nurse', 'teacher', 'caregiver'}。

WEAT：对每个目标词计算 s(w, X, Y) = mean cosine(w, X) - mean cosine(w, Y)；
身份组间的效应为 effect = mean_a(s) - mean_b(s)。

这是教学用玩具示例；真实 WEAT 使用 300 维预训练 embedding。

用法：python3 code/main.py
"""

from __future__ import annotations

import math


# 四维 embedding。轴 0 = “男性化”，1 = “女性化”，2 = “技术”，3 = “照护”。
EMB = {
    # 身份组 A。
    "he":        [ 1.0, 0.0, 0.2,  0.0],
    "his":       [ 0.9, 0.0, 0.1,  0.0],
    "man":       [ 1.0, 0.0, 0.1,  0.1],
    # 身份组 B。
    "she":       [ 0.0, 1.0, 0.0,  0.2],
    "her":       [ 0.0, 0.9, 0.0,  0.1],
    "woman":     [ 0.0, 1.0, 0.1,  0.2],
    # 属性 X：技术/职业。
    "engineer":  [ 0.4, 0.0, 1.0,  0.0],
    "programmer":[ 0.4, 0.0, 1.0,  0.0],
    "scientist": [ 0.3, 0.0, 1.0,  0.1],
    # 属性 Y：照护/家庭。
    "nurse":     [ 0.0, 0.4, 0.0,  1.0],
    "teacher":   [ 0.0, 0.3, 0.1,  1.0],
    "caregiver": [ 0.0, 0.4, 0.0,  1.0],
}


def cos(u: list[float], v: list[float]) -> float:
    nu = math.sqrt(sum(x * x for x in u)) + 1e-9
    nv = math.sqrt(sum(x * x for x in v)) + 1e-9
    return sum(a * b for a, b in zip(u, v)) / (nu * nv)


def weat_score(identity_a: list[str], identity_b: list[str],
               attr_x: list[str], attr_y: list[str]) -> float:
    def s(w):
        mx = sum(cos(EMB[w], EMB[a]) for a in attr_x) / len(attr_x)
        my = sum(cos(EMB[w], EMB[a]) for a in attr_y) / len(attr_y)
        return mx - my
    mean_a = sum(s(w) for w in identity_a) / len(identity_a)
    mean_b = sum(s(w) for w in identity_b) / len(identity_b)
    return mean_a - mean_b


def debias(emb: dict) -> dict:
    """粗略去偏：投影去除性别方向（轴 1 减轴 0）。"""
    new = {k: list(v) for k, v in emb.items()}
    gender_dir = [1.0, -1.0, 0.0, 0.0]
    norm_sq = sum(x * x for x in gender_dir)
    for w in ["engineer", "programmer", "scientist",
              "nurse", "teacher", "caregiver"]:
        proj = sum(a * b for a, b in zip(new[w], gender_dir)) / norm_sq
        new[w] = [a - proj * b for a, b in zip(new[w], gender_dir)]
    return new


def main() -> None:
    global EMB
    print("=" * 70)
    print("WEAT 偏见玩具探针（阶段 18，第 20 课）")
    print("=" * 70)

    A = ["he", "his", "man"]
    B = ["she", "her", "woman"]
    X = ["engineer", "programmer", "scientist"]
    Y = ["nurse", "teacher", "caregiver"]

    pre = weat_score(A, B, X, Y)
    print(f"\n去偏前的 WEAT 效应量：{pre:+.4f}")
    print("（正值表示身份组 A 与 X 的关联强于身份组 B。）")

    EMB = debias(EMB)
    post = weat_score(A, B, X, Y)
    print(f"去偏后的 WEAT 效应量：{post:+.4f}")

    print("\n" + "=" * 70)
    print("要点：基于 embedding 的偏见可以测量，也能通过投影去除性别相关方向")
    print("得到部分缓解。该指标没有降至零，因为玩具示例只有四维；真实去偏方法")
    print("（Bolukbasi 2016）在 300 维 embedding 上运行，可以减弱但无法消除")
    print("这一效应。还需要基于概率和生成文本的指标来捕捉残余行为偏见。")
    print("=" * 70)


if __name__ == "__main__":
    main()
