import random
from collections import defaultdict


GRID = 5
GOAL = (4, 4)
ACTIONS = ("up", "down", "left", "right")
DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def move(pos, action):
    dr, dc = DELTAS[action]
    r, c = pos
    return (min(max(r + dr, 0), GRID - 1), min(max(c + dc, 0), GRID - 1))


def reset():
    return ((0, 0), (GRID - 1, 0))


def step(state, action_pair):
    a1_pos, a2_pos = state
    new1 = move(a1_pos, action_pair[0])
    new2 = move(a2_pos, action_pair[1])
    done = (new1 == GOAL) and (new2 == GOAL)
    reward = 10.0 if done else -1.0
    return (new1, new2), reward, done


def default_q():
    return {a: 0.0 for a in ACTIONS}


def epsilon_greedy(q_table, state, rng, epsilon):
    if rng.random() < epsilon:
        return rng.choice(ACTIONS)
    q = q_table[state]
    return max(ACTIONS, key=lambda a: q[a])


def independent_q(episodes=1500, alpha=0.1, gamma=0.95, epsilon=0.15, rng=None):
    rng = rng or random.Random(0)
    Q1 = defaultdict(default_q)
    Q2 = defaultdict(default_q)
    returns_log = []
    for _ in range(episodes):
        s = reset()
        total = 0.0
        for _ in range(100):
            a1 = epsilon_greedy(Q1, s, rng, epsilon)
            a2 = epsilon_greedy(Q2, s, rng, epsilon)
            s_next, r, done = step(s, (a1, a2))
            total += r
            if done:
                Q1[s][a1] += alpha * (r - Q1[s][a1])
                Q2[s][a2] += alpha * (r - Q2[s][a2])
                break
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
        returns_log.append(total)
    return Q1, Q2, returns_log


def joint_q_learning(episodes=1500, alpha=0.1, gamma=0.95, epsilon=0.15, rng=None):
    rng = rng or random.Random(0)
    joint_actions = [(a, b) for a in ACTIONS for b in ACTIONS]
    Q = defaultdict(lambda: {ja: 0.0 for ja in joint_actions})
    returns_log = []
    for _ in range(episodes):
        s = reset()
        total = 0.0
        for _ in range(100):
            if rng.random() < epsilon:
                ja = rng.choice(joint_actions)
            else:
                ja = max(joint_actions, key=lambda a: Q[s][a])
            s_next, r, done = step(s, ja)
            total += r
            if done:
                Q[s][ja] += alpha * (r - Q[s][ja])
                break
            best_next = max(Q[s_next].values())
            Q[s][ja] += alpha * ((r + gamma * best_next) - Q[s][ja])
            s = s_next
        returns_log.append(total)
    return Q, returns_log


def block_mean(xs, block):
    return [sum(xs[i : i + block]) / block for i in range(0, len(xs) - block + 1, block)]


def evaluate_ind(Q1, Q2, episodes=100, rng=None):
    rng = rng or random.Random(42)
    total = 0.0
    for _ in range(episodes):
        s = reset()
        for _ in range(100):
            a1 = epsilon_greedy(Q1, s, rng, 0.0)
            a2 = epsilon_greedy(Q2, s, rng, 0.0)
            s, r, done = step(s, (a1, a2))
            total += r
            if done:
                break
    return total / episodes


def evaluate_joint(Q, episodes=100, rng=None):
    rng = rng or random.Random(42)
    joint_actions = [(a, b) for a in ACTIONS for b in ACTIONS]
    total = 0.0
    for _ in range(episodes):
        s = reset()
        for _ in range(100):
            ja = max(joint_actions, key=lambda a: Q[s][a])
            s, r, done = step(s, ja)
            total += r
            if done:
                break
    return total / episodes


def main():
    print(f"=== 双智能体协作 GridWorld（{GRID}x{GRID}，共享奖励）===")
    print(f"智能体从 (0,0) 和 ({GRID-1}, 0) 出发；两者都必须到达 {GOAL}")
    print()

    Q1, Q2, log_ind = independent_q(episodes=1500, rng=random.Random(1))
    Q_joint, log_joint = joint_q_learning(episodes=1500, rng=random.Random(1))

    print("学习曲线（每 150 个回合的平均回报）：")
    for i, (a, b) in enumerate(zip(block_mean(log_ind, 150), block_mean(log_joint, 150))):
        print(f"  分组 {i+1}：独立 Q = {a:7.2f}   联合 Q = {b:7.2f}")

    print()
    print("最终贪心评估（100 个回合）：")
    print(f"  独立 Q 平均回报 = {evaluate_ind(Q1, Q2):.2f}")
    print(f"  联合 Q 平均回报 = {evaluate_joint(Q_joint):.2f}")

    print()
    print("说明：联合 Q 能正确考虑全局视角，但其动作空间为 |A|^2。")
    print("CTDE 方法（MAPPO、QMIX）保留分散式 Actor，但使用集中式 Critic。")


if __name__ == "__main__":
    main()
