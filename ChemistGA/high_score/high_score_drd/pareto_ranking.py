import numpy as np


def dominates(score_p, score_q):
    """
    深度思考 1：定义什么是“绝对碾压”（Domination）
    规则：在多目标优化中，A 支配 B 的条件是：
    A 的所有指标都 >= B，且 A 至少有一个指标 > B。
    这里假设我们的 4 个目标（JNK3, GSK3, QED, SA）都是越大越好。
    """
    p_beats_q = False
    for p_val, q_val in zip(score_p, score_q):
        if p_val < q_val:
            return False  # 只要 p 在任何一个指标上输给了 q，p 就不能支配 q
        elif p_val > q_val:
            p_beats_q = True  # p 在某个指标上赢了 q
    return p_beats_q


def fast_non_dominated_sort(scores):
    """
    深度思考 2：非支配排序（寻找帕累托前沿）
    修复版：解决了最后一层梯队索引越界的 Bug
    """
    num_individuals = len(scores)
    domination_set = {i: [] for i in range(num_individuals)}
    dominated_count = {i: 0 for i in range(num_individuals)}
    fronts = [[]]

    # 双重循环，两两比对
    for p in range(num_individuals):
        for q in range(num_individuals):
            if p == q: continue
            if dominates(scores[p], scores[q]):
                domination_set[p].append(q)
            elif dominates(scores[q], scores[p]):
                dominated_count[p] += 1

        # 如果 p 没有被任何人支配，他就是天之骄子（第一梯队）
        if dominated_count[p] == 0:
            fronts[0].append(p)

    # --- 修复后的核心分层逻辑 ---
    current_front = 0
    # 只要当前梯队存在且里面有人，就继续往下找
    while current_front < len(fronts) and len(fronts[current_front]) > 0:
        next_front = []
        for p in fronts[current_front]:
            for q in domination_set[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)

        # 只有在找到了下一层梯队成员时，才把他们加入总名单
        if next_front:
            fronts.append(next_front)

        current_front += 1

    return fronts


def calculate_crowding_distance(scores, front):
    """
    深度思考 3：拥挤度计算（保护少数派）
    逻辑：如果大家都在同一个梯队（比如都是 Front 1），名额不够了怎么淘汰？
    算法会看每个人周围有多挤。在某个指标上独一无二的分子（边界点）拥挤度为无穷大，绝对保留。
    扎堆长得像的分子，拥挤度小，优先淘汰。这是拉升 Novelty 的绝对核心！
    """
    num_individuals = len(front)
    distances = {i: 0.0 for i in front}
    if num_individuals <= 2:
        for i in front: distances[i] = float('inf')
        return distances

    num_objectives = len(scores[0])

    for obj_index in range(num_objectives):
        # 按照当前维度的分数对这一梯队的人进行排序
        sorted_front = sorted(front, key=lambda i: scores[i][obj_index])

        # 边界分子（极端的特长生）永远保送
        distances[sorted_front[0]] = float('inf')
        distances[sorted_front[-1]] = float('inf')

        # 找出当前维度的最大和最小值，用于归一化
        min_val = scores[sorted_front[0]][obj_index]
        max_val = scores[sorted_front[-1]][obj_index]

        if max_val == min_val:
            continue

        # 计算中间分子的拥挤度（与前后分子的距离之和）
        for i in range(1, num_individuals - 1):
            if distances[sorted_front[i]] == float('inf'): continue
            prev_idx = sorted_front[i - 1]
            next_idx = sorted_front[i + 1]
            distances[sorted_front[i]] += (scores[next_idx][obj_index] - scores[prev_idx][obj_index]) / (
                        max_val - min_val)

    return distances


def pareto_selection(population, scores, num_select):
    """
    主帅登场：帕累托大选拔
    输入：population (所有候选分子), scores (每个分子的4维分数列表), num_select (需要挑选的晋级人数，通常是100)
    输出：晋级的 population, 以及他们对应的 scores
    """
    # 1. 给所有人分梯队
    fronts = fast_non_dominated_sort(scores)

    selected_indices = []

    # 2. 从 Front 1 开始，把最强梯队依次塞进入围名单
    for front in fronts:
        if len(selected_indices) + len(front) <= num_select:
            # 如果这个梯队的人数刚好能塞下，全员晋级
            selected_indices.extend(front)
        else:
            # 如果加上这个梯队就超载了，说明要在这个梯队里进行“残酷淘汰”
            # 这时候开启“拥挤度”护体机制
            distances = calculate_crowding_distance(scores, front)
            # 按照拥挤度从大到小（越稀缺越前面）排序
            front.sort(key=lambda i: distances[i], reverse=True)
            # 填满剩下的名额
            remaining_slots = num_select - len(selected_indices)
            selected_indices.extend(front[:remaining_slots])
            break  # 名额已满，选拔结束

    # 根据选出的索引，返回最终活下来的分子和他们的分数
    selected_population = [population[i] for i in selected_indices]
    selected_scores = [scores[i] for i in selected_indices]

    return selected_population, selected_scores