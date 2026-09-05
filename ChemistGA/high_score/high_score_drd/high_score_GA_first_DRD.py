import sys
import os
import warnings

# =================================================================
# 🛡️ 1. 终极防弹衣必须穿在最前面！先铺好所有路径！
# =================================================================
warnings.filterwarnings("ignore")

base_path = "/home/liuyansong/ChemistGA-master/ChemistGA-master"
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, "scoring"))
sys.path.insert(0, os.path.join(base_path, "transformer_model"))
# 确保能找到当前目录下的单靶点专属文件
sys.path.insert(0, os.path.join(base_path, "ChemistGA", "high_score", "high_score_drd"))
# ➕ 告诉系统外层的 ChemistGA 文件夹在哪（crossover.py 就躺在这里面！）
sys.path.insert(0, os.path.join(base_path, "ChemistGA"))

# =================================================================
# 📦 2. 路径铺好之后，再安安心心地引入各种模块！
# =================================================================
import time
import random
import numpy as np
import pandas as pd
from rdkit import Chem
import mutate as mu  # 现在它能顺藤摸瓜找到 crossover 了！

from high_score_properties_drd2 import multi_scoring_functions
from high_score_crossover_first_model_drd import get_synthesis_molecules
from pareto_ranking import pareto_selection
from high_score_properties_drd2 import get_scoring_function

# --- 下面继续接你原来的 def read_file(file_name): 等代码 ---

def read_file(file_name):
    smiles_list = pd.read_csv(file_name, header=None).values.flatten().tolist()
    return smiles_list


def make_initial_population(population_size, file_name):
    mol_list = read_file(file_name)
    population = []
    for i in range(population_size):
        population.append(random.choice(mol_list))
    return population


def calculate_3d_scores(population):
    """
    全新裁判系统（单靶点版）：给每个分子打出严格的 3 维独立分数！
    (DRD2, QED, SA) - 全部处理为越大越好。
    """
    # 1. 召唤三个裁判的实体
    drd2_scorer = get_scoring_function('drd2')
    qed_scorer = get_scoring_function('qed')
    sa_scorer = get_scoring_function('sa')

    # 2. 让裁判批量阅卷
    drd2_list = drd2_scorer(population)
    qed_list = qed_scorer(population)
    sa_list = sa_scorer(population)

    scores_3d = []
    for i in range(len(population)):
        # SA score 原始值越小越好，我们用 (10 - SA)/9 将其反转为越大越好。
        sa_normalized = (10.0 - sa_list[i]) / 9.0

        # 组装这名选手的 3 维雷达图数据
        scores_3d.append([
            float(drd2_list[i]),
            float(qed_list[i]),
            float(sa_normalized)
        ])

    return scores_3d



def tournament_selection(pool_with_scores, k=2):
    """
    锦标赛机制：随机抽 k 个人，选总分最高的胜出
    """
    competitors = random.sample(pool_with_scores, k)
    competitors.sort(key=lambda x: x[1], reverse=True)
    return competitors[0][0]


def reproduce(mating_pool_with_scores, population_size, mutation_rate):
    parent_population = []
    new_population = []

    while len(parent_population) < population_size:
        # ✨ 盲婚哑嫁 升级为 锦标赛竞争！
        parent_A = tournament_selection(mating_pool_with_scores, k=2)
        parent_B = tournament_selection(mating_pool_with_scores, k=2)
        while parent_A == parent_B:
            parent_B = tournament_selection(mating_pool_with_scores, k=2)
        parent_list = [parent_A, parent_B]
        parent_list.sort()
        parent_population.append('.'.join(parent_list))
        parent_population = list(set(parent_population))

    score_list, new_child, all_population, all_population_score = get_synthesis_molecules(parent_population)

    drop_score_index = []
    for index, mol_str in enumerate(new_child):
        mol = Chem.MolFromSmiles(mol_str)
        if mol != None:
            mol_new_child = mu.mutate(mol, mutation_rate)
            if mol_new_child != None:
                new_population.append(Chem.MolToSmiles(mol_new_child))
            else:
                drop_score_index.append(index)
        else:
            drop_score_index.append(index)

    score_list = np.array(score_list)
    if len(drop_score_index) > 0:
        score_list = np.delete(score_list, drop_score_index, axis=0)
    score_list = score_list.tolist()

    return score_list, new_population, all_population, all_population_score


# --- 2. 修复暗雷 ---
population_size = 100
generations = 50
mutation_rate = 0.05

print('population_size', population_size)
print('generations', generations)
print('mutation_rate', mutation_rate)
print('')

# 修复暗雷一：指向正确的单靶点库！
file_name = os.path.join(base_path, 'data/inh/drd_succ_250.csv')

# 修复暗雷二：自动创建单靶点输出文件夹，防止保存时崩溃且不与双靶点冲突
# 👈 请直接用这两行替换原来的代码
out_dir_top3 = os.path.join(base_path, 'output', 'drd_top3_continuous')
out_dir_all = os.path.join(base_path, 'output', 'drd_all_pop_continuous')
os.makedirs(out_dir_top3, exist_ok=True)
os.makedirs(out_dir_all, exist_ok=True)

results = []
size = []
t0 = time.time()
all_active_list = []

for i in range(3):  # 测试先跑 1 个大循环
    max_score = [-99999., '']
    population = make_initial_population(population_size, file_name)
    age = 0
    score_list = None

    try:
        # ✨ 修复：在第 0 代进产房前，强制给老祖宗算分数
        print("给初始老祖宗进行入职体检...")
        initial_scores_3d = calculate_3d_scores(population)
        score_list = [sum(scores) for scores in initial_scores_3d]

        for generation in range(generations):
            # ✨ 将分子和对应的参考总分打包，送入锦标赛！
            mating_pool_with_scores = list(zip(population, score_list))

            # 2. 调用 3080 Ti 生成婴儿
            _, new_children, all_population, _ = reproduce(mating_pool_with_scores, population_size, mutation_rate)

            # 3. 对生出来的整个大池子（父母+婴儿），进行严格的 3 维打分
            print(f"🧬 正在对第 {age} 代的 {len(all_population)} 名候选者进行 3 维体检...")
            all_scores_3d = calculate_3d_scores(all_population)

            # 4. 黄金折中战术 A：帕累托只挑最顶级的 50 名“皇室精英”（死死保住高分！）
            elite_size = int(population_size / 2)
            # elite_size = 40  # 👈 把它改成这样！(如果需要可以自己调)
            elites, elites_scores = pareto_selection(all_population, all_scores_3d, elite_size)

            # 4. 黄金折中战术 B：把被淘汰的平民找出来（同时要打包保留他们的 3 维分数）
            elite_set = set(elites)
            remaining_pairs = []
            for idx in range(len(all_population)):
                if all_population[idx] not in elite_set:
                    remaining_pairs.append((all_population[idx], all_scores_3d[idx]))

            # 闭着眼睛从平民里盲抽 50 个幸运异种（死死保住新颖性和多样性！）
            lucky_size = population_size - elite_size
            if len(remaining_pairs) >= lucky_size:
                lucky_pairs = random.sample(remaining_pairs, lucky_size)
            else:
                lucky_pairs = remaining_pairs

            lucky_losers = [p[0] for p in lucky_pairs]
            lucky_scores = [p[1] for p in lucky_pairs]

            # 4. 黄金折中战术 C：精英与平民会师，组建全新一代 100 人交配池！
            population = elites + lucky_losers
            selected_scores_3d = elites_scores + lucky_scores

            # 5. 为了兼容存盘逻辑，把 3 维分数加起来算个“参考总分”
            score_list = [sum(scores) for scores in selected_scores_3d]
            all_population_score = [sum(scores) for scores in all_scores_3d]

            # --- 🏆 混合双打选拔区结束 🏆 ---

            # 保存文件
            every_pop = pd.concat([pd.DataFrame(population), pd.DataFrame(score_list)], axis=1)
            every_pop.to_csv(os.path.join(out_dir_top3, f"{i}_{age}_every_top3.csv"), index=False, header=False,
                             mode='w')

            all_pop = pd.concat([pd.DataFrame(all_population), pd.DataFrame(all_population_score)], axis=1)
            all_pop.to_csv(os.path.join(out_dir_all, f"{i}_{age}_every_all.csv"), index=False, header=False, mode='a')

            age += 1
            print(f'Generation {age} done. Current valid population size: {len(population)}')

    except Exception as e:
        import traceback
        traceback.print_exc()
        continue

t1 = time.time()
print(f'\nTotal time: {t1 - t0:.2f} seconds')