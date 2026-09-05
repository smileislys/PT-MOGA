import sys
import os
import warnings
import math
import time
import random
import numpy as np
import pandas as pd
from rdkit import Chem
import torch

# --- pymoo: NSGA-III 核心组件 ---
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.config import Config

Config.warnings['not_compiled'] = False

warnings.filterwarnings("ignore")
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# ================= 1. 路径设置 (双靶点 JNK3/GSK3) =================
base_path = "/home/liuyansong/ChemistGA-master/ChemistGA-master"
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, "scoring"))
sys.path.insert(0, os.path.join(base_path, "transformer_model"))
sys.path.insert(0, os.path.join(base_path, "ChemistGA", "high_score", "high_score_jnk_gsk"))
sys.path.insert(0, os.path.join(base_path, "ChemistGA"))

import mutate as mu
from nsga3_6d import get_synthesis_molecules
from high_score_properties_jnk_gsk import get_scoring_function


def read_file(file_name):
    smiles_list = pd.read_csv(file_name, header=None).values.flatten().tolist()
    return smiles_list


def make_initial_population(population_size, file_name):
    mol_list = read_file(file_name)
    population = []
    for i in range(population_size):
        population.append(random.choice(mol_list))
    return population


# ================= 🚀 核心 1：4维专属打分器 (JNK3, GSK3, QED, SA) =================
def calculate_4d_scores(population):
    """
    给每个分子打出严格的 4 维独立分数！
    (JNK3, GSK3, QED, SA_norm) - 全部处理为越大越好。
    """
    jnk3_scorer = get_scoring_function('jnk3')
    gsk3_scorer = get_scoring_function('gsk3')
    qed_scorer = get_scoring_function('qed')
    sa_scorer = get_scoring_function('sa')  # 👈 新增 SA 裁判

    jnk3_list = jnk3_scorer(population)
    gsk3_list = gsk3_scorer(population)
    qed_list = qed_scorer(population)
    sa_list = sa_scorer(population)

    scores_4d = []
    for i in range(len(population)):
        smi = population[i]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scores_4d.append([0.0] * 4)
            continue

        jnk3_val = float(jnk3_list[i])
        gsk3_val = float(gsk3_list[i])
        qed_val = float(qed_list[i])

        # SA score 原始值越小越好，用 (10 - SA)/9 反转为越大越好
        sa_norm = max(0.0, (10.0 - float(sa_list[i])) / 9.0)

        scores_4d.append([jnk3_val, gsk3_val, qed_val, sa_norm])  # 👈 4个维度

    return scores_4d


# ================= 🚀 核心 2：NSGA-III 4维环境选择 =================
def nsga3_environmental_selection(combined_smiles, combined_scores, num_select=100):
    """
    使用 NSGA-III 的参考点机制进行 4 维空间的超平面淘汰
    """
    F = -np.array(combined_scores)
    n_obj = 4  # 👈 明确告诉引擎现在是 4 维（超多目标）！

    # 在 4 维空间部署灯塔 (n_partitions=7 刚好能生成 120 个参考点，非常适合 100 人种群)
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=7)

    indices = np.arange(len(combined_smiles))
    pop = Population.new(X=indices, F=F)
    dummy_problem = Problem(n_var=1, n_obj=n_obj, n_ieq_constr=0)

    survival = ReferenceDirectionSurvival(ref_dirs)
    survivors = survival.do(dummy_problem, pop, n_survive=num_select)

    survivor_indices = [ind.get("X") for ind in survivors]
    selected_smiles = [combined_smiles[i] for i in survivor_indices]
    selected_scores = [combined_scores[i] for i in survivor_indices]

    return selected_smiles, selected_scores


def tournament_selection(pool_with_scores, k=2):
    competitors = random.sample(pool_with_scores, k)
    competitors.sort(key=lambda x: x[1], reverse=True)
    return competitors[0][0]


# ================= 🚀 核心 3：防爆显存的繁衍产房 =================
def reproduce(mating_pool_with_scores, population_size, mutation_rate):
    parent_population = []
    new_population = []

    while len(parent_population) < population_size:
        parent_A = tournament_selection(mating_pool_with_scores, k=2)
        parent_B = tournament_selection(mating_pool_with_scores, k=2)
        while parent_A == parent_B:
            parent_B = tournament_selection(mating_pool_with_scores, k=2)
        parent_list = [parent_A, parent_B]
        parent_list.sort()
        parent_population.append('.'.join(parent_list))
        parent_population = list(set(parent_population))

    # --- 引入“分批喂食”防爆显存机制 ---
    chunk_size = 20
    new_child = []

    for i in range(0, len(parent_population), chunk_size):
        parent_chunk = parent_population[i:i + chunk_size]

        torch.cuda.empty_cache()

        with torch.no_grad():
            _, chunk_children, _, _ = get_synthesis_molecules(parent_chunk)

        new_child.extend(chunk_children)

    for index, mol_str in enumerate(new_child):
        mol = Chem.MolFromSmiles(mol_str)
        if mol != None:
            mol_new_child = mu.mutate(mol, mutation_rate)
            if mol_new_child != None:
                new_population.append(Chem.MolToSmiles(mol_new_child))

    torch.cuda.empty_cache()
    return new_population


# --- 2. 运行配置 ---
population_size = 100
generations = 50
mutation_rate = 0.05

print('population_size', population_size)
print('generations', generations)
print('mutation_rate', mutation_rate)
print('')

# 使用双靶点库
file_name = os.path.join(base_path, 'data/inh/jnk_gsk.csv')

# 👇 核心 4：输出路径改为 4D NSGA-III 的专属文件夹
out_dir_top3 = os.path.join(base_path, 'output', 'jnk_gsk_nsga3_4d_top')
out_dir_all = os.path.join(base_path, 'output', 'jnk_gsk_nsga3_4d_all')
os.makedirs(out_dir_top3, exist_ok=True)
os.makedirs(out_dir_all, exist_ok=True)

t0 = time.time()

for i in range(3):
    population = make_initial_population(population_size, file_name)
    age = 0
    parent_scores_4d = []

    try:
        print(f"\n🚀 Seed {i}: 正在给初始老祖宗进行 4 维体检...")
        parent_scores_4d = calculate_4d_scores(population)

        for generation in range(generations):
            sum_scores = [sum(scores) for scores in parent_scores_4d]
            mating_pool_with_scores = list(zip(population, sum_scores))

            # 1. 繁衍
            offspring_smiles = reproduce(mating_pool_with_scores, population_size, mutation_rate)

            # 2. 4维打分
            print(f"🧬 [Age {age}] 正在对新出生的 {len(offspring_smiles)} 名婴儿进行 4 维体检...")
            offspring_scores_4d = calculate_4d_scores(offspring_smiles)

            combined_smiles = population + offspring_smiles
            combined_scores = parent_scores_4d + offspring_scores_4d

            # 3. NSGA-III 4维环境选择
            population, parent_scores_4d = nsga3_environmental_selection(combined_smiles, combined_scores,
                                                                         population_size)

            score_list = [sum(scores) for scores in parent_scores_4d]
            combined_sum_scores = [sum(scores) for scores in combined_scores]

            # 保存
            every_pop = pd.concat([pd.DataFrame(population), pd.DataFrame(score_list)], axis=1)
            every_pop.to_csv(os.path.join(out_dir_top3, f"{i}_{age}_every_top.csv"), index=False, header=False,
                             mode='w')

            all_pop = pd.concat([pd.DataFrame(combined_smiles), pd.DataFrame(combined_sum_scores)], axis=1)
            all_pop.to_csv(os.path.join(out_dir_all, f"{i}_{age}_every_all.csv"), index=False, header=False, mode='a')

            age += 1
            print(f'✅ Generation {age} done. NSGA-III 4维超平面筛选完毕！')

    except Exception as e:
        import traceback

        traceback.print_exc()
        continue

t1 = time.time()
print(f'\nTotal time: {t1 - t0:.2f} seconds')