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

# =================================================================
# 🛡️ 1. 终极防弹衣与依赖引入
# =================================================================
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

base_path = "/home/liuyansong/ChemistGA-master/ChemistGA-master"
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, "scoring"))
sys.path.insert(0, os.path.join(base_path, "transformer_model"))
sys.path.insert(0, os.path.join(base_path, "ChemistGA", "high_score", "high_score_drd"))
sys.path.insert(0, os.path.join(base_path, "ChemistGA"))

import mutate as mu
from high_score_crossover_first_model_drd import get_synthesis_molecules
from high_score_properties_drd2 import get_scoring_function


def read_file(file_name):
    smiles_list = pd.read_csv(file_name, header=None).values.flatten().tolist()
    return smiles_list


def make_initial_population(population_size, file_name):
    mol_list = read_file(file_name)
    population = []
    for i in range(population_size):
        population.append(random.choice(mol_list))
    return population


# ================= 🚀 核心 1：单靶点 3 维专属打分器 =================
def calculate_3d_scores(population):
    """
    (DRD2, QED, SA) - 全部处理为越大越好。
    """
    drd2_scorer = get_scoring_function('drd2')
    qed_scorer = get_scoring_function('qed')
    sa_scorer = get_scoring_function('sa')

    drd2_list = drd2_scorer(population)
    qed_list = qed_scorer(population)
    sa_list = sa_scorer(population)

    scores_3d = []
    for i in range(len(population)):
        smi = population[i]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scores_3d.append([0.0] * 3)
            continue

        drd2_val = float(drd2_list[i])
        qed_val = float(qed_list[i])
        # SA score 反转为越大越好
        sa_normalized = max(0.0, (10.0 - float(sa_list[i])) / 9.0)

        scores_3d.append([drd2_val, qed_val, sa_normalized])

    return scores_3d


# ================= 🚀 核心 2：NSGA-III 高维淘汰法则 =================
def nsga3_environmental_selection(combined_smiles, combined_scores, num_select=100):
    """
    使用 NSGA-III 的参考点机制进行 3 维空间的淘汰
    取代了之前的 pareto_selection
    """
    F = -np.array(combined_scores)
    n_obj = 3  # 3维目标

    # 在 3 维空间部署灯塔 (n_partitions=12 约生成 91 个参考点，适合 100 人种群)
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)

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


# ================= 🚀 核心 3：防爆显存分批生产 =================
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

        # 进产房前冲水
        torch.cuda.empty_cache()

        # 闭梯度，极大幅度节约显存
        with torch.no_grad():
            score_list_chunk, chunk_children, _, _ = get_synthesis_molecules(parent_chunk)

        new_child.extend(chunk_children)

    # --- 婴儿突变 ---
    for index, mol_str in enumerate(new_child):
        mol = Chem.MolFromSmiles(mol_str)
        if mol != None:
            mol_new_child = mu.mutate(mol, mutation_rate)
            if mol_new_child != None:
                new_population.append(Chem.MolToSmiles(mol_new_child))

    # 离开前终极冲水
    torch.cuda.empty_cache()
    return new_population


# =================================================================
# 🏃 2. 运行配置与主循环
# =================================================================
population_size = 100
generations = 50
mutation_rate = 0.05

print('population_size', population_size)
print('generations', generations)
print('mutation_rate', mutation_rate)
print('')

file_name = os.path.join(base_path, 'data/inh/drd_succ_250.csv')

# 改成专属的 nsga3 输出文件夹
out_dir_top3 = os.path.join(base_path, 'output', 'drd_nsga3_3d_top')
out_dir_all = os.path.join(base_path, 'output', 'drd_nsga3_3d_all')
os.makedirs(out_dir_top3, exist_ok=True)
os.makedirs(out_dir_all, exist_ok=True)

t0 = time.time()

for i in range(3):
    population = make_initial_population(population_size, file_name)
    age = 0
    parent_scores_3d = []

    try:
        print(f"\n🚀 Seed {i}: 正在给初始老祖宗进行 3 维体检...")
        parent_scores_3d = calculate_3d_scores(population)

        for generation in range(generations):
            sum_scores = [sum(scores) for scores in parent_scores_3d]
            mating_pool_with_scores = list(zip(population, sum_scores))

            # 1. 繁衍 (安全分批版)
            offspring_smiles = reproduce(mating_pool_with_scores, population_size, mutation_rate)

            # 2. 3维打分
            print(f"🧬 [Age {age}] 正在对新出生的 {len(offspring_smiles)} 名婴儿进行 3 维体检...")
            offspring_scores_3d = calculate_3d_scores(offspring_smiles)

            combined_smiles = population + offspring_smiles
            combined_scores = parent_scores_3d + offspring_scores_3d

            # 3. NSGA-III 裁判登场！取代了原来的手工 Pareto！
            population, parent_scores_3d = nsga3_environmental_selection(combined_smiles, combined_scores,
                                                                         population_size)

            score_list = [sum(scores) for scores in parent_scores_3d]
            combined_sum_scores = [sum(scores) for scores in combined_scores]

            # 4. 存盘
            every_pop = pd.concat([pd.DataFrame(population), pd.DataFrame(score_list)], axis=1)
            every_pop.to_csv(os.path.join(out_dir_top3, f"{i}_{age}_every_top.csv"), index=False, header=False,
                             mode='w')

            all_pop = pd.concat([pd.DataFrame(combined_smiles), pd.DataFrame(combined_sum_scores)], axis=1)
            all_pop.to_csv(os.path.join(out_dir_all, f"{i}_{age}_every_all.csv"), index=False, header=False, mode='a')

            age += 1
            print(f'✅ Generation {age} done. NSGA-III 筛选完毕，当前精英池人数: {len(population)}')

    except Exception as e:
        import traceback

        traceback.print_exc()
        continue

t1 = time.time()
print(f'\nTotal time: {t1 - t0:.2f} seconds')