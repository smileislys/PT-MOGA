import sys
import os
import warnings
import math
import time
import random
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
import torch  # 👈 必须引入 torch 才能管理显存！

# --- pymoo: NSGA-II 核心组件 (替代 NSGA-III) ---
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.population import Population
from pymoo.core.problem import Problem

# 让 pymoo 保持终端清爽
from pymoo.config import Config

Config.warnings['not_compiled'] = False

# --- 1. 终极防弹衣：路径与警告修正 ---
warnings.filterwarnings("ignore")
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

base_path = "/home/liuyansong/ChemistGA-master/ChemistGA-master"
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, "scoring"))
sys.path.insert(0, os.path.join(base_path, "transformer_model"))
sys.path.insert(0, os.path.join(base_path, "ChemistGA", "high_score", "high_score_jnk_gsk"))
sys.path.insert(0, os.path.join(base_path, "ChemistGA"))

import mutate as mu
from high_score_properties_jnk_gsk import multi_scoring_functions
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


# ================= 🚀 6 维打分器 (保持不变，保证公平基准) =================
def calculate_6d_scores(population):
    jnk3_scorer = get_scoring_function('jnk3')
    gsk3_scorer = get_scoring_function('gsk3')
    qed_scorer = get_scoring_function('qed')
    sa_scorer = get_scoring_function('sa')

    jnk3_list = jnk3_scorer(population)
    gsk3_list = gsk3_scorer(population)
    qed_list = qed_scorer(population)
    sa_list = sa_scorer(population)

    scores_6d = []
    for i in range(len(population)):
        smi = population[i]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scores_6d.append([0.0] * 6)
            continue

        jnk3_val = float(jnk3_list[i])
        gsk3_val = float(gsk3_list[i])
        qed_val = float(qed_list[i])
        sa_norm = max(0.0, (10.0 - float(sa_list[i])) / 9.0)

        try:
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            tpsa_norm = math.exp(-0.5 * ((tpsa - 60.0) / 30.0) ** 2)
        except:
            tpsa_norm = 0.0

        try:
            logp = Crippen.MolLogP(mol)
            logp_norm = math.exp(-0.5 * ((logp - 3.0) / 1.5) ** 2)
        except:
            logp_norm = 0.0

        scores_6d.append([jnk3_val, gsk3_val, qed_val, sa_norm, tpsa_norm, logp_norm])

    return scores_6d


# ================= 🚀 核心替换：NSGA-II 低维引擎强行去跑 6 维 =================
def nsga2_environmental_selection(combined_smiles, combined_scores, num_select=100):
    """
    使用 NSGA-II 的拥挤度距离 (Crowding Distance) 进行淘汰
    """
    F = -np.array(combined_scores)
    n_obj = 6

    # 构造种群
    indices = np.arange(len(combined_smiles))
    pop = Population.new(X=indices, F=F)

    # 临时伪造一个 6 目标的空 Problem
    dummy_problem = Problem(n_var=1, n_obj=n_obj, n_ieq_constr=0)

    # 👈 执行 NSGA-II 的降维打击 (Rank and Crowding Distance)
    survival = RankAndCrowdingSurvival()
    survivors = survival.do(dummy_problem, pop, n_survive=num_select)

    survivor_indices = [ind.get("X") for ind in survivors]

    selected_smiles = [combined_smiles[i] for i in survivor_indices]
    selected_scores = [combined_scores[i] for i in survivor_indices]

    return selected_smiles, selected_scores


def tournament_selection(pool_with_scores, k=2):
    competitors = random.sample(pool_with_scores, k)
    competitors.sort(key=lambda x: x[1], reverse=True)
    return competitors[0][0]


# ================= 🚀 核心装甲：防爆显存的 reproduce =================
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

        # 进产房前，强制冲水
        torch.cuda.empty_cache()

        # 使用 no_grad 强制关闭梯度计算
        with torch.no_grad():
            _, chunk_children, _, _ = get_synthesis_molecules(parent_chunk)

        new_child.extend(chunk_children)
    # ----------------------------------

    for index, mol_str in enumerate(new_child):
        mol = Chem.MolFromSmiles(mol_str)
        if mol != None:
            mol_new_child = mu.mutate(mol, mutation_rate)
            if mol_new_child != None:
                new_population.append(Chem.MolToSmiles(mol_new_child))

    # 离开前终极冲水
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

file_name = os.path.join(base_path, 'data/inh/jnk_gsk.csv')

# 输出文件夹精准对齐 nsga2_6d 任务
out_dir_top3 = os.path.join(base_path, 'output', 'jnk_gsk_nsga2_6d_top')
out_dir_all = os.path.join(base_path, 'output', 'jnk_gsk_nsga2_6d_all')
os.makedirs(out_dir_top3, exist_ok=True)
os.makedirs(out_dir_all, exist_ok=True)

t0 = time.time()

for i in range(3):
    population = make_initial_population(population_size, file_name)
    age = 0
    parent_scores_6d = []

    try:
        print(f"\n🚀 Seed {i}: 正在给初始老祖宗进行 6 维入职体检...")
        parent_scores_6d = calculate_6d_scores(population)

        for generation in range(generations):
            sum_scores = [sum(scores) for scores in parent_scores_6d]
            mating_pool_with_scores = list(zip(population, sum_scores))

            # 1. 繁衍出新一代婴儿 (Offspring)
            offspring_smiles = reproduce(mating_pool_with_scores, population_size, mutation_rate)

            # 2. 给新生的婴儿进行 6 维打分
            print(f"🧬 [Age {age}] 正在对新出生的 {len(offspring_smiles)} 名婴儿进行 6 维体检...")
            offspring_scores_6d = calculate_6d_scores(offspring_smiles)

            combined_smiles = population + offspring_smiles
            combined_scores = parent_scores_6d + offspring_scores_6d

            # 3. 召唤 NSGA-II 裁判
            population, parent_scores_6d = nsga2_environmental_selection(combined_smiles, combined_scores,
                                                                         population_size)

            score_list = [sum(scores) for scores in parent_scores_6d]
            combined_sum_scores = [sum(scores) for scores in combined_scores]

            # 保存文件
            every_pop = pd.concat([pd.DataFrame(population), pd.DataFrame(score_list)], axis=1)
            every_pop.to_csv(os.path.join(out_dir_top3, f"{i}_{age}_every_top.csv"), index=False, header=False,
                             mode='w')

            all_pop = pd.concat([pd.DataFrame(combined_smiles), pd.DataFrame(combined_sum_scores)], axis=1)
            all_pop.to_csv(os.path.join(out_dir_all, f"{i}_{age}_every_all.csv"), index=False, header=False, mode='a')

            age += 1
            print(f'✅ Generation {age} done. NSGA-II 艰难筛选完毕！')

    except Exception as e:
        import traceback

        traceback.print_exc()
        continue

t1 = time.time()
print(f'\nTotal time: {t1 - t0:.2f} seconds')