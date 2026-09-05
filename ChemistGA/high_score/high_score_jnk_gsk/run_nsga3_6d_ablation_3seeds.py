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
sys.path.insert(0, os.path.join(base_path, "ChemistGA", "high_score", "high_score_jnk_gsk"))
sys.path.insert(0, os.path.join(base_path, "ChemistGA"))

import mutate as mu
from high_score_properties_jnk_gsk import get_scoring_function

# ================= 🚨 消融实验核心：导入传统 GB-GA 交叉模块 =================
import gbga_crossover as co

# 🧙‍♂️ 导师神级 Hack：向原作者代码中注入缺失的全局变量，保证原汁原味！
co.average_size = 39.15
co.size_stdev = 3.50
print("⚠️ [消融模式] 已成功加载传统 GB-GA 交叉算法，并注入尺寸约束参数！")


# =========================================================================

def read_file(file_name):
    smiles_list = pd.read_csv(file_name, header=None).values.flatten().tolist()
    return smiles_list


def make_initial_population(population_size, file_name):
    mol_list = read_file(file_name)
    population = []
    for i in range(population_size):
        population.append(random.choice(mol_list))
    return population


# ================= 🚀 6 维超级打分器 =================
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


# ================= 🚀 NSGA-III 高维淘汰法则 =================
def nsga3_environmental_selection(combined_smiles, combined_scores, num_select=100):
    F = -np.array(combined_scores)
    n_obj = 6

    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=4)
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


# ================= 🚨 消融替换核心区：reproduce =================
def reproduce(mating_pool_with_scores, population_size, mutation_rate):
    parent_population = []
    new_population = []

    # 1. 锦标赛选拔父母
    while len(parent_population) < population_size:
        parent_A = tournament_selection(mating_pool_with_scores, k=2)
        parent_B = tournament_selection(mating_pool_with_scores, k=2)
        while parent_A == parent_B:
            parent_B = tournament_selection(mating_pool_with_scores, k=2)
        parent_list = [parent_A, parent_B]
        parent_list.sort()
        parent_population.append('.'.join(parent_list))
        parent_population = list(set(parent_population))

    # 2. 调用传统 GB-GA 交叉生孩子（完全摘除大模型）
    new_child = []
    print(f"      [Ablation] 正在使用传统 GB-GA 物理拼接为 {len(parent_population)} 对父母繁衍后代...")

    for parent_pair_str in parent_population:
        parents = parent_pair_str.split('.')
        if len(parents) == 2:
            mol_A = Chem.MolFromSmiles(parents[0])
            mol_B = Chem.MolFromSmiles(parents[1])
            if mol_A and mol_B:
                child_mol = co.crossover(mol_A, mol_B)
                if child_mol is not None:
                    new_child.append(Chem.MolToSmiles(child_mol))
                else:
                    new_child.append(parents[0])  # 拼接失败，父亲保底
            else:
                new_child.append(parents[0])

    # 3. 婴儿突变
    for index, mol_str in enumerate(new_child):
        mol = Chem.MolFromSmiles(mol_str)
        if mol != None:
            mol_new_child = mu.mutate(mol, mutation_rate)
            if mol_new_child != None:
                new_population.append(Chem.MolToSmiles(mol_new_child))

    return new_population


# ================================================================

# --- 运行配置 ---
population_size = 100
generations = 50
mutation_rate = 0.05

print('population_size', population_size)
print('generations', generations)
print('mutation_rate', mutation_rate)
print('')

file_name = os.path.join(base_path, 'data/inh/jnk_gsk.csv')

# 🚨 修改输出文件夹，标记为 Ablation (消融实验)
out_dir_top3 = os.path.join(base_path, 'output', 'ablation_gbga_nsga3_6d_top')
out_dir_all = os.path.join(base_path, 'output', 'ablation_gbga_nsga3_6d_all')
os.makedirs(out_dir_top3, exist_ok=True)
os.makedirs(out_dir_all, exist_ok=True)

t0 = time.time()

# 🚨 导师修改：跑满 3 个 Seed，确保统计数据的严谨性！
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

            # 1. 繁衍
            offspring_smiles = reproduce(mating_pool_with_scores, population_size, mutation_rate)

            # 2. 体检
            print(f"🧬 [Seed {i} - Age {age}] 正在对新出生的 {len(offspring_smiles)} 名婴儿进行 6 维体检...")
            offspring_scores_6d = calculate_6d_scores(offspring_smiles)

            combined_smiles = population + offspring_smiles
            combined_scores = parent_scores_6d + offspring_scores_6d

            # 3. 淘汰
            population, parent_scores_6d = nsga3_environmental_selection(combined_smiles, combined_scores,
                                                                         population_size)

            score_list = [sum(scores) for scores in parent_scores_6d]
            combined_sum_scores = [sum(scores) for scores in combined_scores]

            every_pop = pd.concat([pd.DataFrame(population), pd.DataFrame(score_list)], axis=1)
            every_pop.to_csv(os.path.join(out_dir_top3, f"{i}_{age}_every_top.csv"), index=False, header=False,
                             mode='w')

            all_pop = pd.concat([pd.DataFrame(combined_smiles), pd.DataFrame(combined_sum_scores)], axis=1)
            all_pop.to_csv(os.path.join(out_dir_all, f"{i}_{age}_every_all.csv"), index=False, header=False, mode='a')

            age += 1
            print(f'✅ Seed {i} - Generation {age} done.')

    except Exception as e:
        import traceback

        traceback.print_exc()
        continue

t1 = time.time()
print(f'\nTotal time: {t1 - t0:.2f} seconds')