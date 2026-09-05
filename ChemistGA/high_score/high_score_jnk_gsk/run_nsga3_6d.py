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

# --- pymoo: NSGA-III 核心组件 (适配 pymoo 0.6.x 最新版) ---
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.core.population import Population
from pymoo.core.problem import Problem

# 👇 加上这两行，让 pymoo 彻底闭嘴，保持终端清爽！
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
# 🚀 导师修改区：导入你刚刚改名后的产房模块 nsga3_6d！
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


# ================= 🚀 核心升级 1：6 维超级打分器 =================
def calculate_6d_scores(population):
    """
    给每个分子打出严格的 6 维独立分数！
    (JNK3, GSK3, QED, SA_norm, TPSA_norm, LogP_norm) - 全部处理为 [0, 1] 之间，越大越好。
    """
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

        # 1-4. 原有 4 维指标 (SA 反转归一化)
        jnk3_val = float(jnk3_list[i])
        gsk3_val = float(gsk3_list[i])
        qed_val = float(qed_list[i])
        sa_norm = max(0.0, (10.0 - float(sa_list[i])) / 9.0)

        # 5. TPSA 归一化 (钟形曲线，目标 60)
        try:
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            tpsa_norm = math.exp(-0.5 * ((tpsa - 60.0) / 30.0) ** 2)
        except:
            tpsa_norm = 0.0

        # 6. LogP 归一化 (钟形曲线，目标 3.0)
        try:
            logp = Crippen.MolLogP(mol)
            logp_norm = math.exp(-0.5 * ((logp - 3.0) / 1.5) ** 2)
        except:
            logp_norm = 0.0

        scores_6d.append([jnk3_val, gsk3_val, qed_val, sa_norm, tpsa_norm, logp_norm])

    return scores_6d


# ================= 🚀 核心升级 2：NSGA-III 高维淘汰法则 =================
# 👇 导师新增：引入 pymoo 基础 Problem 类
from pymoo.core.problem import Problem

def nsga3_environmental_selection(combined_smiles, combined_scores, num_select=100):
    """
    输入：父代 + 子代 (2N 个分子)
    输出：NSGA-III 严格筛选出的下一代精英 (N 个分子)
    """
    # 1. pymoo 默认求极小值，我们的分数是越大越好，所以乘 -1
    F = -np.array(combined_scores)
    n_obj = 6

    # 2. 部署 6 维空间的“灯塔” (Reference Points)
    # ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=3)
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=4)
    # 3. 构造种群 (完美适配 pymoo 0.6.x)
    indices = np.arange(len(combined_smiles))
    pop = Population.new(X=indices, F=F)

    # ------------------ 👇 导师修复核心 👇 ------------------
    # 临时伪造一个 6 目标的空 Problem，0 个约束条件，完美骗过新版 API 的严格审查！
    dummy_problem = Problem(n_var=1, n_obj=n_obj, n_ieq_constr=0)

    # 4. 执行 NSGA-III 降维打击与小生境保留 (把 None 换成了 dummy_problem)
    survival = ReferenceDirectionSurvival(ref_dirs)
    survivors = survival.do(dummy_problem, pop, n_survive=num_select)
    # ------------------ 👆 导师修复核心 👆 ------------------

    survivor_indices = [ind.get("X") for ind in survivors]

    selected_smiles = [combined_smiles[i] for i in survivor_indices]
    selected_scores = [combined_scores[i] for i in survivor_indices]

    return selected_smiles, selected_scores


def tournament_selection(pool_with_scores, k=2):
    competitors = random.sample(pool_with_scores, k)
    competitors.sort(key=lambda x: x[1], reverse=True)
    return competitors[0][0]


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

    score_list, new_child, all_population, all_population_score = get_synthesis_molecules(parent_population)

    for index, mol_str in enumerate(new_child):
        mol = Chem.MolFromSmiles(mol_str)
        if mol != None:
            mol_new_child = mu.mutate(mol, mutation_rate)
            if mol_new_child != None:
                new_population.append(Chem.MolToSmiles(mol_new_child))

    # 注意：在 NSGA-III 架构下，reproduce 只负责生出变异的子代 (offspring)
    # 子代的打分和淘汰交由外层的主循环处理
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

# 修改为 6 目标的输出文件夹
# out_dir_top3 = os.path.join(base_path, 'output', 'jnk_gsk_nsga3_6d_top')
# out_dir_all = os.path.join(base_path, 'output', 'jnk_gsk_nsga3_6d_all')
out_dir_top3 = os.path.join(base_path, 'output', 'now_jnk_gsk_nsga3_6d_top')
out_dir_all = os.path.join(base_path, 'output', 'now_jnk_gsk_nsga3_6d_all')
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
            # 将多维分数求和作为交配选拔的“参考总分”
            sum_scores = [sum(scores) for scores in parent_scores_6d]
            mating_pool_with_scores = list(zip(population, sum_scores))

            # 1. 繁衍出新一代婴儿 (Offspring)
            offspring_smiles = reproduce(mating_pool_with_scores, population_size, mutation_rate)

            # 2. 给新生的婴儿进行 6 维打分
            print(f"🧬 [Age {age}] 正在对新出生的 {len(offspring_smiles)} 名婴儿进行 6 维体检...")
            offspring_scores_6d = calculate_6d_scores(offspring_smiles)

            # ========================================================
            # 🏆 核心：NSGA-III 父子同台竞技 (mu + lambda 策略) 🏆
            # 把 100 个父母和生出来的婴儿放在一起，组成 2N 的大池子
            # ========================================================
            combined_smiles = population + offspring_smiles
            combined_scores = parent_scores_6d + offspring_scores_6d

            # 3. 召唤 NSGA-III 裁判，通过 6 维参考点筛选出最强且分布最广的 100 人，成为下一代！
            population, parent_scores_6d = nsga3_environmental_selection(combined_smiles, combined_scores,
                                                                         population_size)

            # 4. 计算保存用的参考总分
            score_list = [sum(scores) for scores in parent_scores_6d]

            # (如果需要保存 combined 混合池，在这里操作)
            combined_sum_scores = [sum(scores) for scores in combined_scores]

            # 保存文件
            every_pop = pd.concat([pd.DataFrame(population), pd.DataFrame(score_list)], axis=1)
            every_pop.to_csv(os.path.join(out_dir_top3, f"{i}_{age}_every_top.csv"), index=False, header=False,
                             mode='w')

            all_pop = pd.concat([pd.DataFrame(combined_smiles), pd.DataFrame(combined_sum_scores)], axis=1)
            all_pop.to_csv(os.path.join(out_dir_all, f"{i}_{age}_every_all.csv"), index=False, header=False, mode='a')

            age += 1
            print(f'✅ Generation {age} done. 最强王者已就位！')

    except Exception as e:
        import traceback

        traceback.print_exc()
        continue

t1 = time.time()
print(f'\nTotal time: {t1 - t0:.2f} seconds')