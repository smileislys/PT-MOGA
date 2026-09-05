import glob
import pandas as pd
import sys
import os
import random
import warnings
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Crippen

warnings.filterwarnings("ignore")
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# ================= 1. 路径设置 =================
base_path = "/home/liuyansong/ChemistGA-master/ChemistGA-master"
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, "scoring"))
sys.path.insert(0, os.path.join(base_path, "ChemistGA", "high_score", "high_score_jnk_gsk"))

from high_score_properties_jnk_gsk import get_scoring_function

# 👇 核心：精准指向 NSGA-II 6维混合大池子！
output_dir = os.path.join(base_path, "output", "jnk_gsk_nsga2_6d_all")

print("⏳ 正在加载 6 维模型裁判 (JNK3, GSK3, QED, SA) 以及 RDKit 组件...")
scorer_dir = os.path.join(base_path, "ChemistGA", "high_score", "high_score_jnk_gsk")
os.chdir(scorer_dir)

jnk3_scorer = get_scoring_function('jnk3')
gsk3_scorer = get_scoring_function('gsk3')
qed_scorer = get_scoring_function('qed')
sa_scorer = get_scoring_function('sa')

# ================= 2. 独立循环 3 次 Seed =================
for seed in range(3):
    print(f"\n" + "█" * 60)
    print(f"🚀 正在提纯: Seed {seed} (NSGA-II 6目标 - 抽取 500 个) 🚀")
    print("█" * 60)

    all_files = glob.glob(f"{output_dir}/{seed}_*_every_all.csv")
    if not all_files:
        print(f"⚠️ 找不到 Seed {seed} 的 _all 文件，请确认 NSGA-II 6D 任务是否跑完。")
        continue

    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f, header=None)
            df_list.append(df)
        except:
            pass

    total_df = pd.concat(df_list, axis=0, ignore_index=True)
    total_df.columns = ['SMILES', 'Score']

    # 严格去重
    unique_df = total_df.drop_duplicates(subset=['SMILES']).dropna(subset=['SMILES'])
    print(f"   📊 Seed {seed} 原始生成唯一分子总数: {len(unique_df)}")

    # 6 维任务粗筛门槛设为 2.5
    candidate_smiles = unique_df[unique_df['Score'] >= 2.5]['SMILES'].tolist()
    print(f"   ⏳ 正在对 {len(candidate_smiles)} 个候选分子进行 6 维严苛打分...")

    jnk3_scores = jnk3_scorer(candidate_smiles)
    gsk3_scores = gsk3_scorer(candidate_smiles)
    qed_scores = qed_scorer(candidate_smiles)
    sa_scores = sa_scorer(candidate_smiles)

    valid_smiles = []

    # 👇 6 维严苛药企级过滤！
    for i in range(len(candidate_smiles)):
        smi = candidate_smiles[i]
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue

        j_score = float(jnk3_scores[i])
        g_score = float(gsk3_scores[i])
        q_score = float(qed_scores[i])
        s_score = float(sa_scores[i])

        try:
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            logp = Crippen.MolLogP(mol)
        except:
            continue

        if (j_score >= 0.5 and g_score >= 0.5 and
                q_score > 0.6 and s_score <= 4.0 and
                tpsa <= 90.0 and
                1.0 <= logp <= 5.0):
            valid_smiles.append(smi)

    print(f"   ✅ Seed {seed} 最终幸存【6维及格分子】数: {len(valid_smiles)}")

    if len(valid_smiles) == 0:
        continue

    # 强制抽取 500 个作为基准测试集
    sample_size = min(500, len(valid_smiles))
    sampled_smiles = random.sample(valid_smiles, sample_size)

    # 👇 保存为 NSGA-II 6D 的专属 .smi 文件
    output_smi_file = os.path.join(base_path, "output", f"nsga2_6d_for_retro_test_500_seed{seed}.smi")
    with open(output_smi_file, 'w') as f:
        for smi in sampled_smiles:
            f.write(smi + '\n')

    print(f"   ✅ Seed {seed} 成功抽取并保存 {sample_size} 个极品分子！")
    print(f"   📁 文件已落库: {output_smi_file}")