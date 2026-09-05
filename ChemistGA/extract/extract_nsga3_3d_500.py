import glob
import pandas as pd
import sys
import os
import random
import warnings
from rdkit import Chem

warnings.filterwarnings("ignore")
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# ================= 1. 路径设置 =================
base_path = "/home/liuyansong/ChemistGA-master/ChemistGA-master"
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, "scoring"))

# 👇 修正：严格指向 DRD 单靶点打分器
sys.path.insert(0, os.path.join(base_path, "ChemistGA", "high_score", "high_score_drd"))
from high_score_properties_drd2 import get_scoring_function

# 👇 指向 DRD NSGA-III 3维混合大池子
output_dir = os.path.join(base_path, "output", "drd_nsga3_3d_all")

print("⏳ 正在加载 DRD 3维模型裁判 (DRD2, QED, SA)...")
scorer_dir = os.path.join(base_path, "ChemistGA", "high_score", "high_score_drd")
os.chdir(scorer_dir)

# 👇 修正：召唤正确的裁判！
drd2_scorer = get_scoring_function('drd2')
qed_scorer = get_scoring_function('qed')
sa_scorer = get_scoring_function('sa')

# ================= 2. 独立循环 3 次 Seed =================
for seed in range(3):
    print(f"\n" + "█" * 60)
    print(f"🚀 正在提纯: Seed {seed} (DRD单靶点 NSGA-III 3目标 - 抽取 500 个) 🚀")
    print("█" * 60)

    all_files = glob.glob(f"{output_dir}/{seed}_*_every_all.csv")
    if not all_files:
        print(f"⚠️ 找不到 Seed {seed} 的 _all 文件，请确认是否跑完。")
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

    # 粗筛减负 (三修单靶点，门槛总分降为 Score >= 1.0)
    candidate_smiles = unique_df[unique_df['Score'] >= 1.0]['SMILES'].tolist()
    print(f"   ⏳ 正在对 {len(candidate_smiles)} 个候选分子进行 DRD 3维重打分...")

    drd2_scores = drd2_scorer(candidate_smiles)
    qed_scores = qed_scorer(candidate_smiles)
    sa_scores = sa_scorer(candidate_smiles)

    valid_smiles = []

    # 👇 3 维严苛过滤 (DRD2>=0.5, QED>0.6, SA<4.0)
    for i in range(len(candidate_smiles)):
        if float(drd2_scores[i]) >= 0.5 and float(qed_scores[i]) > 0.6 and float(sa_scores[i]) < 4.0:
            mol = Chem.MolFromSmiles(candidate_smiles[i])
            if mol is not None:
                valid_smiles.append(candidate_smiles[i])

    print(f"   ✅ Seed {seed} 最终幸存【DRD 3维及格分子】数: {len(valid_smiles)}")

    if len(valid_smiles) == 0:
        continue

    # 强制抽取 500 个作为基准测试集
    sample_size = min(500, len(valid_smiles))
    sampled_smiles = random.sample(valid_smiles, sample_size)

    # 保存专属命名的 .smi 文件，带上 drd 前缀防混淆
    output_smi_file = os.path.join(base_path, "output", f"drd_nsga3_3d_for_retro_test_500_seed{seed}.smi")
    with open(output_smi_file, 'w') as f:
        for smi in sampled_smiles:
            f.write(smi + '\n')

    print(f"   ✅ Seed {seed} 成功抽取并保存 {sample_size} 个极品分子！")
    print(f"   📁 文件已落库: {output_smi_file}")