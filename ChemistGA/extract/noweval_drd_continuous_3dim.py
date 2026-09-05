import glob
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
import sys
import os
import random
import warnings

warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# ================= 1. 路径设置 =================
base_path = "/home/liuyansong/ChemistGA-master/ChemistGA-master"

sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, "scoring"))

# 👇 核心修改 1：导入 DRD2 打分模块 (假设你的文件夹叫 high_score_drd2)
sys.path.insert(0, os.path.join(base_path, "ChemistGA", "high_score", "high_score_drd"))
from high_score_properties_drd2 import get_scoring_function

# 👇 核心修改 2：精准指向【DRD 连续型】的全量大池子！
output_dir = os.path.join(base_path, "output", "drd_all_pop_continuous")
# 假设你的已知活性分子文件叫 drd2.csv (如果是 drd.csv 请自行修改)
known_actives_file = os.path.join(base_path, "data/inh/drd_succ_250.csv")

# ================= 2. 提前召唤裁判与老祖宗 (节省海量时间) =================
print("⏳ 正在读取 DRD 单靶点种子库 (Radius=3)...")
known_df = pd.read_csv(known_actives_file, header=None)
true_mols = [Chem.MolFromSmiles(s) for s in known_df[0].dropna().tolist()]
true_mols = [m for m in true_mols if m is not None]
true_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=2048) for m in true_mols]

print("⏳ 正在一次性加载 3 维严苛裁判 (DRD2, QED, SA)...请稍候...")
scorer_dir = os.path.join(base_path, "ChemistGA", "high_score", "high_score_drd")
os.chdir(scorer_dir)

# 👇 核心修改 3：裁判换成 DRD2
drd2_scorer = get_scoring_function('drd2')
qed_scorer = get_scoring_function('qed')
sa_scorer = get_scoring_function('sa')

# 用于记录 3 次实验的独立指标，方便最后算均值和方差
nov_list, div_list, scaf_list = [], [], []

# ================= 3. 独立循环 3 次 Seed，完全隔绝处理 =================
for seed in range(3):
    print(f"\n" + "█" * 60)
    print(f"🚀 正在全盘接管并独立处理实验: Seed {seed} (DRD 连续型 Continuous) 🚀")
    print("█" * 60)

    # 🌟 步骤 A: 只读取当前 Seed 的所有历史生成文件
    all_files = glob.glob(f"{output_dir}/{seed}_*_every_all.csv")
    if not all_files:
        print(f"⚠️ 找不到 Seed {seed} 的 _all 文件，请检查路径！")
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

    # 🌟 步骤 B: 在当前 Seed 内部进行严格去重
    unique_df = total_df.drop_duplicates(subset=['SMILES']).dropna(subset=['SMILES'])
    print(f"   📊 Seed {seed} 原始生成唯一分子总数: {len(unique_df)}")

    # 🌟 步骤 C: 粗筛减负 (因为是三修单靶点，门槛总分降为 Score >= 1.0)
    candidate_smiles = unique_df[unique_df['Score'] >= 1.0]['SMILES'].tolist()
    print(f"   ⏳ 正在对 {len(candidate_smiles)} 个候选分子进行 3 维重打分...")

    drd2_scores = drd2_scorer(candidate_smiles)
    qed_scores = qed_scorer(candidate_smiles)
    sa_scores = sa_scorer(candidate_smiles)

    # 🌟 步骤 D: 3 维极度严苛过滤 (斩杀不及格废料)
    pred_mols = []
    valid_smiles = []
    for i in range(len(candidate_smiles)):
        # 👇 核心修改 4：过滤条件变为 3 个 (DRD2>=0.5, QED>0.6, SA<4.0)
        if float(drd2_scores[i]) >= 0.5 and float(qed_scores[i]) > 0.6 and float(sa_scores[i]) < 4.0:
            mol = Chem.MolFromSmiles(candidate_smiles[i])
            if mol is not None:
                pred_mols.append(mol)
                valid_smiles.append(candidate_smiles[i])

    print(f"   ✅ Seed {seed} 最终幸存【DRD三修全能分子】数: {len(pred_mols)}")

    if len(pred_mols) == 0:
        print(f"   ❌ Seed {seed} 全军覆没！跳过抽取。")
        continue

    # 🌟 步骤 E: 计算当次 Seed 的三大指标 (用于监控模型真实探索能力)
    pred_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=2048) for m in pred_mols]

    fraction_similar = 0
    for fp in pred_fps:
        sims = DataStructs.BulkTanimotoSimilarity(fp, true_fps)
        if max(sims) >= 0.4:
            fraction_similar += 1
    nov = (1.0 - fraction_similar / len(pred_fps)) * 100

    similarity = 0.0
    for i in range(len(pred_fps)):
        sims = DataStructs.BulkTanimotoSimilarity(pred_fps[i], pred_fps[:i])
        similarity += sum(sims)
    n = len(pred_fps)
    n_pairs = n * (n - 1) / 2
    div = 1.0 - (similarity / n_pairs) if n_pairs > 0 else 0.0

    scaffolds = set()
    for m in pred_mols:
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
            if scaffold: scaffolds.add(scaffold)
        except:
            pass
    scaf_count = len(scaffolds)

    nov_list.append(nov)
    div_list.append(div)
    scaf_list.append(scaf_count)

    print(f"   🎯 Seed {seed} 指标监控: Novelty={nov:.1f}%, Diversity={div:.3f}, Scaffolds={scaf_count}")

    # 🌟 步骤 F: 仅针对当前 Seed 独立抽取 5000 个分子！(连续型我们用自然抽样即可)
    print(f"   📦 正在为 Seed {seed} 独立抽取 DRD 单靶点 Retro* 测试集...")
    sample_size = min(5000, len(valid_smiles))

    sampled_smiles = random.sample(valid_smiles, sample_size)

    # ✨ 核心修改 5：动态命名，加上 drd_continuous_ 前缀防止文件混淆
    output_smi_file = os.path.join(base_path, "output", f"drd_continuous_for_retro_test_5000_seed{seed}.smi")

    with open(output_smi_file, 'w') as f:
        for smi in sampled_smiles:
            f.write(smi + '\n')

    print(f"   ✅ Seed {seed} 成功抽取 {sample_size} 个极品分子！")
    print(f"   📁 已保存专属文件: {output_smi_file}")

# ================= 4. 最终总结报告 =================
if len(nov_list) > 0:
    print("\n" + "★" * 55)
    print("🏆 DRD 单靶点连续型三代独立池 (Continuous) 评估 🏆")
    print("★" * 55)
    print(f"📌 平均 Novelty (%)  : {np.mean(nov_list):.1f} ± {np.std(nov_list):.1f}")
    print(f"📌 平均 Diversity    : {np.mean(div_list):.3f} ± {np.std(div_list):.3f}")
    print(f"📌 平均 Scaffolds    : {np.mean(scaf_list):.1f} ± {np.std(scaf_list):.1f}")
    print("★" * 55)
    print("\n🎉 DRD 连续型任务圆满完成！请提取 'drd_continuous_for_retro' 喂给 Retro*！")