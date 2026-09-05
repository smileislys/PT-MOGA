import glob
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdMolDescriptors, Crippen
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
sys.path.insert(0, os.path.join(base_path, "ChemistGA", "high_score", "high_score_jnk_gsk"))

from high_score_properties_jnk_gsk import get_scoring_function

# 👇 🚨 核心修改 1：精准指向【GB-GA 消融实验】的 ALL 混合大池子！
output_dir = os.path.join(base_path, "output", "ablation_gbga_nsga3_6d_all")
known_actives_file = os.path.join(base_path, "data/inh/jnk_gsk.csv")

# ================= 2. 提前召唤裁判与老祖宗 =================
print("⏳ 正在读取 双靶点种子库 (Radius=3)...")
known_df = pd.read_csv(known_actives_file, header=None)
true_mols = [Chem.MolFromSmiles(s) for s in known_df[0].dropna().tolist()]
true_mols = [m for m in true_mols if m is not None]
true_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=2048) for m in true_mols]

print("⏳ 正在一次性加载 4 维模型裁判 (JNK3, GSK3, QED, SA)...请稍候...")
scorer_dir = os.path.join(base_path, "ChemistGA", "high_score", "high_score_jnk_gsk")
os.chdir(scorer_dir)

jnk3_scorer = get_scoring_function('jnk3')
gsk3_scorer = get_scoring_function('gsk3')
qed_scorer = get_scoring_function('qed')
sa_scorer = get_scoring_function('sa')

# 用于记录 3 次实验的独立指标，方便最后算均值和方差
nov_list, div_list, scaf_list = [], [], []

# ================= 3. 独立循环 Seed =================
# 🚨 注意：这里循环 range(3)，但如果你只跑了 Seed 0，它会自动跳过找不到的 Seed 1 和 2
for seed in range(3):
    print(f"\n" + "█" * 65)
    print(f"🚀 正在处理 [消融实验] GB-GA 传统交叉: Seed {seed} (6目标 MPO - ALL 混合池) 🚀")
    print("█" * 65)

    all_files = glob.glob(f"{output_dir}/{seed}_*_every_all.csv")
    if not all_files:
        print(f"⚠️ 找不到 Seed {seed} 的 _all 文件，自动跳过该 Seed。")
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
    print(f"   📊 Seed {seed} 原始生成唯一【所有分子】总数: {len(unique_df)}")

    # 🌟 步骤 C: 粗筛减负
    candidate_smiles = unique_df[unique_df['Score'] >= 2.5]['SMILES'].tolist()
    print(f"   ⏳ 正在对 {len(candidate_smiles)} 个初步及格分子进行 6 维绝对值硬性重打分...")

    jnk3_scores = jnk3_scorer(candidate_smiles)
    gsk3_scores = gsk3_scorer(candidate_smiles)
    qed_scores = qed_scorer(candidate_smiles)
    sa_scores = sa_scorer(candidate_smiles)

    # 🌟 步骤 D: 6 维极度严苛过滤 (斩杀不及格废料)
    pred_mols = []
    valid_smiles = []

    for i in range(len(candidate_smiles)):
        smi = candidate_smiles[i]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        j_score = float(jnk3_scores[i])
        g_score = float(gsk3_scores[i])
        q_score = float(qed_scores[i])
        s_score = float(sa_scores[i])

        try:
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            logp = Crippen.MolLogP(mol)
        except:
            continue

        # 药企级 6 维绝对值硬筛选！
        if (j_score >= 0.5 and g_score >= 0.5 and
                q_score > 0.6 and s_score <= 4.0 and
                tpsa <= 90.0 and
                1.0 <= logp <= 5.0):
            pred_mols.append(mol)
            valid_smiles.append(smi)

    print(f"   ✅ Seed {seed} 从 ALL 池中最终淘金获得的【符合 6 维硬指标】数: {len(pred_mols)}")

    if len(pred_mols) == 0:
        print(f"   ❌ Seed {seed} 全军覆没！(这是预料之中的，传统GA在高维极难存活)")
        continue

    # 🌟 步骤 E: 仅针对当前提取的 500 个分子计算当次 Seed 的三大指标
    print(f"   📦 正在为 Seed {seed} 随机抽取【最多500】个合规分子...")
    sample_size = min(500, len(valid_smiles))
    sampled_smiles = random.sample(valid_smiles, sample_size)

    sampled_mols = [Chem.MolFromSmiles(smi) for smi in sampled_smiles]
    sampled_mols = [m for m in sampled_mols if m is not None]

    pred_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=2048) for m in sampled_mols]

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
    for m in sampled_mols:
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
            if scaffold: scaffolds.add(scaffold)
        except:
            pass
    scaf_count = len(scaffolds)

    nov_list.append(nov)
    div_list.append(div)
    scaf_list.append(scaf_count)

    print(f"   🎯 Seed {seed} GB-GA消融 战报: Novelty={nov:.1f}%, Diversity={div:.3f}, Scaffolds={scaf_count}")

    # 🌟 步骤 F: 保存这批分子！
    # 👇 🚨 核心修改 2：保存的命名专门带有 ablation_gbga 标签！
    output_smi_file = os.path.join(base_path, "output", f"ablation_gbga_6d_mpo_for_retro_test_500_seed{seed}.smi")

    with open(output_smi_file, 'w') as f:
        for smi in sampled_smiles:
            f.write(smi + '\n')

    print(f"   ✅ Seed {seed} 成功保存 {sample_size} 个消融测试分子！")
    print(f"   📁 已保存专属文件: {output_smi_file}")

# ================= 4. 最终总结报告 =================
if len(nov_list) > 0:
    print("\n" + "★" * 65)
    print("🏆 [消融实验] 传统 GB-GA 交叉 (500抽样集) 最终评估 🏆")
    print("★" * 65)
    if len(nov_list) > 1:
        print(f"📌 平均 Novelty (%)  : {np.mean(nov_list):.1f} ± {np.std(nov_list):.1f}")
        print(f"📌 平均 Diversity    : {np.mean(div_list):.3f} ± {np.std(div_list):.3f}")
        print(f"📌 平均 Scaffolds    : {np.mean(scaf_list):.1f} ± {np.std(scaf_list):.1f}")
    else:
        print(f"📌 Novelty (%)  : {nov_list[0]:.1f}")
        print(f"📌 Diversity    : {div_list[0]:.3f}")
        print(f"📌 Scaffolds    : {scaf_list[0]:.1f}")
    print("★" * 65)
    print("\n🎉 消融集保存完毕！赶快拿这些分子去送给 Retro* 跑合成率，坐等它暴死吧！")
else:
    print("\n⚠️ 极其惨烈！GB-GA 跑出来的分子在 6 维硬指标过滤下【全军覆没】！连送给 Retro* 安检的资格都没有！")