# import glob
# import pandas as pd
# import numpy as np
# from rdkit import Chem
# from rdkit.Chem import AllChem, DataStructs
# from rdkit.Chem.Scaffolds import MurckoScaffold
# import sys
# import os
# import random
# import warnings
#
# warnings.filterwarnings("ignore")
# from rdkit import RDLogger
# RDLogger.DisableLog('rdApp.*')
#
# # ================= 1. 路径设置 =================
# base_path = "/home/liuyansong/ChemistGA-master/ChemistGA-master"
#
# sys.path.insert(0, base_path)
# sys.path.insert(0, os.path.join(base_path, "scoring"))
#
# # 👇 核心修改 1：打分模块路径
# # ⚠️ 注意：如果你运行报错找不到模块，请把下面的 "high_score_drd2" 和 "high_score_properties_drd2"
# # 替换为你刚才在终端里查到的真实文件夹名和文件名！(比如可能叫 "drd2" 和 "high_score_properties")
# sys.path.insert(0, os.path.join(base_path, "ChemistGA", "high_score", "high_score_drd"))
# from high_score_properties_drd2 import get_scoring_function
#
# # 👇 核心修改 2：精准指向【DRD 离散型】的全量大池子！
# output_dir = os.path.join(base_path, "output", "drd_all_pop_discrete")
# known_actives_file = os.path.join(base_path, "data/inh/drd_succ_250.csv") # 如果你的老祖宗叫 drd.csv 请自行修改
#
# # ================= 2. 提前召唤裁判与老祖宗 (节省海量时间) =================
# print("⏳ 正在读取 DRD 单靶点种子库 (Radius=3)...")
# known_df = pd.read_csv(known_actives_file, header=None)
# true_mols = [Chem.MolFromSmiles(s) for s in known_df[0].dropna().tolist()]
# true_mols = [m for m in true_mols if m is not None]
# true_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=2048) for m in true_mols]
#
# print("⏳ 正在一次性加载 3 维严苛裁判 (DRD2, QED, SA)...请稍候...")
# # ⚠️ 注意：同上，"high_score_drd2" 如果报错请改成你真实的文件夹名
# scorer_dir = os.path.join(base_path, "ChemistGA", "high_score", "high_score_drd")
# os.chdir(scorer_dir)
#
# # 👇 核心修改 3：裁判换成 drd2
# drd2_scorer = get_scoring_function('drd2')
# qed_scorer = get_scoring_function('qed')
# sa_scorer = get_scoring_function('sa')
#
# # 用于记录 3 次实验的独立指标，方便最后算均值和方差
# nov_list, div_list, scaf_list = [], [], []
#
# # ================= 3. 独立循环 3 次 Seed，完全隔绝处理 =================
# for seed in range(3):
#     print(f"\n" + "█" * 60)
#     print(f"🚀 正在全盘接管并独立处理实验: Seed {seed} (DRD 离散型 Discrete) 🚀")
#     print("█" * 60)
#
#     # 🌟 步骤 A: 只读取当前 Seed 的所有历史生成文件
#     all_files = glob.glob(f"{output_dir}/{seed}_*_every_all.csv")
#     if not all_files:
#         print(f"⚠️ 找不到 Seed {seed} 的 _all 文件，请检查路径！")
#         continue
#
#     df_list = []
#     for f in all_files:
#         try:
#             df = pd.read_csv(f, header=None)
#             df_list.append(df)
#         except:
#             pass
#
#     total_df = pd.concat(df_list, axis=0, ignore_index=True)
#     total_df.columns = ['SMILES', 'Score']
#
#     # 🌟 步骤 B: 在当前 Seed 内部进行严格去重
#     unique_df = total_df.drop_duplicates(subset=['SMILES']).dropna(subset=['SMILES'])
#     print(f"   📊 Seed {seed} 原始生成唯一分子总数: {len(unique_df)}")
#
#     # 🌟 步骤 C: 粗筛减负 (单靶点总分较低，门槛降为 Score >= 1.0)
#     candidate_smiles = unique_df[unique_df['Score'] >= 1.0]['SMILES'].tolist()
#     print(f"   ⏳ 正在对 {len(candidate_smiles)} 个候选分子进行 3 维重打分...")
#
#     drd2_scores = drd2_scorer(candidate_smiles)
#     qed_scores = qed_scorer(candidate_smiles)
#     sa_scores = sa_scorer(candidate_smiles)
#
#     # 🌟 步骤 D: 3 维极度严苛过滤 (DRD2>=0.5, QED>0.6, SA<4.0)
#     pred_mols = []
#     valid_smiles = []
#     for i in range(len(candidate_smiles)):
#         if float(drd2_scores[i]) >= 0.5 and float(qed_scores[i]) > 0.6 and float(sa_scores[i]) < 4.0:
#             mol = Chem.MolFromSmiles(candidate_smiles[i])
#             if mol is not None:
#                 pred_mols.append(mol)
#                 valid_smiles.append(candidate_smiles[i])
#
#     print(f"   ✅ Seed {seed} 最终幸存【DRD三修全能分子】数: {len(pred_mols)}")
#
#     if len(pred_mols) == 0:
#         print(f"   ❌ Seed {seed} 全军覆没！跳过抽取。")
#         continue
#
#     # 🌟 步骤 E: 计算全局指标
#     pred_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=2048) for m in pred_mols]
#
#     fraction_similar = 0
#     for fp in pred_fps:
#         sims = DataStructs.BulkTanimotoSimilarity(fp, true_fps)
#         if max(sims) >= 0.4:
#             fraction_similar += 1
#     nov = (1.0 - fraction_similar / len(pred_fps)) * 100
#
#     similarity = 0.0
#     for i in range(len(pred_fps)):
#         sims = DataStructs.BulkTanimotoSimilarity(pred_fps[i], pred_fps[:i])
#         similarity += sum(sims)
#     n = len(pred_fps)
#     n_pairs = n * (n - 1) / 2
#     div = 1.0 - (similarity / n_pairs) if n_pairs > 0 else 0.0
#
#     scaffolds = set()
#     for m in pred_mols:
#         try:
#             scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
#             if scaffold: scaffolds.add(scaffold)
#         except:
#             pass
#     scaf_count = len(scaffolds)
#
#     print(f"   🎯 Seed {seed} 全局发现指标: Novelty={nov:.1f}%, Diversity={div:.3f}, Scaffolds={scaf_count}")
#
#     # =========================================================================
#     # ✨ 导师级绝招：家族人口限额法 (Cluster Capping - Max 5) 强势回归！
#     # =========================================================================
#     print(f"   ⏳ 正在执行 '家族人口限额' (Max 5 per scaffold) 以打破种群垄断...")
#     scaffold_dict = {}
#
#     for sm, m in zip(valid_smiles, pred_mols):
#         try:
#             scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
#             if scaf:
#                 if scaf not in scaffold_dict:
#                     scaffold_dict[scaf] = []
#                 scaffold_dict[scaf].append(sm)
#         except:
#             pass
#
#     capped_pool = []
#     MAX_PER_SCAFFOLD = 5
#     for scaf, mols in scaffold_dict.items():
#         selected = random.sample(mols, min(MAX_PER_SCAFFOLD, len(mols)))
#         capped_pool.extend(selected)
#
#     print(f"   ✅ 截断完成！为了保证公平抽样，候选池分子数从 {len(valid_smiles)} 缩减至更均衡的 {len(capped_pool)} 个。")
#     # =========================================================================
#
#     # 🌟 步骤 F: 抽取 5000 个分子用于 Retro*
#     print(f"   📦 正在为 Seed {seed} 独立抽取 DRD 离散型 Retro* 测试集...")
#     sample_size = min(5000, len(capped_pool))
#     sampled_smiles = random.sample(capped_pool, sample_size)
#
#     # ✨ 动态命名，加上 drd_discrete_ 前缀
#     output_smi_file = os.path.join(base_path, "output", f"drd_discrete_for_retro_test_5000_seed{seed}.smi")
#
#     with open(output_smi_file, 'w') as f:
#         for smi in sampled_smiles:
#             f.write(smi + '\n')
#
#     print(f"   ✅ Seed {seed} 成功抽取 {sample_size} 个均衡多样的极品分子！")
#     print(f"   📁 已保存专属文件: {output_smi_file}")
#
# # ================= 4. 最终总结报告 =================
# if len(nov_list) > 0:
#     print("\n" + "★" * 55)
#     print("🏆 DRD 单靶点离散型 (Discrete All Pop) 评估 🏆")
#     print("★" * 55)
#     print(f"📌 平均全局 Novelty (%)  : {np.mean(nov_list):.1f} ± {np.std(nov_list):.1f}")
#     print(f"📌 平均全局 Diversity    : {np.mean(div_list):.3f} ± {np.std(div_list):.3f}")
#     print(f"📌 平均全局 Scaffolds    : {np.mean(scaf_list):.1f} ± {np.std(scaf_list):.1f}")
#     print("★" * 55)
#     print("\n🎉 DRD 离散型任务圆满完成！请提取 'drd_discrete_for_retro' 喂给 Retro*！")
import glob
import os
import random
import sys
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


# ==================== 1. Configuration ====================

BASE_PATH = "/home/liuyansong/ChemistGA-master/ChemistGA-master"

OUTPUT_DIR = os.path.join(
    BASE_PATH,
    "output",
    "drd_all_pop_discrete",
)

REFERENCE_FILE = os.path.join(
    BASE_PATH,
    "data",
    "inh",
    "drd_succ_250.csv",
)

SCORER_DIR = os.path.join(
    BASE_PATH,
    "ChemistGA",
    "high_score",
    "high_score_drd",
)

SAMPLE_SIZE = 5000
FP_RADIUS = 3
FP_BITS = 2048
NOVELTY_THRESHOLD = 0.4

sys.path.insert(0, BASE_PATH)
sys.path.insert(
    0,
    os.path.join(BASE_PATH, "scoring"),
)
sys.path.insert(0, SCORER_DIR)

from high_score_properties_drd2 import get_scoring_function


# ==================== 2. Reference molecules ====================

print("Loading DRD2 reference molecules...")

reference_df = pd.read_csv(
    REFERENCE_FILE,
    header=None,
)

reference_molecules = []

for smiles in reference_df.iloc[:, 0].dropna():
    mol = Chem.MolFromSmiles(
        str(smiles).strip()
    )

    if mol is not None:
        reference_molecules.append(mol)

reference_fps = [
    AllChem.GetMorganFingerprintAsBitVect(
        mol,
        FP_RADIUS,
        nBits=FP_BITS,
    )
    for mol in reference_molecules
]

if not reference_fps:
    raise RuntimeError(
        "No valid DRD2 reference molecules were loaded."
    )

print(
    f"Loaded {len(reference_fps)} DRD2 reference molecules."
)


# ==================== 3. Scoring models ====================

print("Loading DRD2, QED and SA scoring models...")

os.chdir(SCORER_DIR)

drd2_scorer = get_scoring_function("drd2")
qed_scorer = get_scoring_function("qed")
sa_scorer = get_scoring_function("sa")


# ==================== 4. Metric functions ====================

def calculate_novelty(predicted_fps):
    novel_count = 0

    for fp in predicted_fps:
        similarities = (
            DataStructs.BulkTanimotoSimilarity(
                fp,
                reference_fps,
            )
        )

        if max(similarities) < NOVELTY_THRESHOLD:
            novel_count += 1

    return (
        100.0
        * novel_count
        / len(predicted_fps)
    )


def calculate_diversity(predicted_fps):
    molecule_count = len(predicted_fps)

    if molecule_count < 2:
        return 0.0

    similarity_sum = 0.0

    for index, fp in enumerate(predicted_fps):
        similarities = (
            DataStructs.BulkTanimotoSimilarity(
                fp,
                predicted_fps[:index],
            )
        )

        similarity_sum += sum(similarities)

    pair_count = (
        molecule_count
        * (molecule_count - 1)
        / 2
    )

    return (
        1.0
        - similarity_sum / pair_count
    )


def count_scaffolds(molecules):
    scaffolds = set()

    for mol in molecules:
        try:
            scaffold = (
                MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol,
                    includeChirality=False,
                )
            )

            if scaffold:
                scaffolds.add(scaffold)

        except Exception:
            continue

    return len(scaffolds)


# ==================== 5. Process three seeds ====================

novelty_results = []
diversity_results = []
scaffold_results = []

for seed in range(3):
    print("\n" + "=" * 72)
    print(
        f"DRD2 three-objective discrete task | seed {seed}"
    )
    print("=" * 72)

    # Reproducible random sampling for each seed
    rng = random.Random(seed)

    all_files = sorted(
        glob.glob(
            os.path.join(
                OUTPUT_DIR,
                f"{seed}_*_every_all.csv",
            )
        )
    )

    if not all_files:
        print(
            f"No population files found for seed {seed}."
        )
        continue

    dataframes = []

    for file_path in all_files:
        try:
            frame = pd.read_csv(
                file_path,
                header=None,
            )

            if frame.shape[1] < 2:
                print(
                    f"Skipping incompatible file: {file_path}"
                )
                continue

            frame = frame.iloc[:, :2]
            frame.columns = [
                "SMILES",
                "Score",
            ]

            dataframes.append(frame)

        except Exception as exc:
            print(
                f"Skipping {file_path}: {exc}"
            )

    if not dataframes:
        print(
            f"No valid population data for seed {seed}."
        )
        continue

    total_df = pd.concat(
        dataframes,
        axis=0,
        ignore_index=True,
    )

    total_df = total_df.dropna(
        subset=["SMILES"]
    )

    total_df["SMILES"] = (
        total_df["SMILES"]
        .astype(str)
        .str.strip()
    )

    total_df["Score"] = pd.to_numeric(
        total_df["Score"],
        errors="coerce",
    )

    total_df = total_df.dropna(
        subset=["Score"]
    )

    # Follow the continuous script: remove repeated SMILES strings.
    unique_df = total_df.drop_duplicates(
        subset=["SMILES"],
        keep="first",
    )

    print(
        f"Canonical input entries after SMILES-string "
        f"deduplication: {len(unique_df)}"
    )

    # Coarse prefilter used only to reduce scoring cost.
    candidate_smiles = unique_df.loc[
        unique_df["Score"] >= 1.0,
        "SMILES",
    ].tolist()

    print(
        f"Candidates entering exact rescoring: "
        f"{len(candidate_smiles)}"
    )

    if not candidate_smiles:
        print(
            f"No candidates available for seed {seed}."
        )
        continue

    drd2_scores = drd2_scorer(
        candidate_smiles
    )

    qed_scores = qed_scorer(
        candidate_smiles
    )

    sa_scores = sa_scorer(
        candidate_smiles
    )

    valid_smiles = []
    valid_molecules = []

    for index, smiles in enumerate(
        candidate_smiles
    ):
        drd2_value = float(
            drd2_scores[index]
        )

        qed_value = float(
            qed_scores[index]
        )

        sa_value = float(
            sa_scores[index]
        )

        # Inclusive thresholds consistent with the task definition.
        if (
            drd2_value >= 0.5
            and qed_value >= 0.6
            and sa_value <= 4.0
        ):
            mol = Chem.MolFromSmiles(
                smiles
            )

            if mol is not None:
                valid_smiles.append(
                    smiles
                )
                valid_molecules.append(
                    mol
                )

    print(
        f"Molecules satisfying all three criteria: "
        f"{len(valid_smiles)}"
    )

    if not valid_smiles:
        print(
            f"No success molecules for seed {seed}."
        )
        continue

    # Calculate global statistics on all qualifying molecules.
    predicted_fps = [
        AllChem.GetMorganFingerprintAsBitVect(
            mol,
            FP_RADIUS,
            nBits=FP_BITS,
        )
        for mol in valid_molecules
    ]

    novelty = calculate_novelty(
        predicted_fps
    )

    diversity = calculate_diversity(
        predicted_fps
    )

    scaffold_count = count_scaffolds(
        valid_molecules
    )

    novelty_results.append(
        novelty
    )

    diversity_results.append(
        diversity
    )

    scaffold_results.append(
        scaffold_count
    )

    print(
        f"Full qualifying pool: "
        f"Novelty={novelty:.2f}%, "
        f"Diversity={diversity:.4f}, "
        f"Scaffolds={scaffold_count}"
    )

    # Direct sampling from all qualifying molecules.
    # No scaffold capping and no sampling with replacement.
    if len(valid_smiles) < SAMPLE_SIZE:
        shortage = (
            SAMPLE_SIZE - len(valid_smiles)
        )

        raise RuntimeError(
            f"Seed {seed} has only {len(valid_smiles)} "
            f"unique qualifying molecules, which is "
            f"{shortage} fewer than the required "
            f"{SAMPLE_SIZE}. Do not duplicate molecules to "
            f"artificially reach 5,000. Generate more "
            f"molecules or include additional generations."
        )

    sampled_smiles = rng.sample(
        valid_smiles,
        SAMPLE_SIZE,
    )

    output_smi_file = os.path.join(
        BASE_PATH,
        "output",
        f"drd_discrete_for_retro_test_5000_seed{seed}.smi",
    )

    with open(
        output_smi_file,
        "w",
        encoding="utf-8",
    ) as handle:

        for smiles in sampled_smiles:
            handle.write(
                smiles + "\n"
            )

    print(
        f"Saved exactly {len(sampled_smiles)} "
        f"qualifying molecules."
    )

    print(
        f"Output: {output_smi_file}"
    )


# ==================== 6. Summary ====================

if novelty_results:
    print("\n" + "=" * 72)
    print(
        "DRD2 three-objective discrete task summary"
    )
    print("=" * 72)

    print(
        "Novelty (%): %.2f +/- %.2f"
        % (
            np.mean(novelty_results),
            np.std(novelty_results),
        )
    )

    print(
        "Diversity: %.4f +/- %.4f"
        % (
            np.mean(diversity_results),
            np.std(diversity_results),
        )
    )

    print(
        "Scaffolds: %.1f +/- %.1f"
        % (
            np.mean(scaffold_results),
            np.std(scaffold_results),
        )
    )


