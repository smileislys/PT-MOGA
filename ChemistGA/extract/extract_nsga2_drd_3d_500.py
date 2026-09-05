import os
import random

# ================= 1. 路径设置 =================
base_path = "/home/liuyansong/ChemistGA-master/ChemistGA-master"
output_dir = os.path.join(base_path, "output")

# ================= 2. 独立循环 3 次 Seed =================
for seed in range(3):
    print(f"\n" + "█" * 60)
    print(f"🚀 正在极速降维抽样: Seed {seed} (从 5000 强抽 500) 🚀")
    print("█" * 60)

    # 👇 你的源文件：已经完全及格的 5000 个分子
    source_smi_file = os.path.join(output_dir, f"drd_continuous_for_retro_test_5000_seed{seed}.smi")

    if not os.path.exists(source_smi_file):
        print(f"⚠️ 找不到源文件: {source_smi_file}，请检查文件是否存在！")
        continue

    # 读取所有分子
    with open(source_smi_file, 'r') as f:
        # 去除空行和换行符
        valid_smiles = [line.strip() for line in f.readlines() if line.strip()]

    print(f"   📊 成功从源文件读取及格分子总数: {len(valid_smiles)}")

    if len(valid_smiles) == 0:
        continue

    # 👇 核心：严格截断 500 个
    sample_size = min(500, len(valid_smiles))
    sampled_smiles = random.sample(valid_smiles, sample_size)

    # 👇 你的目标文件：用于公平对比的 500 样本基准文件
    target_smi_file = os.path.join(output_dir, f"nsga2_drd_3d_for_retro_test_500_seed{seed}.smi")

    with open(target_smi_file, 'w') as f:
        for smi in sampled_smiles:
            f.write(smi + '\n')

    print(f"   ✅ Seed {seed} 极速抽样完成！成功保存 {sample_size} 个极品分子！")
    print(f"   📁 新文件已落库: {target_smi_file}")

print("\n🎉 全部降采样完成！现在你可以拿这 3 个 nsga2_drd_3d_for_retro_test_500_seedX.smi 文件去跑评测和 Retro* 了！")