import os
import random

# ================= 1. 路径设置 =================
base_path = "/home/liuyansong/ChemistGA-master/ChemistGA-master"
output_dir = os.path.join(base_path, "output")

# ================= 2. 独立循环 3 次 Seed =================
for seed in range(3):
    print(f"\n" + "█" * 60)
    print(f"🚀 正在极速降维抽样: NSGA-II 4D (从 5000 强抽 500) - Seed {seed} 🚀")
    print("█" * 60)

    # 👇 根据你之前的代码，尝试读取 5000 分子的源文件 (带 test_ 或不带 test_ 都兼容一下)
    source_smi_file = os.path.join(output_dir, f"test_continuous_for_retro_test_5000_seed{seed}.smi")
    if not os.path.exists(source_smi_file):
        # 兼容备用名字
        source_smi_file = os.path.join(output_dir, f"continuous_for_retro_test_5000_seed{seed}.smi")

    if not os.path.exists(source_smi_file):
        print(f"⚠️ 找不到源文件: {source_smi_file}，请检查 output 文件夹里这个文件叫什么名字！")
        continue

    # 读取所有及格分子
    with open(source_smi_file, 'r') as f:
        valid_smiles = [line.strip() for line in f.readlines() if line.strip()]

    print(f"   📊 成功从源文件读取【4维及格分子】总数: {len(valid_smiles)}")

    if len(valid_smiles) == 0:
        continue

    # 👇 核心：严格截断 500 个，对齐 NSGA-III 的样本量！
    sample_size = min(500, len(valid_smiles))
    sampled_smiles = random.sample(valid_smiles, sample_size)

    # 👇 目标文件：统一重命名为 nsga2_4d 标准名字
    target_smi_file = os.path.join(output_dir, f"nsga2_4d_for_retro_test_500_seed{seed}.smi")

    with open(target_smi_file, 'w') as f:
        for smi in sampled_smiles:
            f.write(smi + '\n')

    print(f"   ✅ Seed {seed} 极速抽样完成！成功保存 {sample_size} 个极品分子！")
    print(f"   📁 崭新的基准文件已落库: {target_smi_file}")

print("\n🎉 4目标 (NSGA-II) 全部降采样完成！现在你的 4D 对比实验数据基座已经打造完毕！")