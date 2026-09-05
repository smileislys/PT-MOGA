# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import random
import math
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen

# 🚀 替换掉旧的 4维 multi_scoring_functions，引入底层算分器
from high_score_properties_jnk_gsk import get_scoring_function
from transformer_model.onmt.utils.logging import init_logger
from transformer_model.onmt.translate.translator import build_translator
from transformer_model.onmt.opts_translate import OPT_TRANSLATE


def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)

    return ' '.join(tokens)


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def synthesis(opt, src_data_iter):
    # 强制将显卡序号设为 0
    opt.gpu = 0
    torch.cuda.set_device(0)
    # 强行创建 Transformer 需要的临时文件夹
    import os
    os.makedirs(os.path.dirname(opt.output), exist_ok=True)

    translator = build_translator(opt, report_score=True)

    all_scores, all_predictions = translator.translate(src_path=opt.src,
                                                       src_data_iter=src_data_iter,
                                                       tgt_path=opt.tgt,
                                                       src_dir=opt.src_dir,
                                                       batch_size=opt.batch_size,
                                                       attn_debug=opt.attn_debug)
    torch.cuda.empty_cache()

    return all_predictions


# ================= 🚀 新增：产房专属的 6 维加和裁判 =================
def get_local_6d_sum_scores(population):
    jnk3_scorer = get_scoring_function('jnk3')
    gsk3_scorer = get_scoring_function('gsk3')
    qed_scorer = get_scoring_function('qed')
    sa_scorer = get_scoring_function('sa')

    jnk3_list = jnk3_scorer(population)
    gsk3_list = gsk3_scorer(population)
    qed_list = qed_scorer(population)
    sa_list = sa_scorer(population)

    sum_scores = []
    for i in range(len(population)):
        smi = population[i]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            sum_scores.append(-999.0)  # 废料直接负分滚粗
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

        # 将 6 项指标加和作为挑选前 3 名子代的依据
        total = jnk3_val + gsk3_val + qed_val + sa_norm + tpsa_norm + logp_norm
        sum_scores.append(total)

    return sum_scores


def get_synthesis_molecules(tgt_data_iter):
    opt = OPT_TRANSLATE()
    token_list = []
    for smi in tgt_data_iter:
        token_smi = smi_tokenizer(smi)
        token_list.append(token_smi)
    all_predictions = synthesis(opt, token_list)

    population = []
    all_score = []
    all_population = []
    all_population_score = []

    for pair in all_predictions:
        each_pair_synthesis_list = []
        for each in pair:
            raw_smile = each.replace(" ", "").split('.')
            for sm in raw_smile:
                if Chem.MolFromSmiles(sm) != None:
                    each_pair_synthesis_list.append(sm)

        each_pair_synthesis_list = set(each_pair_synthesis_list)
        each_pair_synthesis_list = list(each_pair_synthesis_list)

        # 🚀 替换打分模块：使用全新的 6 维综合评判选拔前 3 名子代
        if len(each_pair_synthesis_list) > 0:
            score = get_local_6d_sum_scores(each_pair_synthesis_list)
            # print('最大 6 维总得分: ', max(score))

            score_array = np.array(score)
            nev_sort_index = np.argsort(-score_array)
            best_3_index = nev_sort_index[:3]

            sons = np.array(each_pair_synthesis_list)[best_3_index].tolist()
            sons_scores = score_array[best_3_index].tolist()

            all_population.extend(each_pair_synthesis_list)
            all_population_score.extend(score)

            population.extend(sons)
            all_score.extend(sons_scores)

    return all_score, population, all_population, all_population_score


if __name__ == "__main__":
    opt = OPT_TRANSLATE()
    logger = init_logger(opt.log_file)