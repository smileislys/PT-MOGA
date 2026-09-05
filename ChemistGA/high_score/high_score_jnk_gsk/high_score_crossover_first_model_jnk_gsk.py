# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import random
import torch
from rdkit import Chem

from high_score_properties_jnk_gsk import multi_scoring_functions
from transformer_model.onmt.utils.logging import init_logger
from transformer_model.onmt.translate.translator import build_translator
from transformer_model.onmt.opts_translate import OPT_TRANSLATE

import numpy as np


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
    # 强制将显卡序号设为 0（对应你的 RTX 3080 Ti）
    opt.gpu = 0
    torch.cuda.set_device(0)
    # 强行创建 Transformer 需要的临时文件夹，防弹！
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

        # 打分模块
        score = multi_scoring_functions(each_pair_synthesis_list, ['jnk3', 'gsk3', 'qed', 'sa'])
        print('最大得分: ', max(score))

        # 逆序输出他们的index值
        # 稳妥起见，强制转成 np.array 再取负号，防止 list 类型报错
        score_array = np.array(score)
        nev_sort_index = np.argsort(-score_array)
        best_3_index = nev_sort_index[:3]

        # 极其干净卫生的切片与转换
        sons = np.array(each_pair_synthesis_list)[best_3_index].tolist()
        sons_scores = score_array[best_3_index].tolist()

        # 记录所有推荐的分子
        all_population.extend(each_pair_synthesis_list)
        all_population_score.extend(score)

        population.extend(sons)
        all_score.extend(sons_scores)

    return all_score, population, all_population, all_population_score


if __name__ == "__main__":
    opt = OPT_TRANSLATE()
    logger = init_logger(opt.log_file)