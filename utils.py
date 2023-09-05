# pip install pypinyin
import numpy as np
import re
from ltp import LTP
from pypinyin import lazy_pinyin
import sqlite3
import pandas as pd
import spacy
from tqdm import tqdm

def merge_spans(intervals):
    """合并有交集的区间"""
    intervals.sort(key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged

def merge_labeled_spans(spacy_list, text, return_positions=False):
    """按照标签合并有交集的区间"""
    merged_list = []
    for s, e, label in spacy_list:
        if merged_list and merged_list[-1][1] == s and merged_list[-1][2] == label:
            merged_list[-1][1] = e
        else:
            merged_list.append([s, e, label])
    if return_positions:
        return merged_list
    merged_list = {text[s:e] for s, e, _ in merged_list}
    return merged_list

def get_merged_spans(text, ents):
    """获取实体的位置区间，同时合并有交集的部分"""
    all_spans = []
    for ent in ents:
        try:
            spans = [[match.start(), match.end()] for match in re.finditer(ent, text)]
            all_spans.extend(spans)
        except:
            pass
    merged_spans = np.array(merge_spans(all_spans))
    return merged_spans

def get_ents_zh(text, ltp, spacy_model):
    """中文实体抽取函数，返回一个去重后的实体列表，在必要时需要转化为字符串"""
    label_set = ['DATE', 'MONEY', 'PERCENT', 'QUANTITY', 'TIME']
    ner_list = {ent for _, ent in ltp.pipeline([text], tasks=["cws", "pos", "ner"]).ner[0]}
    spacy_list = [(ent.start_char, ent.end_char, ent.label_) for ent in spacy_model(text).ents if ent.label_ in label_set]
    spacy_list = merge_labeled_spans(spacy_list, text)
    work_of_art = set(re.findall(r'(《.*?》)', text) + re.findall(r'(“.*?”)', text))
    ner_list = list(set.union(ner_list, spacy_list, work_of_art))
    return ner_list

def get_ents_en(text, spacy_model):
    """英文实体抽取函数，返回一个去重后的实体列表，在必要时需要转化为字符串"""
    label_set = ['DATE', 'MONEY', 'PERCENT', 'QUANTITY', 'TIME','GPE','LOC',"PERSON","WORK_OF_ART","ORG","NORP","LAW","FAC","LANGUAGE"]
    spacy_list = [(ent.start_char, ent.end_char, ent.label_) for ent in spacy_model(text).ents if ent.label_ in label_set]
    spacy_list = merge_labeled_spans(spacy_list, text)
    ner_list = list(spacy_list)
    return ner_list

def get_labelled_text(text, spacy_model, return_ents=True):
    """标签匿名化函数，输出标签匿名后的文本，可选是否返回对应的实体列表"""
    label_set = ['DATE', 'MONEY', 'PERCENT', 'QUANTITY', 'TIME','GPE','LOC',"PERSON","WORK_OF_ART","ORG","NORP","LAW","FAC","LANGUAGE"]
    spacy_list = [(ent.start_char, ent.end_char, ent.label_) for ent in spacy_model(text).ents if ent.label_ in label_set]
    spacy_list = merge_labeled_spans(spacy_list, text, return_positions=True)
    positions = np.array([ent[:2] for ent in spacy_list])
    labels = [ent[2] for ent in spacy_list]
    ner_list = set()
    for i in range(len(spacy_list)):
        s, e = positions[i]
        label = labels[i]
        ner_list.add(text[s:e])
        text = text[:s] + f'<{label}>' + text[e:]
        positions[i:,:] = positions[i:,:] + len(label) + 2 - (e - s)
    if return_ents:
        return text, list(ner_list)
    else:
        return text

def get_labelled_text_with_id(text, spacy_model, return_ents=True):
    """标签匿名化函数并附加id，输出标签匿名后的文本，可选是否返回对应的实体列表"""
    label_set = ['DATE', 'MONEY', 'PERCENT', 'QUANTITY', 'TIME','GPE','LOC',"PERSON","WORK_OF_ART","ORG","NORP","LAW","FAC","LANGUAGE"]
    label_set = {k:{'<cur_id>': 0} for k in label_set}
    spacy_list = [(ent.start_char, ent.end_char, ent.label_) for ent in spacy_model(text).ents if ent.label_ in label_set.keys()]
    spacy_list = merge_labeled_spans(spacy_list, text, return_positions=True)
    positions = np.array([ent[:2] for ent in spacy_list])
    labels = [ent[2] for ent in spacy_list]
    ner_list = set()
    for i in range(len(spacy_list)):
        s, e = positions[i]
        ent_text = text[s:e]
        label = labels[i]
        if ent_text not in label_set[label]:
            label_set[label][ent_text] = label_set[label]['<cur_id>']
            label_set[label]['<cur_id>'] += 1
        label = f'<{label}_{label_set[label][ent_text]}>'
        ner_list.add(ent_text)
        text = text[:s] + label + text[e:]
        positions[i:,:] = positions[i:,:] + len(label) - (e - s)
    if return_ents:
        return text, list(ner_list)
    else:
        return text

def mark_ents(text, spacy_model, return_ents=True):
    """识别文本中的实体并用'<>'括起来，用于黑盒攻击"""
    label_set = ['DATE', 'MONEY', 'PERCENT', 'QUANTITY', 'TIME','GPE','LOC',"PERSON","WORK_OF_ART","ORG","NORP","LAW","FAC","LANGUAGE"]
    spacy_list = [(ent.start_char, ent.end_char, ent.label_) for ent in spacy_model(text).ents if ent.label_ in label_set]
    spacy_list = merge_labeled_spans(spacy_list, text, return_positions=True)
    positions = np.array([ent[:2] for ent in spacy_list])
    # labels = [ent[2] for ent in spacy_list]
    ner_list = set()
    for i in range(len(spacy_list)):
        s, e = positions[i]
        # label = labels[i]
        ner_list.add(text[s:e])
        text = text[:s] + f'<{text[s:e]}>' + text[e:]
        positions[i:,:] = positions[i:,:] + 2
    if return_ents:
        return text, list(ner_list)
    else:
        return text

def add_pinyin(text, ltp):
    """拼音注入，用于中译英，优化人名翻译"""
    person_ents = {ent for label, ent in ltp.pipeline([text], tasks=["cws", "pos", "ner"]).ner[0] if label == 'Nh'}
    # person_ents = sorted(list(person_ents), key=lambda x: len(x), reverse=True)
    person_ents = get_merged_spans(text, person_ents)
    for i in range(len(person_ents)):
        s, e = person_ents[i]
        if e - s > 3:
            continue
        pinyin = lazy_pinyin(text[s:e])
        for k in range(len(pinyin)):
            if k < 2:
                pinyin[k] = pinyin[k].capitalize()
        if len(pinyin) > 2:
            pinyin = [pinyin[0], pinyin[1] + pinyin[2]]
        pinyin = ' '.join(pinyin)
        text = text[:s] + text[s:e] + f'({pinyin})' + text[e:]
        person_ents[i:,:] = person_ents[i:,:] + len(pinyin) + 2
    return text

if __name__ == '__main__':
    # 一些预处理可以在这里完成
    spacy_model = spacy.load('en_core_web_trf')
    names = ['attribute_ruler', 'tagger', 'parser', 'lemmatizer']
    spacy_model.disable_pipes(names)
    # conn = sqlite3.connect('../database/database.sqlite')
    conn = sqlite3.connect('../database/attack.sqlite')
    data = conn.execute('SELECT id, sub_model_560m_raw FROM EN').fetchall()
    for id, text in tqdm(data):
        # labelled_text, ents = get_labelled_text(text, spacy_model)
        marked_text = mark_ents(text, spacy_model, return_ents=False)
        # print(text)
        # labelled_text = get_labelled_text_with_id(text, spacy_model, return_ents=False)
        # print(labelled_text)
        conn.execute('UPDATE EN SET sub_model_560m = ? WHERE id = ?', (marked_text, id))
        conn.commit()