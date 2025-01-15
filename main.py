import sys
import re
from tqdm import tqdm
import random
import json
import argparse
import time
from tqdm import tqdm
import requests
import faiss
import numpy as np
from simcse import SimCSE
from shared.prompt import instance
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import pipeline
from KNN_RE import get_train_example, find_knn_example, find_lmknn_example, check_list, check_Semantics_list, get_top_index
from load_model import get_emb,Args
from distort import Distort
from distort import Filter
from label_explain import SemEval_label2exp
from shared.const import semeval_reltoid
from shared.const import semeval_idtoprompt
from shared.const import ace05_reltoid
from shared.const import ace05_idtoprompt
from shared.const import tacred_reltoid
from shared.const import scierc_reltoid
from shared.const import wiki_reltoid
sys.path.append("/group/40064/johnbli/Code/UGC")
from examples.gpt_http import retry_request_openai_summary

params = {
    # 任务信息
    "env": "production",  # 必填，（取值范围：production、development）
    "task_name": "逃出精神病院",
    "task_type": "文本生成任务",  # 必填，任务类型（取值范围：qa任务、文本生成任务、整段对话生产、多轮对话、其他）
    "task_description": "",
    "business_id":"134"
}
import copy
import math
from sklearn.metrics import f1_score

GPT_PROMPT = '''
You are a relation extraction expert specializing in the field of relation extraction, responsible for identifying and extracting the relationship between entities from text to support applications such as information retrieval, knowledge graph construction and data analysis. The following is the entity information:

Entity 1: {ent1}

Entity 2: {ent2}

The categories of relations include the following: {rels}

Your task is to process the text given below to accurately identify and extract the relationship between entity 1 and entity 2 in the text from the list of relation categories.

**Requirements**
You must accurately identify the type of relationship between entities in the sentence.
The response should not contain any information other than the entity category.

**Examples**
{examples}

Please judge the relationship between the following sentences based on the above examples:
**Input**
{sen}

Output format:
Relation: XXX

'''

TRAIN_SUP = '''
Sample {num}:
    Entity 1: {ent1}
    Entity 2: {ent2}
    Input:{sen}
    Output:Relation: {label}
'''


def compute_f1(Labels, predict):

    # 计算 F1 值
    macro_f1 = f1_score(Labels, predict, average='macro')  # 'weighted' 考虑各类标签的不平衡问题
    micro_f1 = f1_score(Labels, predict, average='micro')  # 'weighted' 考虑各类标签的不平衡问题

    print("micro_f1 Score:", micro_f1)
    print("macro_f1 Score:", macro_f1)

def inference(model, tokenizer, test_data, select_items, train_list, idtoprompt, reltoid, args):
    Labels = []
    predict = []
    num = 0
    ids = 0
    correct = []

    for test_item in tqdm(test_data):

        sentences_list = test_item["sentences"][0]
        ent1 = " ".join(sentences_list[test_item["ner"][0][0][0]:test_item["ner"][0][0][1]+1])
        ent2 = " ".join(sentences_list[test_item["ner"][0][1][0]:test_item["ner"][0][1][1]+1])
        sentence = " ".join(sentences_list)
        rels = ""
        rels_lst = {}
        for k,v in idtoprompt.items():
            rels = rels + v + '; '
            rels_lst[v] = k

        label = reltoid[test_item["relations"][0][0][-1]]
        
        # 添加instance
        example = ""
        if args.use_instance:
            if args.use_knn:
                selected_items = select_items[ids]
                ids += 1
                # print(selected_items)
            elif args.use_pipline:
                selected_items = select_items[ids]
                ids += 1
            else:
                selected_items = random.sample(train_list, args.k_sample)
            for i,item in enumerate(selected_items):
                train_ent1,train_ent2,train_sentence,train_label = get_example(item)
                train_label = idtoprompt[reltoid[train_label]]
                example = example + TRAIN_SUP.format(num = i, ent1 = train_ent1,ent2 = train_ent2,label=train_label, sen = train_sentence)
        
        Labels.append(label)
        res_lst = get_Qwen_res_7b(model, tokenizer, ent1, ent2, rels,example, sentence, args)        #Qwen2.5-7b
        
        res = res_lst[0].replace("Relation: ","") # 无理由

        if res not in rels_lst:
            num += 1
            predict.append(0 if label!=0 else 1)
        else:
            predict.append(rels_lst[res])

    compute_f1(Labels, predict)

def get_Qwen_res_7b(model, tokenizer, ent1, ent2, rels, examples, sentence, args):
    prompt = GPT_PROMPT.format(ent1 = ent1,ent2 = ent2,rels=rels,examples=examples, sen = sentence)
    messages = [{'role': 'system', 'content': 'You are an AI assistant '}, {'role': 'user', 'content': prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # print(text)
    model_inputs = tokenizer([text], return_tensors="pt").to('npu:{}'.format(args.num_npu))
    generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
        )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return [response]

def get_example(item):
    sentences_list = item["sentences"][0]
    ent1 = " ".join(sentences_list[item["ner"][0][0][0]:item["ner"][0][0][1]+1])
    ent2 = " ".join(sentences_list[item["ner"][0][1][0]:item["ner"][0][1][1]+1])
    sentence = " ".join(sentences_list)
    label = item["relations"][0][0][-1]
    return ent1,ent2,sentence,label

def main(model, tokenizer, test_data, train_dict, train_sentences, train_distribution, args, reltoid, idtoprompt, Filter_all):
    # use_instance = True
    # use_knn = True
    # use_pipline = False
    # k_sample = 50
    select_items = []
    
    # 创建语义反事实空间
    # semantic_items = Filter_all.entMask_distort()
    # train_distort = [Filter_all.get_distort_sen(x) for x in semantic_items]
    # train_distor_dict = {Filter_all.get_distort_sen(x):x for x in semantic_items}
    # 创建长度反事实空间
    # lengths_items = Filter_all.length_distort('/group/40064/johnbli/Code/GPT-RE/dataset/semeval_gpt/train_len_distort.json')
    # train_distort = [instance(x).reference for x in lengths_items]
    # train_distor_dict = {instance(x).reference : x for x in lengths_items}
    # # 创建分布反事实空间
    # knn_model_distort = SimCSE("/group/40064/johnbli/bert-based-models/sup-simcse-roberta-large")
    # knn_model_distort.build_index(train_distort, device="cpu")


    if args.use_knn:
        knn_model = SimCSE("/group/40064/johnbli/bert-based-models/sup-simcse-roberta-large")
        knn_model.build_index(train_reference, device="cpu")
        select_items = find_knn_example(knn_model, test_data, train_dict, args.k_sample, True)
        # 检查items分布，得到
        # distribution_score = check_list(select_items, Filter_all) #分布偏差分数
        # semantics_score = check_Semantics_list(select_items, Filter_all)
        # all_score = [semantics_score[i] + distribution_score[i] for i in range(len(semantics_score))]

        # all_score = distribution_score

        # # 做筛选
        # top_index = get_top_index(select_items, all_score, args.theta)
        # # 修改examples
        # new_test_data = []
        # for idx in top_index:
        #     new_test_data.append(test_data[idx])
        # # new_test_data = test_data[top_index]
        # new_items = find_knn_example(knn_model_distort, new_test_data, train_distor_dict, args.k_sample, True)
        # for i, index in enumerate(new_items):
        #     select_items[top_index[i]] = new_items[i]   


    if args.use_pipline:
        index_flat = faiss.IndexFlatL2(768)  # 创建一个使用L2距离的平面索引
        cpu_index_flat = index_flat  # 直接使用CPU索引
        args =Args()
        dataset = 'SemEval' # 需要修改
        retriever_plm = "/group/40064/johnbli/bert-based-models/bert_base_uncased"
        device = torch.device("npu:{}".format(args.num_npu))
        train_samples = get_emb(args, train_list, dataset, retriever_plm, k_sample, device) #使用全部数据集
        embed_list = np.array(train_samples)
        cpu_index_flat.add(embed_list)  # 使用CPU索引添加特征向量
    if args.use_pipline:
        test_samples = get_emb(args, test_data, dataset, retriever_plm, k_sample, device)
        D, I = cpu_index_flat.search(test_samples, k_sample)
        select_items = []
        for j in range(I.shape[0]):
            j_select = [train_dict[train_sentences[i]] for i in I[j,:k_sample]]
            select_items.append(j_select)
    

    # 推理

    inference(model, tokenizer, test_data, select_items, train_list, idtoprompt, reltoid, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='semeval_gpt', choices=["ace05","semeval_gpt","tacred","scierc","wiki80"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_knn", type=bool, default=False)
    parser.add_argument("--use_pipline", type=bool, default=False)
    parser.add_argument("--use_instance", type=bool, default=True)
    parser.add_argument("--distort", type=bool, default=False)
    parser.add_argument("--lm_mask", type=int, default=0)
    parser.add_argument("--k_sample", type=int, default=5)
    parser.add_argument("--num_npu", type=int, default=7)
    parser.add_argument("--theta", type=float, default=0.3)
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    tacred_idtoprompt = {tacred_reltoid[k]:k.upper() for k in tacred_reltoid.keys()}
    scierc_idtoprompt = {scierc_reltoid[k]:k.upper() for k in scierc_reltoid.keys()}
    wiki_idtoprompt = {wiki_reltoid[k]:k.upper() for k in wiki_reltoid.keys()}

    if args.task == 'semeval_gpt':
        reltoid = semeval_reltoid
        idtoprompt = semeval_idtoprompt
    elif args.task == 'tacred':
        reltoid = tacred_reltoid
        idtoprompt = tacred_idtoprompt
    elif args.task == 'scierc':
        reltoid = scierc_reltoid
        idtoprompt = scierc_idtoprompt
    print(args)

    # 读取测试数据
    dataset_path = "/group/40064/johnbli/Code/GPT-RE/dataset/{}/test.json".format(args.task)
    test_data = []
    with open(dataset_path, "r") as f:
        for line in f:
            line = json.loads(line)
            if len(line['relations'][0]) == 0:
                line['relations'] = [[[line["ner"][0][0][0], line["ner"][0][0][1], line["ner"][0][1][0], line["ner"][0][1][1], "NONE"]]]
            test_data.append(line)
    
    # 读取训练数据
    example_dict = get_train_example("/group/40064/johnbli/Code/GPT-RE/dataset/{}/train.json".format(args.task), reltoid, 0)
    train_list = [x for y in example_dict.values() for x in y]
    # train_list = [x for x in train_list if semeval_reltoid[x["relations"][0][0][4]] != 0]
    
    # 获取distort训练数据
    if args.distort:
        distorter = Distort(train_list)
        append_set = distorter.swap_entities()
        train_list += append_set
    
    # 初始化过滤器
    Filter_all = Filter(train_list)

    train_distribution = [Filter_all.get_distribution(x) for x in train_list]
    train_sentences = [instance(x).sentence for x in train_list]
    train_reference = [instance(x).reference for x in train_list]
    train_dict = {instance(x).reference:x for x in train_list}



    # 加载模型
    model_name_or_path = '/group/40064/heyfonli/model_hub/Qwen2.5-7B-Instruct/'
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='npu:{}'.format(args.num_npu), trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    main(model, tokenizer, test_data, train_dict, train_sentences, train_distribution, args, reltoid, idtoprompt, Filter_all)

    print(args)