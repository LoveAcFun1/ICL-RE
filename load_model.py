from model import RE_BertModel
from utils import GetDataEnumerater,prepare_label_set
from transformers import AutoTokenizer
from KNN_RE import get_train_example
from tqdm import tqdm
import numpy as np
import torch
import os
import json

semeval_reltoid = {"Other":0,"Cause-Effect":1, "Component-Whole":2, "Entity-Destination":3, "Entity-Origin":4, "Product-Producer": 5, "Member-Collection":6, "Message-Topic": 7, "Content-Container":8, "Instrument-Agency":9}

class Args():
    def __init__(self):
        self.relation_type_set = None
        self.relation_set_dict = None
        self.tokenizer = None
        self.relation_class_num = None
        self.max_seq_length = None
        self.dataset = 'SemEval'
        self.retriever_plm = '/group/40064/johnbli/bert-based-models/roberta-large'

# args = Args()
def get_samples(a_list):
    for item in a_list:
        item['sentence'] = item['sentences'][0]
        item['head_entity'] = {'start_idx': item['ner'][0][0][0], 'end_idx': item['ner'][0][0][1], 'span': ' '.join(item['sentences'][0][item['ner'][0][0][0]:item['ner'][0][0][1]+1]), 'type': None}
        item['tail_entity'] = {'start_idx': item['ner'][0][1][0], 'end_idx': item['ner'][0][1][1], 'span': ' '.join(item['sentences'][0][item['ner'][0][1][0]:item['ner'][0][1][1]+1]), 'type': None}
        item['relation_type'] = item['relations'][0][0][-1]
    return a_list

def get_emb(args, data_list, dataset, retriever_plm, k_short, device):
    setattr(args, 'dataset', dataset)
    setattr(args, 'retriever_plm', retriever_plm)

    # with open(f'./SRVF-main/data/{args.dataset}/test.json', encoding='utf-8') as f:
    #     test_samples = json.loads(f.read())
    if isinstance(data_list, str):
        with open(f'./SRVF-main/data/{args.dataset}/sampled_{k_short}_shot_train.json', encoding='utf-8') as f:
            test_samples = json.loads(f.read())
    else:
        test_samples = get_samples(data_list)

    relation_type_set, relation_set_dict = prepare_label_set(dataset=args.dataset)
    setattr(args, 'relation_type_set', relation_type_set)
    setattr(args, 'relation_set_dict', relation_set_dict)
    setattr(args, 'relation_class_num', len(relation_type_set))
    setattr(args, 'tokenizer', AutoTokenizer.from_pretrained(args.retriever_plm))
    setattr(args, 'max_seq_length', 128)

    ckpt_dir = "/group/40064/johnbli/Code/GPT-RE/checkpoint-stage1/bert_base_uncased/SemEval-42-{}-shot".format(k_short)

    checkpoint_bert_model = torch.load(os.path.join(ckpt_dir, 'bert_re_retriever.ckpt'), map_location=device)
    trained_RE_model = RE_BertModel(PLM=args.retriever_plm,
                                    PLM_hidden_size=768,
                                    relation_class_num=len(args.relation_type_set)).to(device)
    trained_RE_model.load_state_dict(checkpoint_bert_model['model_state_dict'])
    emb_retriever_stage1 = trained_RE_model
    emb_retriever_stage1.eval()
    ##数据处理

    dataloader_test_examples = GetDataEnumerater(args=args,
                                                samples=test_samples,
                                                batch_size=64,
                                                )
    for out_batch_idx, batch in tqdm(enumerate(dataloader_test_examples), desc=f'Running retrieval'):

        batch_samples_emb = emb_retriever_stage1.get_emb(
            input_ids=batch[0].to(device),
            special_mask=batch[1].to(device),
            token_type_ids=batch[2].to(device),
            attention_mask=batch[3].to(device),
        )
        if out_batch_idx == 0:
            logits_test_examples = batch_samples_emb.cpu()
        else:
            logits_test_examples = torch.cat((logits_test_examples, batch_samples_emb.cpu()), dim=0)

    return logits_test_examples

def find_lmknn_example(gpu_index_flat, test_samples, train_dict, train_sentences, k):
    # out_list = []
    
    D, I = gpu_index_flat.search(test_samples, k)
    knn_list = [train_dict[train_sentences[i]] for i in I[0,:k]]
    

    return knn_list


# args =Args()
# path = "/group/40064/johnbli/Code/GPT-RE/dataset/semeval_gpt/train.json"
# dataset = 'SemEval'
# retriever_plm = "/group/40064/johnbli/bert-based-models/roberta-large"
# k_short = 5
# device = torch.device("npu:7")
# train_samples = get_emb(args, path, dataset, retriever_plm, k_short, device)

# test_samples = train_samples
# import faiss

# index_flat = faiss.IndexFlatL2(1024)  # 创建一个使用L2距离的平面索引
# cpu_index_flat = index_flat  # 直接使用CPU索引
# embed_list = np.array(train_samples)
# cpu_index_flat.add(embed_list)  # 使用CPU索引添加特征向量

# D, I = cpu_index_flat.search(embed_list, 3)
# print("666")