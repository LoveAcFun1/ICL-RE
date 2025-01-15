# from simcse import SimCSE
from shared.prompt import instance
import json
# import faiss
from tqdm import tqdm
from transformers import pipeline
import numpy as np
from simcse import SimCSE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import copy


def get_train_example(example_path, reltoid, no_na):
    example_dict = {k:list() for k in reltoid.values()}
    with open(example_path, "r") as f:
        for line in f.read().splitlines():
            tmp_dict = json.loads(line)
            if tmp_dict["relations"] == [[]]:
                rel = "NONE"
                tmp_dict["relations"] = [[[tmp_dict["ner"][0][0][0], tmp_dict["ner"][0][0][1], tmp_dict["ner"][0][1][0], tmp_dict["ner"][0][1][1], "NONE"]]]
                example_dict[reltoid[rel]].append(tmp_dict)
            else:
                rel = tmp_dict["relations"][0][0][4]
                example_dict[reltoid[rel]].append(tmp_dict)
    return example_dict

def find_knn_example(model, test_dicts, train_dict, k, entity_info):
    out_list = []
    for test_dict in test_dicts:
        if entity_info:
            test_sentences = instance(test_dict).reference
        else:
            test_sentences = reference_test(test_dict)
        test_id = test_dict["doc_key"]
        label_other = 0
        out_list.append(test_sentences)
    knn_result = model.search(out_list, device="cpu", threshold=0.0, top_k=k)
    #print(knn_result)
    # knn_list = [train_dict[x[0]] for x in knn_result]
    knn_list = [[train_dict[y[0]]for y in x] for x in knn_result]

    return knn_list

def find_lmknn_example(gpu_index_flat, test_dicts, train_dict, train_sentences, k):
    out_list = []
    extractor = pipeline(model="/group/40064/johnbli/bert-based-models/roberta-large", task="feature-extraction")
    for test_dict in tqdm(test_dicts):
        test_sentence = instance(test_dict).lm_mask
        result = extractor(test_sentence, return_tensors=True)
        
        embed = result.detach().numpy().copy()
        xq = np.array([embed[0][-3]])
        D, I = gpu_index_flat.search(xq, k)
        knn_list = [train_dict[train_sentences[i]] for i in I[0,:k]]
        out_list.append(knn_list)

    return out_list


def cosine_sim(matrix):
    # 计算余弦相似度矩阵
    similarity_matrix = cosine_similarity(matrix)

    # 计算平均相似度（排除对角线元素）
    average_similarity = (np.sum(similarity_matrix) - np.trace(similarity_matrix)) / (len(matrix) * (len(matrix) - 1))
    return average_similarity

def kmeans_sim(rep):
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(rep)
    intra_cluster_distance = np.mean(np.sqrt(np.sum((rep - kmeans.cluster_centers_[0])**2, axis=1)))
    return intra_cluster_distance


def check_list(select_items, Filter_all):
    model = SimCSE("/group/40064/johnbli/bert-based-models/sup-simcse-roberta-large")

    # dis_list = [kmeans_sim(np.array(model.encode([Filter_all.get_distribution(x) for x in items]))) for items in select_items]
    examples = []
    for items in select_items:
        k_size = len(items)
        tmp_e = [Filter_all.get_distribution(x) for x in items]
        examples = examples + tmp_e
    embeddings = model.encode(examples)
    embeddings_out = np.array(embeddings.reshape(-1, k_size, 1024))
    dis_list = [kmeans_sim(embeddings_out[i]) for i in range(embeddings_out.shape[0])]

    # 归一化列表
    min_val = min(dis_list)
    max_val = max(dis_list)
    normalized_list = [(x - min_val) / (max_val - min_val) for x in dis_list]

    return normalized_list


def check_Semantics_list(select_items, Filter_all):
    model = SimCSE("/group/40064/johnbli/bert-based-models/sup-simcse-roberta-large")

    # dis_list = [kmeans_sim(np.array(model.encode([Filter_all.get_semantic(x) for x in items]))) for items in select_items]
    examples = []
    for items in select_items:
        k_size = len(items)
        tmp_e = [Filter_all.get_semantic(x) for x in items]
        examples = examples + tmp_e
    embeddings = model.encode(examples)
    embeddings_out = np.array(embeddings.reshape(-1, k_size, 1024))
    dis_list = [kmeans_sim(embeddings_out[i]) for i in range(embeddings_out.shape[0])]

    # 归一化列表
    min_val = min(dis_list)
    max_val = max(dis_list)
    normalized_list = [(x - min_val) / (max_val - min_val) for x in dis_list] 

    return normalized_list


def get_top_index(select_items, normalized_list, theta):
    out_items = []
    # 获取前10%的索引
    num_top_items = max(1, int(len(normalized_list) * theta))  # 确保至少有一个元素被选中
    top_indices = np.argsort(normalized_list)[-num_top_items:]

    # for i, items in enumerate(select_items): 
    #     if i in top_indices:
    #         for item in items:
    #             item["is_knn"] = False
    #     else:
    #         for item in items:
    #             item["is_knn"] = True
    #     out_items.append(items)
    return top_indices


def reference_test(train_sen):
    pre_sen = instance(train_sen).reference
    item = copy.deepcopy(train_sen)
    sentences_list = item["sentences"][0]
    ent1 = 'Mask1'
    ent2 = 'Mask2'
    if item["ner"][0][1][0] < item["ner"][0][0][0]:
        sentences_list[item["ner"][0][0][0]:item["ner"][0][0][1]+1] = [ent1]
        sentences_list[item["ner"][0][1][0]:item["ner"][0][1][1]+1] = [ent2]
    else:
        sentences_list[item["ner"][0][1][0]:item["ner"][0][1][1]+1] = [ent2]
        sentences_list[item["ner"][0][0][0]:item["ner"][0][0][1]+1] = [ent1]

    item["sentences"][0] = sentences_list
    p1 = -1
    p2 = -1
    for i,word in enumerate(sentences_list):
        if word == 'Mask1':
            p1 = i
        if word == 'Mask2':
            p2 = i
    if p1 != -1 and p2 != -1:
        item["ner"][0][0][0] = p1 
        item["ner"][0][0][1] = p1
        item["ner"][0][1][0] = p2 
        item["ner"][0][1][1] = p2


    sentence = " ".join(sentences_list)
    sen_len = len(sentences_list)
    ent_dis = abs(item["ner"][0][0][0] - item["ner"][0][1][0])
    distribution = pre_sen + ("The relation between \"" + ent1 + "\" and \""
                        + ent2 + "\" in the sentence \"" + sentence + "\"" )
    return distribution


####################LM######################
# semeval_reltoid = {"Other":0,"Cause-Effect":1, "Component-Whole":2, "Entity-Destination":3, "Entity-Origin":4, "Product-Producer": 5, "Member-Collection":6, "Message-Topic": 7, "Content-Container":8, "Instrument-Agency":9}
# example_dict = get_train_example("/group/40064/johnbli/Code/GPT-RE/dataset/semeval_gpt/train.json", semeval_reltoid, 0)
# train_list = [x for y in example_dict.values() for x in y]
# train_list = [x for x in train_list if semeval_reltoid[x["relations"][0][0][4]] != 0]
# train_dict = {instance(x).lm_mask:x for x in train_list}
# train_sentences = [instance(x).lm_mask for x in train_list]
# # res = faiss.StandardGpuResources()  # 不再需要初始化GPU资源

# index_flat = faiss.IndexFlatL2(1024)  # 创建一个使用L2距离的平面索引

# # gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)  # 不需要转移到GPU
# cpu_index_flat = index_flat  # 直接使用CPU索引

# extractor = pipeline(model="/group/40064/johnbli/bert-based-models/roberta-large", task="feature-extraction")
# embed_array = []
# for item in tqdm(train_sentences[:10]):
#     result = extractor(item, return_tensors=True)
#     embeds = result[0].detach().numpy().copy()
#     embed_array.append(embeds[-3,:])

# embed_list = np.array(embed_array)
# cpu_index_flat.add(embed_list)  # 使用CPU索引添加特征向量

# find_lmknn_example(cpu_index_flat, train_list[:2], train_dict, train_sentences, 2)

###########KNN####################

# semeval_reltoid = {"Other":0,"Cause-Effect":1, "Component-Whole":2, "Entity-Destination":3, "Entity-Origin":4, "Product-Producer": 5, "Member-Collection":6, "Message-Topic": 7, "Content-Container":8, "Instrument-Agency":9}

# example_dict = get_train_example("/group/40064/johnbli/Code/GPT-RE/dataset/semeval_gpt/train.json", semeval_reltoid, 0)


# train_list = [x for y in example_dict.values() for x in y]
# train_list = [x for x in train_list if semeval_reltoid[x["relations"][0][0][4]] != 0]
# train_sentences = [instance(x).sentence for x in train_list]
# train_dict = {instance(x).sentence:x for x in train_list}

# test_list = [x for y in example_dict.values() for x in y]


# knn_model = SimCSE("princeton-nlp/sup-simcse-roberta-large")
# knn_model.build_index(train_sentences, device="cpu")

# a = find_knn_example(knn_model, test_list[:2],train_dict,5,True)
# print("666")