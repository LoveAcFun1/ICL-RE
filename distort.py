import math
import json
# import faiss
from tqdm import tqdm
from transformers import pipeline
import numpy as np
import copy
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from shared.prompt import instance


PROMPT= '''
You are an excellent counterfactual example generation expert. Your task is to generate counterfactual analysis examples for relation extraction. Now you are given two entities and sentences:

Entity 1: {ent1}

Entity 2: {ent2}

Sentence: {sen}

**Requirements**
You need to insert appropriate content between entities to increase the distance between entities.
The content you add should conform to the real logic.

**Output format**
Sentence: XXX
'''

class Distort:
    def __init__(self, train_set):
        self.train_set = train_set


    def swap_entities(self):
        """
        交换句子中的实体位置。
        :param data: 包含句子和实体信息的字典。
        :return: 修改后的句子和实体信息。
        """
        OUT = []
        for data in self.train_set:
            data = copy.deepcopy(data)
            sentences = data["sentences"][0]
            ner = data["ner"][0]

            # 获取实体的位置和文本
            first_entity_start, first_entity_end, _ = ner[0]
            second_entity_start, second_entity_end, _ = ner[1]
            
            # 提取实体
            first_entity = sentences[first_entity_start:first_entity_end+1]
            second_entity = sentences[second_entity_start:second_entity_end+1]
            
            # 交换实体
            new_sentences = sentences[:first_entity_start] + second_entity + \
                            sentences[first_entity_end+1:second_entity_start] + first_entity + \
                            sentences[second_entity_end+1:]
            
            # 更新NER位置，这里简化处理，假设实体长度不变
            ner[0] = [second_entity_start, second_entity_start + len(first_entity) - 1, ner[0][2]]
            ner[1] = [first_entity_start, first_entity_start + len(second_entity) - 1, ner[1][2]]
            
            # 更新句子
            data["sentences"][0] = new_sentences
            data["ner"][0] = ner
            OUT.append(data)

        return OUT

    def length_distort(self, sentence_list, ent1, ent2):

        pass
    def entMask_distort(self):
        OUT = []
        for item in self.train_set:
            item = copy.deepcopy(item)
            sentences_list = item["sentences"][0]
            ent1 = 'Mask1'
            ent2 = 'Mask2'
            if item["ner"][0][1][0] < item["ner"][0][0][0]:
                sentences_list[item["ner"][0][0][0]:item["ner"][0][0][1]+1] = ent1
                sentences_list[item["ner"][0][1][0]:item["ner"][0][1][1]+1] = ent2
            else:
                sentences_list[item["ner"][0][1][0]:item["ner"][0][1][1]+1] = ent2
                sentences_list[item["ner"][0][0][0]:item["ner"][0][0][1]+1] = ent1

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
                item['relations'] = [[[p1, p1, p2, p2, item["relations"][0][0][-1]]]]
                OUT.append(item)
        return OUT
    


class Filter:
    def __init__(self, train_set):
        self.train_set = train_set
        self.train_len = []
        self.entity_dis = []
        self.entity_frequency = {}
        self.pair_frequency = []
        for i, item in enumerate(train_set):
            sentences_list = item["sentences"][0]
            ent1 = " ".join(sentences_list[item["ner"][0][0][0]:item["ner"][0][0][1]+1])
            ent2 = " ".join(sentences_list[item["ner"][0][1][0]:item["ner"][0][1][1]+1])
            if ent1 in self.entity_frequency:
                self.entity_frequency[ent1] += 1
            else:
                self.entity_frequency[ent1] = 1   
            if ent2 in self.entity_frequency:
                self.entity_frequency[ent2] += 1
            else:
                self.entity_frequency[ent2] = 1

        for i, item in enumerate(train_set):
            self.train_len.append(len(item["sentences"][0]))
            self.entity_dis.append(abs(item["ner"][0][1][0] - item["ner"][0][0][0]))
            sentences_list = item["sentences"][0]
            ent1 = " ".join(sentences_list[item["ner"][0][0][0]:item["ner"][0][0][1]+1])
            ent2 = " ".join(sentences_list[item["ner"][0][1][0]:item["ner"][0][1][1]+1])
            self.pair_frequency.append(self.entity_frequency[ent1]+self.entity_frequency[ent2])

    def len_dis_fre(self):
        OUT = []
        for i, item in enumerate(self.train_set):
            sentences_list = item["sentences"][0]
            ent1 = " ".join(sentences_list[test_sen["ner"][0][0][0]:test_sen["ner"][0][0][1]+1])
            ent2 = " ".join(sentences_list[test_sen["ner"][0][1][0]:test_sen["ner"][0][1][1]+1])
            fre1 = self.entity_frequency[ent1]
            fre2 = self.entity_frequency[ent2]
            dis = self.entity_dis[i]
            length = self.train_len[i]
            OUT.append([fre1, fre2, dis, length])
        return OUT
    
    def length_filter(self, test_sen, theta):
        sentence_list = test_sen["sentences"][0]
        length = len(sentence_list)
        good_set = []
        bad_set = []
        for i, train_len in enumerate(self.train_len):
            if abs(train_len - length) < theta:
                good_set.append(self.train_set[i])
            else:
                bad_set.append(self.train_set[i])

        return good_set, bad_set

    def ent_dis_filter(self, test_sen, theta):
        ent_dis = test_sen["ner"][0][1][0] - test_sen["ner"][0][0][0]
        good_set = []
        bad_set = []
        for i, train_entdis in enumerate(self.entity_dis):
            if abs(train_entdis - abs(ent_dis)) < theta:
                good_set.append(self.train_set[i])
            else:
                bad_set.append(self.train_set[i])
        return good_set, bad_set

    def frequency_filter(self, test_sen, theta):
        sentences_list = test_sen["sentences"][0]
        ent1 = " ".join(sentences_list[test_sen["ner"][0][0][0]:test_sen["ner"][0][0][1]+1])
        ent2 = " ".join(sentences_list[test_sen["ner"][0][1][0]:test_sen["ner"][0][1][1]+1])
        fre_1 = 0 if ent1 not in self.entity_frequency else self.entity_frequency[ent1]
        fre_2 = 0 if ent2 not in self.entity_frequency else self.entity_frequency[ent2]
        good_set = []
        bad_set = []
        for i, train_entdis in enumerate(self.pair_frequency):
            if abs(train_entdis - fre_1-fre_2) < theta:
                good_set.append(self.train_set[i])
            else:
                bad_set.append(self.train_set[i])
        return good_set, bad_set

    def get_distribution(self, train_sen):
        sentences_list = train_sen["sentences"][0]
        ent1 = " ".join(sentences_list[train_sen["ner"][0][0][0]:train_sen["ner"][0][0][1]+1])
        ent2 = " ".join(sentences_list[train_sen["ner"][0][1][0]:train_sen["ner"][0][1][1]+1])
        fre1 = self.entity_frequency[ent1]
        fre2 = self.entity_frequency[ent2]
        sen_len = len(sentences_list)
        ent_dis = abs(train_sen["ner"][0][0][0] - train_sen["ner"][0][1][0])
        distribution = ("The length of a sentence is \""+ str(sen_len) +"\" , the distance between entities is \""+str(ent_dis)+"\" and entity 1 frequency is \""+ str(fre1)+"\", entity 1 frequency is \"" + str(fre2)+"\"" )
        distribution2 = ("Length: \""+ str(sen_len) +"\" distence: \""+str(ent_dis)+ "\" frequency 1 : \""+ str(fre1)+"\" frequency 2 : \""+ str(fre2)+"\"")
        return distribution

    def get_distort_sen(self, train_sen):
        pre_sen = instance(train_sen['origin']).reference
        sentences_list = train_sen["sentences"][0]
        sentence = " ".join(sentences_list)
        ent1 = " ".join(sentences_list[train_sen["ner"][0][0][0]:train_sen["ner"][0][0][1]+1])
        ent2 = " ".join(sentences_list[train_sen["ner"][0][1][0]:train_sen["ner"][0][1][1]+1])

        sen_len = len(sentences_list)
        ent_dis = abs(train_sen["ner"][0][0][0] - train_sen["ner"][0][1][0])
        a = "The length of a sentence is \""+ str(sen_len) +"\" , the distance between entities is \""+str(ent_dis)+"\"."
        distribution = pre_sen + ("The relation between \"" + ent1 + "\" and \""
                          + ent2 + "\" in the sentence \"" + sentence + "\"" )
        
        return distribution

    def get_semantic(self, train_sen):
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
        return instance(item).reference

    # 获取反事实分析样例
    def entMask_distort(self):
        OUT = []
        for item_train in self.train_set:
            item = copy.deepcopy(item_train)
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
                item['origin'] = item_train
                item["ner"][0][0][0] = p1 
                item["ner"][0][0][1] = p1
                item["ner"][0][1][0] = p2 
                item["ner"][0][1][1] = p2
                item['relations'] = [[[p1, p1, p2, p2, item["relations"][0][0][-1]]]]
                OUT.append(item)
        return OUT

    def length_distort(self, path):
        OUT = []
        with open(path, 'r') as f:
            for line in f:
                OUT.append(json.loads(line))
        return OUT









# def get_train_example(example_path, reltoid, no_na):
#     example_dict = {k:list() for k in reltoid.values()}
#     with open(example_path, "r") as f:
#         for line in f.read().splitlines():
#             tmp_dict = json.loads(line)
#             if tmp_dict["relations"] == [[]]:
#                 rel = "NONE"
#                 example_dict[reltoid[rel]].append(tmp_dict)
#             else:
#                 rel = tmp_dict["relations"][0][0][4]
#                 example_dict[reltoid[rel]].append(tmp_dict)
#     return example_dict
# semeval_reltoid = {"Other":0,"Cause-Effect":1, "Component-Whole":2, "Entity-Destination":3, "Entity-Origin":4, "Product-Producer": 5, "Member-Collection":6, "Message-Topic": 7, "Content-Container":8, "Instrument-Agency":9}
# example_dict = get_train_example("/group/40064/johnbli/Code/GPT-RE/dataset/semeval_gpt/train.json", semeval_reltoid, 0)
# train_list = [x for y in example_dict.values() for x in y]
# filter = Filter(train_list)
# # a, b = filter.length_filter(train_list[0], 10)
# # a, b = filter.ent_dis_filter(train_list[0], 4)
# a, b = filter.frequency_filter(train_list[0], 5)
# print(len(a))
# print(len(b))

