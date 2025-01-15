import math
import json
# import faiss
from tqdm import tqdm
from transformers import pipeline
import numpy as np
import copy
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import re


LEN_PROMPT= '''
You are an excellent counterfactual example generation expert. Your task is to generate counterfactual analysis examples for relation extraction. Now you are given two entities and sentences:

Entity 1: {ent1}

Entity 2: {ent2}

Sentence: {sen}

**Requirements**
The content you add should conform to the real logic.
Do not output anything other than the augmented sentence.
Insert appropriate text between \"{ent1}\" and \"{ent2}\" to increase their distance in the sentence.
The generated sentence must contain \"{ent1}\" and \"{ent2}\", and the tenses of the two cannot change.

**Output format**
XXX
'''

class LLM_Distort:
    def __init__(self, train_set):
        self.train_set = train_set
        model_name_or_path = '/group/40064/heyfonli/model_hub/Qwen2.5-7B-Instruct/'
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='npu:6', trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


    def generate(self, ent1, ent2, sentence):
        prompt = LEN_PROMPT.format(ent1 = ent1,ent2 = ent2, sen = sentence)
        messages = [{'role': 'system', 'content': 'You are an AI assistant '}, {'role': 'user', 'content': prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # print(text)
        model_inputs = self.tokenizer([text], return_tensors="pt").to('npu:6')
        generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
            )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def len_distort(self, output_path):
        f2 = open(output_path, 'a', encoding='utf-8')
        OUT = []
        for item_train in tqdm(self.train_set):
            item = copy.deepcopy(item_train)
            sentences_list = item["sentences"][0]
            ent1 = sentences_list[item["ner"][0][0][0]:item["ner"][0][0][1]+1]
            ent2 = sentences_list[item["ner"][0][1][0]:item["ner"][0][1][1]+1]
            sentences_list[item["ner"][0][0][0]:item["ner"][0][0][1]+1] = ['Mask1']
            sentences_list[item["ner"][0][1][0]:item["ner"][0][1][1]+1] = ['Mask2']
            sentence = " ".join(sentences_list)
            res = self.generate('Mask1', 'Mask2', sentence)
            res_list = re.findall(r'\w+|[^\w\s]', res, re.UNICODE)
            op,p1 = self.replace_word(res_list, 'Mask1', ent1)
            op,p2 = self.replace_word(op, 'Mask2', ent2)
            if p1 != -1 and p2!= -1:
                item["sentences"][0] = op
                item["ner"][0][0][0] = p1 
                item["ner"][0][0][1] = p1 + len(ent1) - 1
                item["ner"][0][1][0] = p2 
                item["ner"][0][1][1] = p2 + len(ent2) - 1
                item['relations'] = [[[p1, p1 + len(ent1) - 1, p2, p2 + len(ent2) - 1, item["relations"][0][0][-1]]]]
                json_str = json.dumps(item)
                # 将JSON字符串写入文件，并添加换行符以便区分
                f2.write(json_str + '\n')

    def replace_word(self, words, old_word, new_words):
        # 替换列表中的单词
        result = []
        p = -1
        for i, word in enumerate(words):
            if word == old_word:
                result.extend(new_words)  # 使用extend添加多个元素
                p = i
            else:
                result.append(word)
        return result, p


def get_train_example(example_path, reltoid, no_na):
    example_dict = {k:list() for k in reltoid.values()}
    with open(example_path, "r") as f:
        for line in f.read().splitlines():
            tmp_dict = json.loads(line)
            if tmp_dict["relations"] == [[]]:
                rel = "NONE"
                example_dict[reltoid[rel]].append(tmp_dict)
            else:
                rel = tmp_dict["relations"][0][0][4]
                example_dict[reltoid[rel]].append(tmp_dict)
    return example_dict
semeval_reltoid = {"Other":0,"Cause-Effect":1, "Component-Whole":2, "Entity-Destination":3, "Entity-Origin":4, "Product-Producer": 5, "Member-Collection":6, "Message-Topic": 7, "Content-Container":8, "Instrument-Agency":9}
example_dict = get_train_example("/group/40064/johnbli/Code/GPT-RE/dataset/semeval_gpt/train.json", semeval_reltoid, 0)
train_list = [x for y in example_dict.values() for x in y]
filter = LLM_Distort(train_list)
# a, b = filter.length_filter(train_list[0], 10)
# a, b = filter.ent_dis_filter(train_list[0], 4)
filter.len_distort('/group/40064/johnbli/Code/GPT-RE/dataset/semeval_gpt/train_len_distort.json')

