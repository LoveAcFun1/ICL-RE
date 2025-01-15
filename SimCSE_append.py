from simcse import SimCSE
import torch
from tqdm import tqdm
from torch import Tensor, device
from typing import List, Dict, Tuple, Type, Union
import numpy as np
from numpy import ndarray
import transformers
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

def cosine_similarity(A, B):
    # 归一化A和B
    A_norm = A / A.norm(dim=1, keepdim=True)
    B_norm = B / B.norm(dim=2, keepdim=True)
    
    # 计算余弦相似度
    similarity = torch.einsum('ij,ikj->ik', A_norm, B_norm)
    return similarity

class SimCSE_append(SimCSE):
    def __init__(self, model_name_or_path: str, 
                device: str = None,
                num_cells: int = 100,
                num_cells_in_search: int = 10,
                pooler = None):
        # 首先调用父类的构造函数
        super().__init__(model_name_or_path, 
                device,
                num_cells,
                num_cells_in_search,
                pooler)

    def encode_mask(self, relations: Tensor,
                sentence: Union[str, List[str]], 
                k: int = 2,
                device: str = None, 
                return_numpy: bool = False,
                normalize_to_unit: bool = True,
                keepdim: bool = False,
                batch_size: int = 64,
                max_length: int = 128) -> Union[ndarray, Tensor]: 

        target_device = self.device if device is None else device
        self.model = self.model.to(target_device)
        
        single_sentence = False
        if isinstance(sentence, str):
            sentence = [sentence]
            single_sentence = True

        embedding_list = [] 
        with torch.no_grad():
            total_batch = len(sentence) // batch_size + (1 if len(sentence) % batch_size > 0 else 0)
            
            for batch_id in tqdm(range(total_batch)):
                inputs = self.tokenizer(
                    sentence[batch_id*batch_size:(batch_id+1)*batch_size], 
                    padding=True, 
                    truncation=True, 
                    max_length=max_length, 
                    return_tensors="pt"
                )
                tmp_relation = relations[batch_id*batch_size:(batch_id+1)*batch_size]
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
                outputs = self.model(**inputs, return_dict=True)
                embeddings = outputs.last_hidden_state
                if normalize_to_unit:
                    embeddings = embeddings / embeddings.norm(dim= -1, keepdim=True)
                
                similarity = cosine_similarity(tmp_relation, embeddings)
                similarity[:, 0] = float('-inf')
                # 提取相似度最高的两个索引
                _, top_two_indices = torch.topk(similarity, k, dim=1, largest=True, sorted=False)

                mask = torch.ones_like(embeddings, dtype=torch.int)
                for i in range(similarity.shape[0]):
                    mask[i, top_two_indices[i],:] = 0
                embeddings = embeddings * mask
                pooler_embeddings = self.model.pooler(embeddings)

                embedding_list.append(pooler_embeddings.cpu())
        embeddings = torch.cat(embedding_list, 0)
        
        if single_sentence and not keepdim:
            embeddings = embeddings[0]
        
        if return_numpy and not isinstance(embeddings, ndarray):
            return embeddings.numpy()
        return embeddings
            


                

model =  SimCSE_append("/group/40064/johnbli/bert-based-models/sup-simcse-roberta-large")

A = model.encode(["i love you","i love"])

model.encode_mask(A, ["i love you","i love"])                

                

