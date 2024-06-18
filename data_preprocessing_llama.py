import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import numpy
llama2chat_with_truthx = "../gpt2"
tokenizer = AutoTokenizer.from_pretrained(
    llama2chat_with_truthx
)
TQ_dataset= pd.read_csv("./test_QA.csv")

dataset=[]
# print(tokenizer.sep_token)
# tokenizer.add_tokens("[SEP]")
# tokenizer.sep_token_id = "[SEP]"

max_len= 0

for index in range(TQ_dataset.shape[0]):
    item = TQ_dataset.iloc[index]
    print(len(tokenizer(item["Question"] + "." + item["Incorrect Answers"], return_tensors="np")["input_ids"].squeeze(0)))
    best_len= len(tokenizer(item["Question"] + "." + item["Best Answer"], return_tensors="np")["input_ids"].squeeze(0))
    inc_len= len(tokenizer(item["Question"] + "." + item["Incorrect Answers"], return_tensors="np")["input_ids"].squeeze(0))
    if best_len>inc_len:
        if best_len>max_len:
            max_len=best_len
    else:
        if inc_len>max_len:
            max_len=inc_len

print(max_len)
for index in range(TQ_dataset.shape[0]):
    tmp_dic={
            "pos_dic":{"coexit_list":[]},
            "neg_dic":{"coexit_list":[]}
            }
    item= TQ_dataset.iloc[index]
    # 用llama分词法切分句子并转换为input index
    # print(tokenizer(item["Question"], return_tensors="pt"))
    Q_len= len(tokenizer(item["Question"]+ ".", return_tensors="pt")["input_ids"].squeeze(0))
    pos_total_len=len(tokenizer(item["Question"]+ "."+item["Best Answer"], return_tensors="pt")["input_ids"].squeeze(0))
    neg_total_len=len(tokenizer(item["Question"]+ "."+item["Incorrect Answers"], return_tensors="pt")["input_ids"].squeeze(0))


    pos_tokenized_dic=tokenizer(item["Question"]+ "."+item["Best Answer"], \
                                return_tensors="pt",padding="max_length" ,max_length=max_len).data
    print(pos_tokenized_dic)
    # pos sample preprocessing
    pos_tokenized_dic["input_ids"]=pos_tokenized_dic["input_ids"].squeeze(0).tolist()
    pos_tokenized_dic["attention_mask"]=pos_tokenized_dic["attention_mask"].squeeze(0).tolist()
    # print(pos_tokenized_dic["attention_mask"])
    tmp_dic["pos_dic"]["pos"]= pos_tokenized_dic

    # neg sample preprocessing
    neg_tokenized_dic=tokenizer(item["Question"]+ "."+item["Incorrect Answers"], \
                                     return_tensors="pt",padding="max_length" ,max_length=max_len).data
    neg_tokenized_dic["input_ids"]= neg_tokenized_dic["input_ids"].squeeze(0).tolist()
    # print(len(neg_tokenized_dic["input_ids"].squeeze(0).tolist()))
    neg_tokenized_dic["attention_mask"]= neg_tokenized_dic["attention_mask"].squeeze(0).tolist()
    tmp_dic["neg_dic"]["neg"]= neg_tokenized_dic

    # 找到当前Best和Incorrect中分词列表中相同词的位置
    for i, pos in enumerate(tmp_dic["pos_dic"]["pos"]["input_ids"]):
        if i < Q_len or i>=pos_total_len:
            continue
        for j, neg in enumerate(tmp_dic["neg_dic"]["neg"]["input_ids"]):
            if j<Q_len or j>=neg_total_len:
                continue
            # 只要一样就加进去，不管是否有一方已经出现过（不可能两方都出现过），这样在对比学习中是不同的“一对”样本
            if pos==neg:
                tmp_dic["pos_dic"]["coexit_list"].append(i)
                tmp_dic["neg_dic"]["coexit_list"].append(j)
    dataset.append(tmp_dic)

# print(dataset)
with open("llamaTruthx_dataset.json",'w') as f:
    json.dump(dataset,f, indent=1,sort_keys=True)