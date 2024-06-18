import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import json
import matplotlib.pyplot as plt
class CustomLLaMAModel(nn.Module):
    def __init__(self, model_name):
        super(CustomLLaMAModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)

    def forward(self, input_ids, attention_mask=None):
        # Forward pass through the model
        outputs = self.model(input_ids, attention_mask=attention_mask)
        # print(outputs.keys.items())
        # for (name, module) in self.model.named_modules():
        #     print(name)

        # Extract hidden states and attentions
        hidden_states = outputs.hidden_states  # List of hidden states from each layer
        attentions = outputs.attentions  # List of attention matrices from each layer

        # Extracting the output from each FFN and attention layer
        ffn_outputs = hidden_states  # All FFN layer outputs
        attention_outputs = attentions  # All attention layer outputs

        return ffn_outputs, attention_outputs
def plot(twoDtensor):
    plt.imshow((twoDtensor).numpy(), cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.title('2D Tensor')
    plt.show()

def build_batch_and_mask(example,type):
    # get some attribute
    Q=example["Q"]
    input=example[type+"_dic"][type]
    input_len=len(input)
    concat_len=len(Q)+input_len

    # print("input len:",input_len)
    # print("concat_len: ",concat_len)

    mask= torch.ones([input_len,concat_len])
    for i in range(input_len):
        for j in range(input_len-i-1):
            mask[i][concat_len-1-j]=0

    # build batch input
    Q = [item for sublist in (Q, input) for item in sublist]
    batch_inputs = torch.tensor(Q).unsqueeze(0).repeat(input_len,1)
    # print(batch_inputs)
    # print("mask.shape: ",mask.shape)
    # print("batch_inputs.shape: ",batch_inputs.shape)

    return batch_inputs, mask

def extract_token_rep(example, model, type):
    input = torch.tensor(example[type + "_dic"][type]["input_ids"])

    ffn_outputs, attention_outputs = model(input,attention_mask=torch.Tensor(example[type+"_dic"][type]["attention_mask"]))
    return ffn_outputs, attention_outputs

def extract_token_rep_withmask(example, model, type):
    Q_len=len(example["Q"])

    batch_inputs, attention_mask = build_batch_and_mask(example, type)
    plot(attention_mask)
    ffn_outputs, attention_outputs = model(batch_inputs, attention_mask=attention_mask)


    ffn_outputs=ffn_outputs[0].numpy()
    attention_outputs=attention_outputs[0].numpy()
    tgt_ffn_output=[]
    tgt_att_output=[]


    for idx in example[type+"_dic"]["coexit_list"]:
        tgt_ffn_output.append(ffn_outputs[idx,Q_len+idx,:])
        tgt_att_output.append(attention_outputs[idx,:,Q_len+idx,:])

    tgt_ffn_output=torch.tensor(tgt_ffn_output)
    tgt_att_output=torch.tensor(tgt_att_output)

    # print("ffn shape: ",tgt_ffn_output.shape)
    # print("att shape: ",tgt_att_output.shape)
    coexist_len = len(example[type + "_dic"]["coexit_list"])
    assert coexist_len==tgt_ffn_output.shape[0] and coexist_len==tgt_att_output.shape[0], "coexit len and output shape not match! "

    # for i in range(coexist_len):
    #     plot(tgt_att_output[i])

    return tgt_ffn_output,tgt_att_output


# Example usage:
model_name = "../gpt2"  # Replace with your LLaMA model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = CustomLLaMAModel(model_name)

# pos and neg list
pos_ffnlist=[]
pos_attlist=[]
neg_ffnlist=[]
neg_attlist=[]

with torch.no_grad():
    with open("llamaTruthx_dataset.json") as f:
        dic_list=json.load(f)
        # Forward pass to get the intermediate outputs
        # print(dic_list)
        for example in dic_list:

            pos_tgt_ffn_output,pos_tgt_att_output= extract_token_rep(example, model, "pos")
            # print(pos_tgt_ffn_output[-1])
            for i, ffn_output in enumerate(pos_tgt_ffn_output):
                for index in example["pos_dic"]["coexit_list"]:
                    # 需要reshape吗
                    if len(ffn_output.shape)==3:
                        pos_ffnlist.append(ffn_output[:,index,:].squeeze(0))
                    elif len(ffn_output.shape)==2:
                        pos_ffnlist.append(ffn_output[index, :])
            for i, attention_output in enumerate(pos_tgt_att_output):
                for index in example["pos_dic"]["coexit_list"]:
                    # 需要reshape吗
                    # print(index)
                    # plot(attention_output[:, :, index, :].squeeze(0))  # 结论：gpt2会mask掉后面的
                    pos_attlist.append(attention_output[:,:,index,:].reshape(-1))

            neg_tgt_ffn_output, neg_tgt_att_output = extract_token_rep(example, model, "neg")
            for i, ffn_output in enumerate(neg_tgt_ffn_output):
                for index in example["neg_dic"]["coexit_list"]:
                    # 需要reshape吗
                    if len(ffn_output.shape) == 3:
                        neg_ffnlist.append(ffn_output[:, index, :].squeeze(0))
                    elif len(ffn_output.shape) == 2:
                        neg_ffnlist.append(ffn_output[index, :])
            for i, attention_output in enumerate(neg_tgt_att_output):
                if i==1:
                    break
                for index in example["neg_dic"]["coexit_list"]:
                    # 需要reshape吗
                    # print(index)
                    # plot(attention_output[:, :, index, :].squeeze(0)) # 结论：gpt2会mask掉后面的
                    neg_attlist.append(attention_output[:, :, index, :].reshape(-1))

# print(neg_list)

# extract_token_rep(example,"neg")

# tgt_ffn_output,tgt_att_output= extract_token_rep_withmask(example, model, "pos")

# Print the shapes of the outputs
# print("Number of FFN layers:", len(tgt_ffn_output))
# for i, ffn_output in enumerate(tgt_ffn_output):
#     print(f"FFN Layer {i} Output Shape:", tgt_ffn_output.shape)
#
# print("\nNumber of Attention layers:", len(tgt_att_output))
# for i, attention_output in enumerate(tgt_att_output):
#     print(f"Attention Layer {i} Output Shape:", tgt_att_output.shape)


# ffn_output.shape:[batch_size, sentence_len, embedding_size]
# attention_output.shape: [batch_size, heads_number, sentence_length, attention_embedding_size]

# every collected output of attn layers is like: [1, heads_number, 1, attention_embedding_size]