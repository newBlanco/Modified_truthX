import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

llama2chat_with_truthx = "Llama_truthx"
tokenizer = AutoTokenizer.from_pretrained(
    llama2chat_with_truthx
)
model = AutoModelForCausalLM.from_pretrained(
    llama2chat_with_truthx,
    low_cpu_mem_usage=True
    # torch_dtype=torch.float16,
    # from_tf=True
)

question = "What are the benefits of eating an apple a day?"
encoded_inputs = tokenizer(question, return_tensors="pt")["input_ids"]
print(encoded_inputs)
outputs = model.generate(encoded_inputs.cuda())[0, encoded_inputs.shape[-1] :]
outputs_text = tokenizer.decode(outputs, skip_special_tokens=True).strip()
print(outputs_text)


# using TruthfulQA prompt
from llm import PROF_PRIMER as TRUTHFULQA_PROMPT

encoded_inputs = tokenizer(TRUTHFULQA_PROMPT.format(question), return_tensors="pt")[
    "input_ids"
]
encoded_inputs = tokenizer(question, return_tensors="pt")["input_ids"]
outputs = model.generate(encoded_inputs.cuda())[0, encoded_inputs.shape[-1] :]
outputs_text = (
    tokenizer.decode(outputs, skip_special_tokens=True).split("Q:")[0].strip()
)
print(outputs_text)
