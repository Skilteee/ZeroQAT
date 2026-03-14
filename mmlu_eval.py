import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import torch

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from langchain.prompts import PromptTemplate
import random
from utils import model_quantization
from quantize.utils import smooth_and_quant_temporary, smooth_and_quant_inplace

# 固定随机种子
def set_seed(seed: int = 42):
    random.seed(seed)                      # Python 随机数
    np.random.seed(seed)                   # Numpy 随机数
    torch.manual_seed(seed)                # CPU 上的随机数
    torch.cuda.manual_seed(seed)           # 当前 GPU
    torch.cuda.manual_seed_all(seed)       # 所有 GPU

seed = 42
set_seed(seed)


dataset = load_dataset("cais/mmlu", "all", split='test')

template = """Answer the following multiple choice question by giving the most appropriate response. Answer should be one among [A, B, C, D, E]

Question: {prompt}\n
A) {a}\n
B) {b}\n
C) {c}\n
D) {d}\n

Answer:"""

prompt = PromptTemplate(template=template, input_variables=['prompt', 'a', 'b', 'c', 'd'])

sample = dataset[0]

def format_text(example):
    text = prompt.format(prompt=example['question'], a=example['choices'][0], b=example['choices'][1], c=example['choices'][2], d=example['choices'][3],)
    return {"text": text}

def smooth_and_quant(model, model_name):
    is_llama = 'llama' in model_name.lower()
    layers = model.model.decoder.layers if not is_llama else model.model.layers
    for layer in layers:
        smooth_and_quant_inplace(layer, isllama=is_llama)


# model_id = "meta-llama/Llama-2-7b-hf"
model_id = "facebook/opt-6.7b"
# model_id = "huggyllama/llama-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    load_in_8bit=False,
    trust_remote_code=True
)

dataset = dataset.map(format_text)
# dataset = dataset.shuffle(seed=seed).select(range(200))

w_train_bits, a_train_bits = 4, 4
model, _ = model_quantization(model, model_id, w_train_bits, a_train_bits)
state = torch.load('/home/Qitao/project/ZO_quant_new/opt-6.7b-W4A4.pth')
print(model.load_state_dict(state, strict=False))
# omni_state = torch.load('/home/Qitao/project/ZO_quant_new/omni_parameters/Llama-2-7b-w4a4.pth')
# layers = model.model.layers if 'llama' in model_id.lower() else model.model.decoder.layers
# for i in range(len(layers)):
#     layers[i].load_state_dict(omni_state[i], strict=False)
model = model.cuda()
smooth_and_quant(model, model_id)






def get_ans(inputs, model):

    # pos_embeds = self.embed_positions(attention_mask, past_key_values_length)
    logits = model(input_ids=inputs['input_ids'].cuda(), attention_mask=inputs['attention_mask'].cuda()).logits[0, -1]

    # Create a list of tuples having (logit, 'option') format
    options_list = [(logits[tokenizer(' A').input_ids[-1]], 'A'), (logits[tokenizer(' B').input_ids[-1]], 'B'),
                    (logits[tokenizer(' C').input_ids[-1]], 'C'), (logits[tokenizer(' D').input_ids[-1]], 'D'),
                    (logits[tokenizer(' E').input_ids[-1]], 'E')]
    options_list = sorted(options_list, reverse=True)
    ans_list = []

    ans_list.append(options_list[0][1])

    return ans_list

actual_map = {0:'A', 1:'B', 2:'C', 3:'D'}
def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """

    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    actual = [actual_map[each] for each in actual]

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def build_5shot_prompt(example, subject_dataset):
    # 随机挑选 5 个示例作为 few-shot 示例
    # candidates = [ex for ex in subject_dataset if ex != example]
    few_shots = random.sample(subject_dataset, min(5, len(subject_dataset)))

    shots_text = ""
    for shot in few_shots:
        shot_prompt = template.format(
            prompt=shot['question'],
            a=shot['choices'][0],
            b=shot['choices'][1],
            c=shot['choices'][2],
            d=shot['choices'][3],
        )
        correct_letter = actual_map[shot['answer']]
        if len(shots_text) < 7000:
            shots_text += shot_prompt + f" {correct_letter}\n\n"


    # 构造 target prompt
    target_prompt = prompt.format(
        prompt=example['question'],
        a=example['choices'][0],
        b=example['choices'][1],
        c=example['choices'][2],
        d=example['choices'][3],
    )

    return shots_text + target_prompt

aps = []
data_by_subject = {}
subjects = set([ex['subject'] for ex in dataset])
for subject in subjects:
    data_by_subject[subject] = [ex for ex in dataset if ex['subject'] == subject]

bar = tqdm(enumerate(dataset), total=len(dataset))
with torch.no_grad():
    for i, data in bar:
        # subject_dataset = [ex for ex in dataset if ex["subject"] == data["subject"]]

        five_shot_prompt = build_5shot_prompt(data, data_by_subject[data['subject']])
        inputs = tokenizer(five_shot_prompt, return_tensors='pt')
        if inputs['input_ids'].shape[1] > 2048:
            continue

        ans_list = get_ans(inputs, model)
        average_precision = apk([data['answer']], ans_list, k=1)
        aps.append(average_precision)
    # ans1, ans2, ans3 = ans_list


mean_average_precision = np.mean(aps)

print(f"Mean Average Precision: {mean_average_precision}")
