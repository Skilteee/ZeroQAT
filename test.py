from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments
import os
import transformers
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import re
import sys
import datetime

from trainer_wiki import OurTrainer
from run import parse_args
from datautils import get_loaders
# from lm_eval import evaluator
from models.LMClass import LMClass
from transformers import TrainerCallback

from typing import List, Union

import numpy as np
from transformers import PreTrainedTokenizerBase
from safetensors.torch import load_file


class PrintRankCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        print(f"Local rank: {args.local_rank}, device: {args.device}")



# model_name = "huggyllama/llama-7b"
# model_name = "meta-llama/Llama-2-7b-hf"
model_name = 'meta-llama/Llama-2-7b-chat-hf'
# model_name = "ChenMnZ/Llama-2-7b-EfficientQAT-w2g128"
# model_name = './vla-llama'
# model_name = "facebook/opt-6.7b"

model_family = re.findall(r"/(.*?)-", model_name)[0]
model_nick_name = model_name.split("/")[-1]
is_llama = 'llama' in model_name.lower()





tokenizer = AutoTokenizer.from_pretrained(model_name)
if is_llama:
    model = transformers.LlamaForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.float16,
                                                 # device_map="auto",
                                                          )
else:
    model = transformers.OPTForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.float16,
                                                 # device_map="auto",
                                                 )

# state = load_file('/home/Qitao/.cache/huggingface/hub/models--ChenMnZ--Llama-2-7b-EfficientQAT-w2g128/snapshots/a980167e4558094f5a295e4c3e8d6d5203d49e24/model.safetensors')
# state.pop('model.embed_tokens.weight')
# state.pop('lm_head.weight')
# new_state = {}
# for key in state:
#     if 'weight' in key:
#         new_state[key.replace('qweight', 'weight')] = state[key]

args = parse_args()
args.trainer = 'zo'
# # llama2-7b W2A16
# args.zo_eps = 1e-4
# args.learning_rate = 1e-8
# # opt-13b
args.zo_eps = 5e-4
args.learning_rate = 5e-7
# # opt-6.7b
# args.zo_eps = 5e-4
# args.learning_rate = 5e-7
# # opt-2.7b W2A16
# args.zo_eps = 1e-4
# args.learning_rate = 1e-8
# # opt-2.7b
# args.zo_eps = 5e-4
# args.learning_rate = 5e-7
# # llama1-13b W2A16
# args.zo_eps = 5e-4
# args.learning_rate = 1e-6
# # # llama1-13b W4A4
# args.zo_eps = 1e-4
# args.learning_rate = 1e-8
# # llama1-7b
# args.zo_eps = 5e-4
# args.learning_rate = 5e-7
args.enhanced = 'zo'
args.cache_dir = './cache'
args.model = model_name
args.net = model_name
args.model_nick_name = model_nick_name
args.model_family = model_family
args.max_steps = 12000
args.calib_dataset = 'wikitext2'
# args.calib_dataset = 'c4'
args.nsamples = 128
args.isllama = 'llama' in args.model.lower()
model.seqlen = 2048 # 2048
# args.nsamples = int(args.nsamples * (2048 / model.seqlen))
args.w_train_bits = 6
args.a_train_bits = 6
args.eval_few_shot = False
args.tasks = 'piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande'
args.per_device_train_batch_size = 4
# args.lr_scheduler_type = "cosine"
# args.warmup_steps = 100
args.seed = 4
args.resume = None
args.num_steps_before_decay = [4000]
# args.resume = '/home/Qitao/project/ZO_quant_new/qat_init_parameters/llama1-7b.pth'
# args.tasks = 'piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande'

# from transformers import AutoModelForVision2Seq, AutoProcessor
# processor = AutoProcessor.from_pretrained('openvla/openvla-7b', trust_remote_code=True)
# vla = AutoModelForVision2Seq.from_pretrained(
#         'openvla/openvla-7b',
#         torch_dtype=torch.float16,
#         low_cpu_mem_usage=True,
#         trust_remote_code=True,
#     )
# tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
# action_tokenizer = ActionTokenizer(tokenizer)
# lm = vla.language_model




# state = torch.load('state.pth')
# state_model = model.state_dict()
# for each in state:
#     try:
#         print(each, (state[each] - state_model[each]).abs().norm() / state_model[each].abs().norm())
#     except:
#         continue
#     input()
# import code
# code.interact(local=locals())
# state.pop('lm_head.weight')
# state.pop('model.embed_tokens.weight')
# model.load_state_dict(state, strict=False)


cache_dataloader = f'{args.cache_dir}/dataloader_{args.model_family}_{args.calib_dataset}_{args.nsamples}_{model.seqlen}_{args.seed}.cache'
if os.path.exists(cache_dataloader):
    dataloader = torch.load(cache_dataloader)
else:
    dataloader, _ = get_loaders(
        args.calib_dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
    )
    torch.save(dataloader, cache_dataloader)

all_input_ids = []
all_attention_masks = []

for input_ids, attention_mask in dataloader:
    all_input_ids.extend(input_ids.tolist())
    all_attention_masks.extend(torch.ones(input_ids.shape, dtype=torch.long).tolist())
    # all_attention_masks.extend(attention_mask.tolist())

hf_dataset = Dataset.from_dict({
    "input_ids": all_input_ids,
    "attention_mask": all_attention_masks
})

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# 日志文件名按时间生成
log_file = os.path.join(log_dir, model_nick_name + f"-W{args.w_train_bits}A{args.a_train_bits}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

class Logger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 重定向 stdout 和 stderr
sys.stdout = Logger(log_file)
sys.stderr = sys.stdout

# 确保有 pad_token（GPT2 没有默认的）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4. 创建 data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = OurTrainer(
    model=model,
    args=args,
    train_dataset=hf_dataset,
    data_collator=data_collator,
    callbacks=[PrintRankCallback()]
)

trainer.train()

# if args.eval_few_shot:
#
#     lm = LMClass(args, tokenizer, model)
#     lm.seqlen = 2048
#
#     t_results = evaluator.simple_evaluate(
#         lm,
#         tasks=args.tasks,
#         num_fewshot=0,
#         limit=None,
#     )
#
#     print(t_results)



# evaluate(model, args)

