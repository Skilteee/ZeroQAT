from datasets import load_dataset
import os
from trainer import OurTrainer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dataset = load_dataset("yahma/alpaca-cleaned")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Choose a base model (LLaMA-2 or Mistral)
model_name = "meta-llama/Llama-2-7b-chat-hf"  # Or "mistralai/Mistral-7B-v0.1"
# model_name = "huggyllama/llama-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if "llama" in model_name:
    tokenizer.add_special_tokens(
        {
            "eos_token": "[PAD]",
            "bos_token": "</s>",
            "unk_token": "</s>",
            "pad_token": "</s>",
        }
    )
else:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             torch_dtype=torch.float16,
                                             device_map="auto")

from transformers import TrainingArguments

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=1e-7,
    num_train_epochs=3,
    save_strategy="epoch",
    output_dir="./output",
    logging_dir="./logs",
    evaluation_strategy="no",
    fp16=True,  # 仍然使用 fp16
    fp16_full_eval=True,
    max_grad_norm=1.0,  # 限制梯度大小，防止梯度爆炸
    save_total_limit=2,
    remove_unused_columns=False,
)


def preprocess_function(examples):
    # 构造 Prompt
    prompts = [
        f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
        for instruction, input_text in zip(examples["instruction"], examples["input"])
    ]

    # Tokenize
    model_inputs = tokenizer(prompts, padding="max_length", truncation=True, max_length=512)

    # Tokenize output (labels)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["output"], padding="max_length", truncation=True, max_length=512)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# 处理数据集
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# import code
# code.interact(local=locals())

from run import parse_args
args = parse_args()
args.trainer = 'zo'
args.lr = 1e-6
# args.enhanced = 'zo'

from fastchat.llm_judge import MTBench
#
# 评测 MT-Bench 任务
bench = MTBench()
score = bench.evaluate(model, tokenizer)
# import code
# code.interact(local=locals())
trainer = OurTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"]
)
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"]
# )

trainer.train()