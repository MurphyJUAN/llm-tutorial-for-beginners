# %%
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
from typing import Dict, List
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import Trainer
device = 'cuda' if torch.cuda.is_available() else 'cpu'   
# %%  
"""
Model Initialization
"""
tokenizer = AutoTokenizer.from_pretrained('NousResearch/Llama-2-7b-hf')
# %%
# TODO: 下面參數可以自己調整
hidden_size = 256
intermediate_size = (int(hidden_size * 8/3 / 128) + 1) * 128

config = AutoConfig.for_model(
    model_type="llama",
    hidden_size=hidden_size,
    intermediate_size=intermediate_size,
    num_attention_heads=16,
    num_hidden_layers=4,
    num_key_value_heads=8
)

# print 出 config 觀察看看
print(config)
# %%
model = AutoModelForCausalLM.from_config(
    config,
    torch_dtype=torch.float32
).to(device)

# print 出 model 架構觀察看看
print(model)
# %%
# print 出 model 的每一層以及其參數大小
def print_model_parameters(model):
    print("Layer Name & Parameters")
    print("----------------------------")
    total_params = 0
    for name, parameter in model.named_parameters():
        param_size = parameter.size()
        param_count = torch.prod(torch.tensor(param_size)).item()
        total_params += param_count
        print(f"{name:50} | Size: {str(param_size):30} | Count: {str(param_count):20}")
    print("----------------------------")
    print(f"Total Parameters: {total_params} ({total_params / 1000000:.1f} M)")

# 觀察看看這個模型總共的參數量是 _____ M 呢 ?
print_model_parameters(model)

# %%
# 使用 Kaiming 初始化模型參數
def kaiming_initialization(model):
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            torch.nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='leaky_relu')
        elif 'bias' in name:
            # 一般偏置项可以初始化为 0
            torch.nn.init.constant_(param, 0)

kaiming_initialization(model)
# %%
# Inference function (使用 topp sampling)
def inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_text: str = "從前從前，",
    max_new_tokens: int = 16
):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    # TODO: 設定 decoding strategy
    outputs = model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=40,
        top_p=0.95,
        temperature=0.8
    )
    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
    # print(outputs)
    print(generated_text)

# 目前模型是隨機intial的，所以預期模型生成的文字會胡言亂語，顛三倒四，但可以先確認模型是可以運作的
inference(model, tokenizer)
# %%
"""
Dataset Setup
"""
dataset_name_or_path = 'adam89/TinyStoriesChinese'
# 使用所有 trainset 下去訓練
# ds_train = load_dataset(dataset_name_or_path, split='train')
# 只用 10% 的 trainset 下去訓練，可自行調整
ds_train = load_dataset(dataset_name_or_path, split='train[:10%]')
ds_val = load_dataset('dataset_name_or_path', split='validation')
#%%
# TODO: 將 training 和 validation data 翻譯成繁體中文
"""
from googletrans import Translator
data = {
    "story_zh": "莉莉和本是朋友。他们喜欢在公园里玩。"
}
translator = Translator()
translated_story = translator.translate(data['story_zh'], src='zh-CN', dest='zh-TW')
"""
#%%
# TODO: 要根據新的 dataset 的格式來修改
def process_func(
    examples: Dict[str, List]
) -> Dict[str, List]:
    max_token = 2048    

    encoded_texts = tokenizer(examples['story_zh'], add_special_tokens=False)
    input_ids_list = encoded_texts['input_ids']

    new_input_ids_list, new_attn_mask_list = [], []
    for input_ids in input_ids_list:
        temp = input_ids[-max_token+1:] + [tokenizer.eos_token_id]
        new_input_ids_list.append(temp)
        new_attn_mask_list.append([1] * len(temp))
    return {
        'input_ids': new_input_ids_list,
        'attention_mask': new_attn_mask_list
    }

# %%
num_proc = 8  
ds_train = ds_train.shuffle()   
ds_train = ds_train.map(
    process_func,
    batched=True,
    num_proc=num_proc,
    remove_columns=ds_train.column_names,
    desc='Running tokenizer on train_set: '
)
ds_val = ds_val.map(
    process_func,
    batched=True,
    num_proc=num_proc,
    remove_columns=ds_val.column_names,
    desc='Running tokenizer on val_set: '
)

print(ds_train)
print(ds_val)
# %%
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
# %%
"""
開始訓練
"""
# %%
# TODO: 設定你的訓練參數
batch_size = 4
gradient_accumulation_steps = 1
epochs = 3

training_args = TrainingArguments(
    output_dir='saves',
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    eval_steps=1000,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=1e-4,
    lr_scheduler_type='cosine',
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    logging_steps=50,
    report_to=None,
    num_train_epochs=epochs,
    save_steps=1000,
    save_total_limit=2,
    seed=3407
)
# %%
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
trainer.train()
# %%
# 訓練好了，來淺試一下吧
inference(
    model,
    tokenizer,
    "從前，有一個小女孩在森林撿到一隻小兔子，"
)
# %%
model.save_pretrain('my_model')