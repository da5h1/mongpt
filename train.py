import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Dict, List, Set
import random
import copy
import math
from argparse import ArgumentParser
import numpy as np
import bert_score
import sys
import time
start_time = time.time()
error_file = open('logs/output.err', 'w')
sys.stderr = error_file
random.seed(100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
N_EPOCHS: int = 15
print("nepochs:", N_EPOCHS)
save_model = False
print("save model:", save_model)
prompt_tuning = False
print("prompt tuning:", prompt_tuning)

parser = ArgumentParser()
parser.add_argument('--lr', dest='learning_rate', type=float, required=False, default=1e-5,
                    help='The learning rate for LLM training.')
parser.add_argument('--minibatch', dest='minibatch_size', type=int, required=False, default=4,#изменить
                    help='The mini-batch size for LLM training and inference.')
parser.add_argument('--accumulate', dest='accumulate_gradients', type=int, default=1,#изменить
                    required=False, help='The accumulate gradients iteration size.')
args = parser.parse_args()

def load_model():
    model_path = "ai-forever/mGPT-1.3B-mongol"
    print("model path:", model_path)
    tokenizer_path = "ai-forever/mGPT-1.3B-mongol"
    print("tokenizer path:", tokenizer_path)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(model_path).cuda()

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size:', size_all_mb)
    return tokenizer, model

def load_dolly_dataset(dataset_name: str, shuffle: bool = False):
    res = dict()
    with open(dataset_name, "r", encoding="utf-8") as f:
        ds = json.load(f)
    if shuffle:
        random.shuffle(ds)
    categories = list(set([ds[i]["category"] for i in range(len(ds))]))
    for category in categories:
        res[category] = []
    for sample in ds:
        assert sample["instruction"] != ""
        assert sample["response"] != ""
        res[sample["category"]].append((
            ' '.join(sample["instruction"].split()).strip(),
            ' '.join(sample["context"].split()).strip(),
            ' '.join(sample["response"].split()).strip()
        ))
    return res

def train_val_split(ds: dict, truncate: float):
    train = {}
    val = {}
    categories = sorted(list(ds.keys()))
    for category in categories:
        train[category] = ds[category][:round(truncate * len(ds[category]))]
        val[category] = ds[category][round(truncate * len(ds[category])):]
        assert train[category] != []
        assert val[category] != []
    return train, val

def cut_ds(ds:dict, lim: int = 100):
    new_ds = {}
    categories = sorted(list(ds.keys()))
    for category in categories:
        new_ds[category] = []
    i = 0
    l = 0
    while l < lim:
        for category in categories:
            try:
                new_ds[category].append(ds[category][i])
            except:
                continue
            l += 1
            if l >= lim:
                break
        i += 1    
    return new_ds

def decode_input_ids(input_ids: list, tokenizer: AutoTokenizer):
    clear_input_ids = []
    for item in input_ids:
        if item != 2:
            clear_input_ids.append(item)
    return tokenizer.batch_decode([clear_input_ids], skip_special_tokens=True)

def decode_labels(labels: list, tokenizer: AutoTokenizer):
    clear_labels = []
    for item in labels:
        if item != -100:
            clear_labels.append(item)
    return [tokenizer.batch_decode([clear_labels], skip_special_tokens=True)[0].strip()]
            
def get_ds_size(ds: dict):
    l = 0
    for category in sorted(list(ds.keys())):
        l += len(ds[category])
    return l

def format_dolly(sample, is_instruction_first: bool):
    instruction = f"{sample[0]}".strip()
    context = f"{sample[1]}".strip()
    if is_instruction_first:
        prompt = instruction + " " + context
    else:
        prompt = context + " " + instruction
    prompt = " ".join(prompt.split()).strip()
    assert len(prompt) != 0
    if len(sample) > 2:
        response = f'{sample[2].strip()}'
        assert len(response) != 0
        prompt += (" " + response)
    return prompt

def tokenize_prompt(prompt: str, tokenizer: AutoTokenizer, add_eos_token: bool = True,
                    add_labels: bool = True) -> dict[str, list[int]]:
    result = tokenizer(prompt, padding=False, return_tensors=None)
    if (result['input_ids'][-1] != tokenizer.eos_token_id and add_eos_token):
        result['input_ids'].append(tokenizer.eos_token_id)
        result['attention_mask'].append(1)
    if add_labels:
        result['labels'] = result['input_ids'].copy()
    return result
    
def generate_and_tokenize_prompt(data_point: tuple[str, str, str], is_instruction_first: bool, 
                                 tokenizer: AutoTokenizer):
    full_prompt = format_dolly(data_point, is_instruction_first)
    tokenized_full_prompt = tokenize_prompt(full_prompt, tokenizer)
    user_prompt = format_dolly(data_point[0:2], is_instruction_first)
    tokenized_user_prompt = tokenize_prompt(user_prompt, tokenizer)
    user_prompt_len = len(tokenized_user_prompt["input_ids"])
    user_prompt_len -= 1
    tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
    return tokenized_full_prompt

def filter_by_max_seq_len(dataset: dict, max_seq_len: int, tokenizer: AutoTokenizer):
    lens = []
    for category in sorted(list(dataset.keys())):
        filtered = []
        for sample in dataset[category]:
            res = generate_and_tokenize_prompt(sample, True, tokenizer)
            lens.append(len(res["input_ids"]))
            if len(res["input_ids"]) <= max_seq_len:
                filtered.append(sample)
            del res
        del dataset[category]
        dataset[category] = filtered
        del filtered
    return dataset, lens

def tokenize_dataset(dataset: dict, tokenizer: AutoTokenizer):
    tokenized = dict()
    for category in sorted(list(dataset.keys())):
        tokenized_subset = []
        for sample in dataset[category]:
            tokenized_subset.append(generate_and_tokenize_prompt(sample, True, tokenizer))
            #if len(sample[1]) > 0:
            #    tokenized_subset.append(generate_and_tokenize_prompt(sample, False, tokenizer))
        tokenized[category] = tokenized_subset
        del tokenized_subset
    return tokenized
  
def generate_minibatch(dataset1: dict, categories: list[str], minibatch: int, padding: int, iter_idx: int):
    if minibatch < len(categories):
        environments = random.sample(population = categories, k = minibatch)
    elif minibatch == len(categories):
        environments = categories
    else:
        environments = copy.copy(categories)
        while len(environments) < minibatch:
            environments.append(random.choice(categories))
    input_ids = []
    attention_mask = []
    labels = []
    for idx, env in enumerate(environments):
        sample = random.choice(dataset1[env])
        input_ids.append(torch.tensor(data=sample["input_ids"], dtype=torch.long))
        attention_mask.append(torch.tensor(data=sample["attention_mask"], dtype=torch.long))
        labels.append(torch.tensor(data=sample["labels"], dtype=torch.long))
    batched_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=padding).cuda()
    batched_attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0).cuda()
    batched_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100).cuda()
    assert len(batched_input_ids) == len(batched_attention_mask) and len(batched_attention_mask) == len(batched_labels)
    return batched_input_ids, batched_attention_mask, batched_labels

def predict(testset: list, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, minibatch: int):
    gen_config = model.generation_config
    gen_config.max_new_tokens = 300
    gen_config.min_new_tokens = 1
    #gen_config.no_repeat_ngram_size = 5
    #gen_config.do_sample = True
    #gen_config.top_k = 6
    #gen_config.top_p = 0.2311
    gen_config.num_return_sequences = 1
    gen_config.pad_token_id = tokenizer.eos_token_id
    gen_config.eos_token_id = tokenizer.eos_token_id
    n_batches = math.ceil(len(testset) / minibatch)
    true_answers = []
    predicted_answers = []
    input_prompts = []
    for batch_idx in range(n_batches):
        batch_start = batch_idx * minibatch
        batch_end = min(len(testset), batch_start + minibatch)
        input_ids = []
        attention_mask = []
        for sample_idx in range(batch_start, batch_end):
            prompt = format_dolly(testset[sample_idx][0:2], True)
            tokenized_text = tokenize_prompt(prompt, tokenizer, add_labels = False)
            input_ids.append(torch.tensor(data=tokenized_text["input_ids"], dtype=torch.long))
            attention_mask.append(torch.tensor(data=tokenized_text["attention_mask"], dtype=torch.long))
            true_answers.append(testset[sample_idx][2])

        batched_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id).cuda()
        batched_attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True,padding_value=0).cuda()
        with torch.no_grad():
            generated_ids = model.generate(input_ids = batched_input_ids, attention_mask = batched_attention_mask, generation_config=gen_config)
        predicted_answers += tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        input_prompts += [cur.strip() for cur in tokenizer.batch_decode(input_ids, skip_special_tokens=True)]
        del input_ids, attention_mask
        del batched_input_ids, batched_attention_mask, generated_ids
    assert len(predicted_answers) == len(testset)
    assert len(true_answers) == len(testset)
    for idx in range(len(testset)):
        assert len(predicted_answers[idx]) >= len(input_prompts[idx])
        assert predicted_answers[idx].startswith(input_prompts[idx])
        predicted_answers[idx] = ' '.join(predicted_answers[idx][len(input_prompts[idx]):].split()).strip() 
    return input_prompts, predicted_answers, true_answers

def evaluate(questions: list[str], predicted_answers: list[str], true_answers: list[str], minibatch: int):
    assert len(true_answers) == len(predicted_answers)
    f1_list = []
    _, _, F1 = bert_score.score(predicted_answers, true_answers, 
                                model_type="xlm-roberta-large", 
                                num_layers = 17, batch_size = args.minibatch_size)
    for x in F1:
        f1_list.append(x.item())
    assert len(true_answers) == len(f1_list)
    f1_mean = float(np.mean(f1_list))
    res = []
    for pred, true, f1_val, question in zip(predicted_answers, true_answers, f1_list, questions):
        res.append({
            "PROMPT": question,
            "TRUE": true,
            "PRED": pred,
            "F1": f1_val
        })
    return f1_mean, sorted(res, key=lambda it: it["F1"])

def add_prompt_tuning(ds: dict, prompt_start: str):
    new_ds = {}
    for category in sorted(list(ds.keys())):
        new_ds[category] = []
        for item in ds[category]:
            new_ds[category].append((prompt_start+" "+item[0], item[1], item[2]))
    return new_ds
            

tokenizer, model = load_model()

ds = load_dolly_dataset("/userspace/cdd/dolly_translation/ds_for_training_mon.json")
ds, lens = filter_by_max_seq_len(ds, 545, tokenizer)
if prompt_tuning:
    mon_prompt_start = "Даалгаврын хариу бичих. Монголоор зөв бичээрэй. Та өөрийгөө Монголын зохиолч Сэнгийн Эрдэнэ гэж төсөөлөөд үз дээ. Кирилл үсгээр бичнэ."
    ds = add_prompt_tuning(ds, mon_prompt_start)
train_ds, val_ds = train_val_split(ds, truncate = 0.9)
val_ds, test_ds = train_val_split(val_ds, truncate = 0.5)

with open("/userspace/cdd/val_ds.json", "w", encoding = "utf-8") as val_file:
    json.dump(val_ds, val_file, ensure_ascii=False)
with open("/userspace/cdd/test_ds.json", "w", encoding = "utf-8") as test_file:
    json.dump(test_ds, test_file, ensure_ascii=False)

tokenized_train_ds = tokenize_dataset(dataset = train_ds, tokenizer = tokenizer)
print("0.95 prompts smaller than", np.percentile(lens, 95))
print("len of train dataset:", get_ds_size(train_ds))
print("len of val dataset:", get_ds_size(val_ds))

all_categories = sorted(list(train_ds.keys()))
n_samples_for_training = len(train_ds[all_categories[0]])
for category in all_categories[1:]:
    n_samples_for_training += len(train_ds[category])
n_training_batches = int(np.ceil(n_samples_for_training / args.minibatch_size))
accumulate_gradients = args.accumulate_gradients
if accumulate_gradients > 1:
    while (n_training_batches % accumulate_gradients) != 0:
        n_training_batches += 1
print("accumulate_gradients", accumulate_gradients)
print("batch size:", args.minibatch_size)
print('nbatches:', n_training_batches)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.learning_rate,
)     

#number_of_bur_samples_in_batch = []
#min_border = 1
#max_border = (2/3)*args.minibatch_size 
#lin_space = np.linspace(min_border, max_border, N_EPOCHS)
#for x in lin_space:
#    number_of_bur_samples_in_batch.append(round(x))
#print("number of buryat samples in batch per epoch:", number_of_bur_samples_in_batch) 
   
scaler = torch.cuda.amp.GradScaler(enabled=True)

best_scores = []
avg_scores = []
worst_scores = []

model.eval()
validation_questions = []
validation_true = []
validation_pred = []
for category in sorted(list(val_ds.keys())):
    prompt_, pred_, true_ = predict(val_ds[category], tokenizer, model, args.minibatch_size)
    validation_pred += pred_
    validation_true += true_
    validation_questions += prompt_
    del pred_, true_, prompt_
new_f1, detailed_validation_report = evaluate(validation_questions, validation_pred, validation_true, args.minibatch_size)
average_example = min(detailed_validation_report, key=lambda it: abs(it["F1"] - new_f1))
print(f'initial validation BERT F1 = {new_f1}.')
print("average example:")
print(average_example)
print("3 best examples:")
print(detailed_validation_report[-3:])
print("3 worst examples:")
print(detailed_validation_report[:3])
best_f1 = new_f1
best_scores.append(detailed_validation_report[-1]["F1"])
avg_scores.append(new_f1)
worst_scores.append(detailed_validation_report[0]["F1"])
del detailed_validation_report, validation_pred, validation_true

for epoch in range(1, N_EPOCHS + 1):
    print(f"epoch {epoch} is started")
    total_training_loss_val = 0.0
    training_fct_loss_val = 0.0
    weight_norm_val = 0.0
    training_penalty_val = 0.0
    model.train()
    for iter_idx in range(1, n_training_batches + 1):
        input_ids, attention_mask, labels = generate_minibatch(
            dataset1 = tokenized_train_ds,
            categories = all_categories,
            minibatch = args.minibatch_size,
            padding = tokenizer.pad_token_id,
            iter_idx = iter_idx
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            try:
                res = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels, return_dict = True)
            except Exception as e:
                print(e)
                continue
        loss = res.loss
        instant_loss = loss.detach().cpu().float().numpy()
        if iter_idx % 100 == 0:
            print(instant_loss)
        total_training_loss_val += instant_loss
        if accumulate_gradients > 1:
            loss = loss / accumulate_gradients
        scaler.scale(loss).backward()
        if accumulate_gradients > 1:
            if iter_idx % accumulate_gradients == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        del input_ids, attention_mask, labels, res
    total_training_loss_val /= float(n_training_batches / accumulate_gradients)
    training_fct_loss_val /= float(n_training_batches / accumulate_gradients)
    weight_norm_val /= float(n_training_batches / accumulate_gradients)
    training_penalty_val /= float(n_training_batches / accumulate_gradients)
    print(f'Epoch {epoch}: training loss is {total_training_loss_val}.')

    model.eval()
    validation_questions = []
    validation_true = []
    validation_pred = []
    for category in sorted(list(val_ds.keys())):
        prompt_, pred_, true_ = predict(val_ds[category], tokenizer, model, args.minibatch_size)
        validation_pred += pred_
        validation_true += true_
        validation_questions += prompt_
        del pred_, true_, prompt_
    new_f1, detailed_validation_report = evaluate(validation_questions, validation_pred, validation_true, args.minibatch_size)
    average_example = min(detailed_validation_report, key=lambda it: abs(it["F1"] - new_f1))
    print(f'Epoch {epoch}: validation BERT F1 = {new_f1}.')
    print("average example:")
    print(average_example)
    print("3 best examples:")
    print(detailed_validation_report[-3:])
    print("3 worst examples:")
    print(detailed_validation_report[:3])
    best_scores.append(detailed_validation_report[-1]["F1"])
    avg_scores.append(new_f1)
    worst_scores.append(detailed_validation_report[0]["F1"])
    del detailed_validation_report, validation_pred, validation_true
    if new_f1 > best_f1:
        best_f1 = new_f1
        if save_model:
            model.save_pretrained("saved_model_prompt_tuning")
        print(f'new best F1 = {best_f1}.')
    print(f'Epoch {epoch} is finished.')
print("best scores")
print(best_scores)
print("mean scores")
print(avg_scores)
print("worst scores")
print(worst_scores)
end_time = time.time()
print(start_time - end_time)

        

