import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Dict, List, Set
import random
import math
from argparse import ArgumentParser
import numpy as np
import logging
import bert_score
import sys

import time
print(time.__version__)

error_file = open('logs/output.err', 'w')
sys.stderr = error_file
random.seed(100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

llm_training_logger = logging.getLogger(__name__)
parser = ArgumentParser()
parser.add_argument('--minibatch', dest='minibatch_size', type=int, required=False, default=4,#изменить
                    help='The mini-batch size for LLM training and inference.')
args = parser.parse_args()

def load_model():
    model_path = "/userspace/cdd/saved_model"
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

def predict(testset: list, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, minibatch: int, min_new_tokens, no_repeat_ngram_size, do_sample, top_k, top_p):
    gen_config = model.generation_config
    gen_config.max_new_tokens = 300
    gen_config.min_new_tokens = min_new_tokens
    gen_config.no_repeat_ngram_size = no_repeat_ngram_size
    gen_config.do_sample = do_sample
    gen_config.top_k = top_k
    gen_config.top_p = top_p
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
    return f1_mean, sorted(res, reverse = True, key=lambda it: it["F1"]), min(res, key=lambda it: abs(it["F1"] - f1_mean))
    
tokenizer, model = load_model()

with open("/userspace/cdd/val_ds.json", 'r') as val_file:
    val_ds = json.load(val_file)

with open("/userspace/cdd/test_ds.json", 'r') as test_file:
    test_ds = json.load(test_file)
    
best_f1 = 0.0
model.eval()

test = True

if not test:
    for index in range(100):
        print("--------------", index)
        min_new_tokens = random.choices([0, 1, 2], [0.2, 0.6, 0.2], k = 1)[0]
        no_repeat_ngram_size = random.choices([0, 3, 4, 5], [0.2, 0.2, 0.3, 0.3], k = 1)[0]
        do_sample = random.choices([True, False], [0.7, 0.3], k = 1)[0]
        top_k = random.choice(range(11))
        top_p = random.random()
        print("min_new_tokens, no_repeat_ngram_size, do_sample, top_k, top_p")
        print(min_new_tokens, no_repeat_ngram_size, do_sample, top_k, top_p)
        validation_questions = []
        validation_true = []
        validation_pred = []
        for category in sorted(list(val_ds.keys())):
            prompt_, pred_, true_ = predict(val_ds[category], tokenizer, model, args.minibatch_size, min_new_tokens, no_repeat_ngram_size, do_sample, top_k, top_p)
            validation_pred += pred_
            validation_true += true_
            validation_questions += prompt_
            del pred_, true_, prompt_
        new_f1, detailed_validation_report, average_example = evaluate(validation_questions, validation_pred, validation_true, args.minibatch_size)
        print(f'BERT F1 = {new_f1}.')
        print("average example:")
        print(average_example)
        print("3 best examples:")
        print(detailed_validation_report[:3])
        print("3 worst examples:")
        print(sorted(detailed_validation_report, reverse = False, key=lambda it: it["F1"])[:3])
        if new_f1 > best_f1:
            best_f1 = new_f1
            print(f'new best F1 = {best_f1}.')
        del detailed_validation_report, validation_pred, validation_true
else:
    print("TEST")
    min_new_tokens = 2
    no_repeat_ngram_size = 0
    do_sample = False
    top_k = 5
    top_p = 0.5
    validation_questions = []
    validation_true = []
    validation_pred = []
    for category in sorted(list(test_ds.keys())):
        prompt_, pred_, true_ = predict(val_ds[category], tokenizer, model, args.minibatch_size, min_new_tokens, no_repeat_ngram_size, do_sample, top_k, top_p)
        validation_pred += pred_
        validation_true += true_
        validation_questions += prompt_
        del pred_, true_, prompt_
    new_f1, detailed_validation_report, average_example = evaluate(validation_questions, validation_pred, validation_true, args.minibatch_size)
    print(f'BERT F1 = {new_f1}.')
    print("VALIDATION_REPORT")
    for sample in detailed_validation_report:
        print("PROMPT:", sample["PROMPT"])
        print("TRUE:", sample["TRUE"])
        print("PRED:", sample["PRED"])
        print("F1:", sample["F1"])
    del detailed_validation_report, validation_pred, validation_true
