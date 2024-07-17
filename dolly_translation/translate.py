from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import json
import torch
import bert_score
import numpy as np
import random
import sys
eng_ds = []
mon_ds = []
with open("databricks-dolly-15k.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        eng_ds.append(json.loads(line))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

facebook_model = AutoModelForSeq2SeqLM.from_pretrained("models/facebook_model").cuda()
facebook_tokenizer = AutoTokenizer.from_pretrained("models/facebook_tokenizer") 
facebook_tokenizer.src_lang = "en_XX"
google_model = AutoModelForSeq2SeqLM.from_pretrained("models/google_model").cuda()
google_tokenizer = AutoTokenizer.from_pretrained("models/google_tokenizer")
nllb_model = AutoModelForSeq2SeqLM.from_pretrained("models/nllb_model").cuda()
nllb_tokenizer = AutoTokenizer.from_pretrained("models/nllb_tokenizer") 
nllb_translator = pipeline('translation', model=nllb_model, tokenizer=nllb_tokenizer, src_lang="eng_Latn", tgt_lang='khk_Cyrl', device = 0)

def facebook_translate(text):
    text = text[:2500]
    encoded = facebook_tokenizer(text, return_tensors="pt").to(device)
    generated_tokens = facebook_model.generate(
        **encoded,
        forced_bos_token_id=facebook_tokenizer.lang_code_to_id["mn_MN"],
        max_length=500
    )
    return facebook_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

def google_translate(text):
    text = text[:2500]
    text = "<2mn> "+text
    input_ids = google_tokenizer(text, return_tensors="pt").input_ids.to(device)
    outputs = google_model.generate(input_ids=input_ids, max_length=500)
    return google_tokenizer.decode(outputs[0], skip_special_tokens=True)

def nllb_translate(text):
    text = text[:2500]
    return nllb_translator(text, max_length=500)[0]["translation_text"]

def mean_score_and_max_idx(text1, text2, text3):
    a = []
    P, R, F1 = bert_score.score([text1, text2, text3], [text2, text3, text1], model_type="xlm-roberta-large", num_layers = 17)
    for x in F1:
        a.append(x)
    mp = {0: random.choice([0,1]), 1: random.choice([1,2]), 2: random.choice([0,2])}
    idx = mp[np.argmax(a)]
    return np.mean(a), idx
    
def instruction_str(instruction):
    if instruction["context"] == "":
        return instruction["instruction"] + "\n" + instruction["response"]
    else:
        return instruction["instruction"] + "\n" + instruction["context"] + "\n" + instruction["response"]

def translate_dictonary(dt, translator):
    new_dt = {}
    new_dt["instruction"] = translator(dt["instruction"])
    if dt["context"] == "":
        new_dt["context"] = ""
    else:
        new_dt["context"] = translator(dt["context"])   
    new_dt["response"] = translator(dt["response"])
    new_dt["category"] = dt["category"]
    new_dt["score"] = str(-1.0)
    new_dt["index"] = str(0)
    return new_dt

start = 10000
end = 15000
for i,x in enumerate(eng_ds[start:end]):
    if i % 500 == 0:
        print(str(i), file=sys.stderr)
    d1 = translate_dictonary(x, nllb_translate)
    d2 = translate_dictonary(x, facebook_translate)
    d3 = translate_dictonary(x, google_translate)
    a = [d1, d2, d3]
    score, idx = mean_score_and_max_idx(instruction_str(d1),instruction_str(d2),instruction_str(d3))
    a[idx]["score"] = str(score)
    a[idx]["index"] = str(i+start)
    mon_ds.append(a[idx])
            
file_name = "mongolian_dolly2.json"
with open(file_name, "w", encoding = "utf-8") as final:
    json.dump(mon_ds, final, ensure_ascii=False)
    