import json
import sys
import numpy as np
import re
def cleaned(s):
    return re.sub(r'\n+', '\n', s)

with open("/userspace/cdd/dolly_translation/mongolian_dolly.json", "r", encoding="utf-8") as f:
    mn = json.load(f)
with open("/userspace/cdd/dolly_translation/mongolian_dolly1.json", "r", encoding="utf-8") as f1:
    mn1 = json.load(f1)
with open("/userspace/cdd/dolly_translation/mongolian_dolly2.json", "r", encoding="utf-8") as f2:
    mn2 = json.load(f2)
eng_ds = []
with open("/userspace/cdd/dolly_translation/databricks-dolly-15k.jsonl", "r", encoding="utf-8") as f3:
    for line in f3:
        eng_ds.append(json.loads(line))

ids_for_translate = []
ids_for_long_sentences = []
ids_for_block = [492,3708,11604,11743,274]#[492]
skiped = 0
summ = 0

ds = mn + mn1 + mn2

TRAINING_TRUNCATE = 0.87
ds_for_training = [item for item in ds if float(item["score"]) > TRAINING_TRUNCATE]
ids_for_training = [int(item["index"]) for item in ds_for_training]
with open("ds_for_training_mon.json", "w", encoding = "utf-8") as final:
    json.dump(ds_for_training, final, ensure_ascii=False)

ls = []
for item in ds_for_training:
    ls.append(float(item["score"]))
import matplotlib.pyplot as plt
plt.plot(ls)
plt.savefig('график.png')

TRANSLATE_DATASET_LENGTH = 200
worst_ids = [int(item["index"]) for item in sorted(ds, key=lambda x: x["score"])[:500]]
prompt_lens = []
for idx in worst_ids:
    prompt_lens.append(len(eng_ds[idx]["instruction"].split()) + len(eng_ds[idx]["context"].split()) + len(eng_ds[idx]["response"].split()))

percent = np.percentile(prompt_lens, 90)
for n,idx in enumerate(worst_ids):
    if idx in ids_for_block:
        continue
    elif len(eng_ds[idx]["instruction"].split()) + len(eng_ds[idx]["context"].split()) + len(eng_ds[idx]["response"].split()) >= percent:
        ids_for_long_sentences.append(idx)
        continue
    else:
        ids_for_translate.append(idx)
        
ids_for_translate = ids_for_translate[:TRANSLATE_DATASET_LENGTH]
with open('to_translate.txt', 'w') as file:
    sys.stdout = file
    for n,idx in enumerate(ids_for_translate):
        summ += len(eng_ds[idx]["instruction"].split()) + len(eng_ds[idx]["context"].split()) + len(eng_ds[idx]["response"].split())
        print("INSTRUCTION N"+str(n)+" ID"+str(idx))
        print(cleaned(eng_ds[idx]["instruction"]))
        print("CONTEXT N"+str(n)+" ID"+str(idx))
        print(cleaned(eng_ds[idx]["context"]))
        print("RESPONSE N"+str(n)+" ID"+str(idx))
        print(cleaned(eng_ds[idx]["response"]))
    sys.stdout = sys.__stdout__

print("percentile", percent)
print("sum_of_200_worst", sum(prompt_lens[100:300]))
print("sum_of_200_worst_cleaned", summ)
print("len of long 500", len(ids_for_long_sentences))
print("len of blocked", len(ids_for_block))
print("len ids for training", len(ids_for_training))
#with open("to_translate.txt", 'r') as file:
#    print(len(file.read().split()))