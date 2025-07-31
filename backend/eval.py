# backend/eval.py
import json, glob
from sklearn.metrics import accuracy_score
from rouge import Rouge
from api import boundly_query, Query

TEST_DIR = "data/test_questions/*.json"   # each file: {"q": "...", "label": "yes"}

rouge = Rouge()
ys_true, ys_pred = [], []
for path in glob.glob(TEST_DIR):
    sample = json.load(open(path))
    res = await boundly_query(Query(question=sample["q"]))
    pred = res["answer"]["decision"].lower()  # assume your JSON schema has this
    ys_true.append(sample["label"])
    ys_pred.append(pred)

print("Exact-match accuracy:", accuracy_score(ys_true, ys_pred))

# Explanation quality
refs = [json.load(open(p))["explanation"] for p in glob.glob(TEST_DIR)]
hypos = [await boundly_query(Query(question=json.load(open(p))["q"]))["answer"]["explanation"]
         for p in glob.glob(TEST_DIR)]
scores = rouge.get_scores(hypos, refs, avg=True)
print("ROUGE-L:", scores["rouge-l"]["f"])