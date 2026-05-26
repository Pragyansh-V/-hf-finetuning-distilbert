# -hf-finetuning-distilbert/DistilBERT Emotion Classifier — Fine-tuning with LoRA

Fine-tuned DistilBERT on a 6-class emotion classification task using 
HuggingFace Trainer API and LoRA (Low-Rank Adaptation). Trained on the 
dair-ai/emotion dataset of 16,000 English tweets.

🤗 Live model: https://huggingface.co/Pragyansh14/distilbert-emotion-classifier

---

## What this project covers

- Loading and auditing a dataset from HuggingFace Hub
- Manual tokenisation with AutoTokenizer (WordPiece, padding, attention masks)
- Applying LoRA via the `peft` library — only 1.1% of parameters trained
- Fine-tuning with HuggingFace `Trainer` API on Apple MPS GPU
- Per-class evaluation with accuracy, weighted F1, and confusion matrix
- Pushing the final model to HuggingFace Hub

---

## Model performance

| Emotion | F1 Score | Support |
|---------|----------|---------|
| joy | 0.92 | 704 |
| sadness | 0.90 | 550 |
| anger | 0.85 | 275 |
| fear | 0.80 | 212 |
| love | 0.80 | 178 |
| surprise | 0.73 | 81 |

**Overall accuracy: 87.3% | Weighted F1: 0.87 | Macro F1: 0.83**

---

## Key findings

**Class imbalance drives performance gaps**
The dataset has significant imbalance — joy (33.5%) and sadness (29.2%) 
dominate while surprise is only 3.6% of training data. Before training, 
I predicted surprise would have the lowest F1 score. It did (0.73) — 
confirming that class frequency directly determines model performance 
on minority classes.

**Majority class gravity**
When the model is uncertain between emotions, it defaults toward joy — 
the most frequent class. 20% of surprise examples were misclassified 
as joy, and 16% of love examples were misclassified as joy.

**LoRA efficiency**
Full fine-tuning would update 66,958,086 parameters. LoRA reduced this 
to 742,662 — just 1.1% — with no meaningful performance loss. Training 
completed in 11 minutes on Apple M-series MPS GPU.

**Label noise**
Qualitative inspection revealed mislabelled examples — some "love" 
examples read as frustration, some "surprise" examples contained 
multiple simultaneous emotions. A model trained on noisy labels 
inherits those errors.

---

## Inclusive AI lens

Performance gaps across emotion classes are not just a technical metric — 
they have real-world consequences. A model deployed in an emotion-aware 
interface would reliably detect joy and sadness but miss surprise 36% 
of the time and misread love 24% of the time. Users whose emotional 
states fall into minority classes receive a systematically worse 
experience. Fixing this requires either collecting more balanced data 
or applying class-weighted loss during training.

---

## How to use the model

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="Pragyansh14/distilbert-emotion-classifier"
)

result = classifier("I can't believe this happened!")
print(result)
# [{'label': 'surprise', 'score': 0.87}]
```

---

## Stack

Python 3.14 · HuggingFace datasets 4.8.5 · transformers 5.9.0 · 
torch 2.12.0 (MPS) · peft 0.15 · scikit-learn · pandas · seaborn

---

## How to run locally

```bash
git clone https://github.com/Pragyansh-V/-hf-finetuning-distilbert.git
cd -hf-finetuning-distilbert
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook finetune.ipynb
```

---

## Part of a learning series

This is Project 2 in a series of HuggingFace projects building toward 
AI/ML engineering specialisation with a focus on inclusive AI.

- Project 1: [SST-2 Sentiment Model Audit](https://github.com/Pragyansh-V/hf-sentiment-audit)
- Project 2: DistilBERT Emotion Classifier with LoRA ← you are here
- Project 3: Coming soon — RAG pipeline