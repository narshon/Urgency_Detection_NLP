#from google.colab import drive
import os
#drive.mount('/content/drive', force_remount=True)
#root_dir = "drive/MyDrive/Pipeline"

#! pip install datasets transformers==4.11.3

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
from datasets import load_dataset
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

train_data_url = 'input/no_train_df_v7.csv'
dev_data_url = 'input/no_dev_df_v7.csv'
test_data_url = 'input/no_test_df_v7.csv'
result_data_url = 'output/exp39_results.csv'

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
        #'support': _
    }

def huggingFaceMBERTTrain(model, tokenizer, train_data_url, dev_data_url):

    dataset = load_dataset('csv', data_files={'train': train_data_url,
                                            'dev': dev_data_url})

    #train_dataset, test_dataset = load_dataset('imdb', split=['train', 'test'])
    train_dataset = dataset['train'].map(tokenize, batched=True, batch_size=len(dataset['train']))
    dev_dataset = dataset['dev'].map(tokenize, batched=True, batch_size=len(dataset['dev']))

    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    dev_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    training_args = TrainingArguments(
        output_dir='model/results',
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        #warmup_steps=10,
        weight_decay=0.01,
        evaluation_strategy="epoch", #epoch
        #eval_steps = 100,
        #logging_steps = 100,
        #evaluate_during_training=True,
        logging_dir='logs/'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset
    )

    return trainer

name = 'Davlan/bert-base-multilingual-cased-finetuned-swahili'
model = AutoModelForSequenceClassification.from_pretrained(name)
tokenizer = AutoTokenizer.from_pretrained(name)



trainer_system = huggingFaceMBERTTrain(model, tokenizer, train_data_url, dev_data_url)
trainer_system.train()

trainer_system.evaluate()

trainer_system.save_model("model/FineTuned/")
tokenizer.save_pretrained("model/FineTuned/")

#Load model and tokenizer from disk
model = AutoModelForSequenceClassification.from_pretrained('model/FineTuned')
tokenizer = AutoTokenizer.from_pretrained('model/FineTuned')

#initialize classifier
classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, return_all_scores =True)

sentences_df_system = pd.read_csv(test_data_url)
results_system = classifier(sentences_df_system.text.values.tolist())

results_df_system = pd.DataFrame(results_system, columns = ["Label_0", "Label_1"])
results_system_preds = pd.DataFrame(results_df_system['Label_1'].tolist(), index=results_df_system.index, columns=['Label_1', 'score'])
results_system_preds['mid'] = sentences_df_system.mid.values
results_system_preds['text'] = sentences_df_system.text.values.tolist()
results_system_preds['label'] = sentences_df_system.label.values

results_system_preds.to_csv(result_data_url)

