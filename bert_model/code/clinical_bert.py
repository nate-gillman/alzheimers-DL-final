import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from transformers import AdamW
from datasets import load_metric
from tqdm.auto import tqdm
from transformers import get_scheduler
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification



class ADNI_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }

    

def main():

    # STEP 1: Decide what attribute we'll classify with respect to; options are ["AEHEVNT", "AEHCOMM", "CONCAT"]
    description = "AEHCOMM"

    # STEP 2: Load the training data
    X_train, X_test = list(pd.read_csv("../data/X_train_S.csv")[description]), list(pd.read_csv("../data/X_test_S.csv")[description])    
    y_train, y_test = list(pd.read_csv("../data/y_train_S.csv")["DIAGNOSIS"]), list(pd.read_csv("../data/y_test_S.csv")["DIAGNOSIS"])
    for i in range(len(y_train)):
        y_train[i] = y_train[i] - 1
    for i in range(len(y_test)):
        y_test[i] = y_test[i] - 1
        
    # the model we gonna train, base uncased BERT
    # check text classification models here: https://huggingface.co/models?filter=text-classification
    model_name = "bert-base-uncased"
    # max sequence length for each document/sentence sample
    max_length = 512
    
    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT")
    #tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)

    # tokenize the dataset, truncate when passed `max_length`, 
    # and pad with 0's when less than `max_length`
    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=max_length)
    valid_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=max_length)
    
    # convert our tokenized data into a torch Dataset
    train_dataset = ADNI_Dataset(train_encodings, np.array(y_train).astype('int64'))
    valid_dataset = ADNI_Dataset(valid_encodings, np.array(y_test).astype('int64'))
    
    #model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3) #.to("cuda")
    model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT", num_labels=10)
    metric= load_metric("accuracy")
    optimizer = AdamW(model.parameters(), lr=0.001)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    num_epochs = 20
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
)

    
    progress_bar = tqdm(range(num_training_steps))

    model.train()#Set the train status and enable Batch Normalization and Dropout.
    for epoch in range(num_epochs):
        for batch in train_loader:
            batch = {k: v for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            output_file = open('loss_log.txt','a')
            output_file.write('Loss: %1.2f\n' %  (float(loss),))
            output_file.close()
            print(loss)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
            temp_acc = metric.compute()
            print(temp_acc["accuracy"])
            output_file = open('acc_log.txt','a')
            output_file.write('Acc: %1.2f\n' %  (temp_acc["accuracy"],))
            output_file.close()
        progress_bar.update(1)
    
    
    eval_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True)
    
    model.eval()
    metric= load_metric("accuracy")
    for batch in eval_loader:
        batch = {k: v for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)  
    
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        #temp_acc = metric.compute()
        #print(metric["accuracy"])
    
    results = metric.compute()
    print("final accuracy " + str(results["accuracy"]))
    return None

if __name__ == '__main__':
    main()


