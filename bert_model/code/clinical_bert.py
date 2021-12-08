import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AdamW
from datasets import load_metric
from tqdm.auto import tqdm
from transformers import get_scheduler
from transformers import AutoTokenizer
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



def main():

    batch_size = 32
    learning_rate = 0.001

    # STEP 1: Decide what attribute we'll classify with respect to; options are ["AEHEVNT", "AEHCOMM", "CONCAT"]
    description = "AEHCOMM"
    #description = "AEHEVNT"

    # STEP 2: Load the training data
    X_train, X_test = list(pd.read_csv("../data/X_train.csv")[description]), list(pd.read_csv("../data/X_test.csv")[description])    
    y_train, y_test = list(pd.read_csv("../data/y_train.csv")["DIAGNOSIS"]), list(pd.read_csv("../data/y_test.csv")["DIAGNOSIS"])
    
    for i in range(len(y_train)):
        y_train[i] = y_train[i] - 1
    for i in range(len(y_test)):
        y_test[i] = y_test[i] - 1
        
   
    # max sequence length for each document/sentence sample
    max_length = 512
    
    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT")

    # tokenize the dataset, truncate when passed `max_length`, 
    # and pad with 0's when less than `max_length`
    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=max_length)
    valid_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=max_length)
    
    # convert our tokenized data into a torch Dataset
    train_dataset = ADNI_Dataset(train_encodings, np.array(y_train).astype('int64'))
    valid_dataset = ADNI_Dataset(valid_encodings, np.array(y_test).astype('int64'))
    
   
    model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT", num_labels=3)
    metric= load_metric("accuracy")
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    num_epochs = 10
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
)

    
    progress_bar = tqdm(range(num_training_steps))
    loss_total = 0
    acc_total = 0 
    
    model.train()#Set the train status and enable Batch Normalization and Dropout.
    for epoch in range(num_epochs):
        loss_total = 0
        acc_total = 0
        for batch in train_loader:
            batch = {k: v for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss_total += loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
            temp_acc = metric.compute()
            acc_total += temp_acc["accuracy"]
        loss = loss_total/len(train_loader)
        output_file = open('loss_log.txt','a')
        output_file.write('Loss: %1.2f\n' %  (float(loss),))
        output_file.close()
        acc = acc_total/len(train_loader)
        output_file = open('acc_log.txt','a')
        output_file.write('Acc: %1.2f\n' %  (acc,))
        output_file.close()
        progress_bar.update(1)
    
    
    eval_loader = DataLoader(valid_dataset, batch_size, shuffle=True)
    
    model.eval()
    metric= load_metric("accuracy")
    for batch in eval_loader:
        batch = {k: v for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)  
    
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    
    results = metric.compute()
    print("final accuracy " + str(results["accuracy"]))
    return None

if __name__ == '__main__':
    main()


