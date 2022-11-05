# !pip install -q transformers pytorch_lightning nervaluate
# !wget -q https://raw.githubusercontent.com/dumitrescustefan/ronec/master/data/train.json
# !wget -q https://raw.githubusercontent.com/dumitrescustefan/ronec/master/data/valid.json
# !wget -q https://raw.githubusercontent.com/dumitrescustefan/ronec/master/data/test.json

import logging, os, sys, json, torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig, Trainer, TrainingArguments
from pytorch_lightning.callbacks import EarlyStopping
from nervaluate import Evaluator
from time import perf_counter
from sklearn.metrics import f1_score

# =================================== TAGS =================================== #

# RONECv2 classes
tags = [
    "O",
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "NAT_REL_POL",
    "EVENT",
    "LANGUAGE",
    "WORK_OF_ART",
    "DATETIME",
    "PERIOD",
    "MONEY",
    "QUANTITY",
    "NUMERIC",
    "ORDINAL",
    "FACILITY",
]

# BIO2 tags list
bio2tags = [
    "O",
    "B-PERSON",
    "I-PERSON",
    "B-ORG",
    "I-ORG",
    "B-GPE",
    "I-GPE",
    "B-LOC",
    "I-LOC",
    "B-NAT_REL_POL",
    "I-NAT_REL_POL",
    "B-EVENT",
    "I-EVENT",
    "B-LANGUAGE",
    "I-LANGUAGE",
    "B-WORK_OF_ART",
    "I-WORK_OF_ART",
    "B-DATETIME",
    "I-DATETIME",
    "B-PERIOD",
    "I-PERIOD",
    "B-MONEY",
    "I-MONEY",
    "B-QUANTITY",
    "I-QUANTITY",
    "B-NUMERIC",
    "I-NUMERIC",
    "B-ORDINAL",
    "I-ORDINAL",
    "B-FACILITY",
    "I-FACILITY",
]

# ============================ MAIN TEST FUNCTION ============================ #


def test():
    # Load test data
    with open("train.json", "r", encoding="utf8") as f:
        train_dataset = json.load(f)
    with open("valid.json", "r", encoding="utf8") as f:
        valid_dataset = json.load(f)
    with open("test.json", "r", encoding="utf8") as f:
        test_dataset = json.load(f)

    # Train model
    model = MyModel()
    model.train(train_dataset, valid_dataset, "output")

    # Load model
    model = MyModel()
    model.load("output")

    # Inference
    start_time = perf_counter()
    f1_strict_score = model.predict(test_dataset)
    stop_time = perf_counter()

    print(f"Predicted in {stop_time-start_time:.2s}.")
    print(f"F1-strict score = {f1_strict_score:.5f}")  # this is the score we want :)


# ================================== DATASET ================================= #


class MyDataset(Dataset):
    """
    This class is used to parse data from test instances.
    In particular, test instances are phrases.
    """

    def __init__(self, tokenizer, instances):
        self.tokenizer = tokenizer  # we'll need this in the __getitem__ function
        self.instances = instances  # save the data for further use

    def __len__(self):
        return len(self.instances)  # return how many instances we have. It's a list after all

    def __getitem__(self, index):
        """
        Parse the data from a single instance at given index.
        """
        instance = self.instances[index]
        instance_ids, instance_ner_ids, instance_token_idx = [], [], []

        # Process each word
        for i in range(len(instance["tokens"])):
            word = instance["tokens"][i]  # this is the ith word in the sentence
            ner_id = instance["ner_ids"][i]  # the ith numeric value corresponding to the ith word

            word_ids = self.tokenizer.encode(
                word,
                add_special_tokens=False)  # tokenize the word, CAREFUL as it could give you 2 or more tokens per word
            word_labels = [ner_id]

            if len(word_ids) > 1:  # we have a word split in more than 1 tokens, fill appropriately
                # the filler will be O, if the class is Other/None, or I-<CLASS>
                if ner_id == 0:  # this is an O, all should be Os
                    word_labels.extend([0] * (len(word_ids) - 1))
                else:
                    if word_labels[0] % 2 == 0:  # this is even, so it's an I-<class>, fill with the same Is
                        word_labels.extend([word_labels[0]] * (len(word_ids) - 1))
                    else:  # this is a B-<class>, we'll fill it with Is (add 1)
                        word_labels.extend([(word_labels[0] + 1)] * (len(word_ids) - 1))

            # Add to our instance lists
            instance_ids.extend(word_ids)  # extend with the token list
            instance_ner_ids.extend(word_labels)  # extend with the ner_id list
            instance_token_idx.extend([i] * len(word_ids))  # extend with the id of the token (to reconstruct words)

        return {
            "instance_ids": instance_ids,
            "instance_ner_ids": instance_ner_ids,
            "instance_token_idx": instance_token_idx
        }


# ================================= COLLATOR ================================= #


class MyCollator(object):

    def __init__(self, tokenizer, max_seq_len):
        self.max_seq_len = max_seq_len  # this will be our model's maximum sequence length
        self.tokenizer = tokenizer  # we still need our tokenizer to know that the pad token's id is

    def __call__(self, input_batch):
        # Question for you: print the input_batch to see what it contains ;)
        output_batch = {"input_ids": [], "labels": [], "token_idx": [], "attention_mask": []}

        max_len = 0  # we'll need first to find out what is the longest line and then pad the rest to this length

        for instance in input_batch:
            instance_len = min(len(instance["instance_ids"]),
                               self.max_seq_len - 2)  # we will never have instances > max_seq_len-2
            max_len = max(max_len, instance_len)  # update max

        for instance in input_batch:  # for each instance
            instance_ids = instance["instance_ids"]  # it's clearer if we use variables again
            instance_ner_ids = instance["instance_ner_ids"]
            instance_token_idx = instance["instance_token_idx"]

            # create the attention mask
            # this is a vector of 1s if the token is to be processed (0 if it's padding)
            instance_attention_mask = [1] * len(instance_ids)  # just a list of 1s for now

            # cut to max sequence length, if needed
            # notice how easy it is to process them together
            if len(
                    instance_ids
            ) > self.max_seq_len - 2:  # we need the -2 to accomodate for special tokens, this is a transformer's quirk
                instance_ids = instance_ids[:self.max_seq_len - 2]
                instance_ner_ids = instance_ner_ids[:self.max_seq_len - 2]
                instance_token_idx = instance_token_idx[:self.max_seq_len - 2]
                instance_attention_mask = instance_attention_mask[:self.max_seq_len - 2]
            # Depending on your chosen model, the transformer might not have cls and sep, so don't use them
            # if self.tokenizer.cls_token_id and self.tokenizer.sep_token_id:
            #     instance_ids = [self.tokenizer.cls_token_id] + instance_ids + [self.tokenizer.sep_token_id]
            #     instance_ner_ids = [0] + instance_ner_ids + [0]
            #     instance_token_idx = [
            #         -1
            #     ] + instance_token_idx  # no need to pad the last, will do so automatically at return

            # how much would we need to pad?
            pad_len = max_len - len(instance_ids)  # with this much

            if pad_len > 0:
                # pad the instance_ids
                instance_ids.extend([self.tokenizer.pad_token_id] *
                                    pad_len)  # notice we're padding with tokenizer.pad_token_id

                # pad the instance_ner_ids
                instance_ner_ids.extend([0] * pad_len)  # pad with zeros

                # pad the token_ids
                instance_token_idx.extend([-1] * pad_len)  # notice we're padding with -1 as 0 is a valid word index

                # pad the attention mask
                instance_attention_mask.extend([0] * pad_len)  # pad with zeros as well

            # add to batch
            output_batch["input_ids"].append(instance_ids)
            output_batch["labels"].append(instance_ner_ids)
            output_batch["token_idx"].append(instance_token_idx)
            output_batch["attention_mask"].append(instance_attention_mask)

        # we're done cutting and padding, let's transform them to tensors
        output_batch["input_ids"] = torch.tensor(output_batch["input_ids"])
        output_batch["labels"] = torch.tensor(output_batch["labels"])
        output_batch["token_idx"] = torch.tensor(output_batch["token_idx"])
        output_batch["attention_mask"] = torch.tensor(output_batch["attention_mask"])

        return output_batch


# =================================== MODEL ================================== #


class MyModel():

    def __init__(self):
        self.transformer_model = TransformerModel()

    def load(self, model_resource_folder):
        self.transformer_model.load_state_dict(torch.load(model_resource_folder + "transformer_model.pt"))

    def save(self, model_resource_folder):
        os.makedirs("output", exist_ok=True)
        open("output/transformer_model.pt", 'w')
        torch.save(self.transformer_model.state_dict(), model_resource_folder + "transformer_model.pt")

    def train(self, train_dataset, validation_dataset, model_resource_folder):
        early_stop = EarlyStopping(monitor='valid/strict', min_delta=0.001, patience=5, verbose=True, mode='max')

        trainer = pl.Trainer(
            devices=-1,  # uncomment this when training on gpus
            accelerator="gpu",  # uncomment this when training on gpus
            max_epochs=-1,  # set this to -1 when training fully
            callbacks=[early_stop],
            #limit_train_batches=100,  # comment this out when training fully
            #limit_val_batches=5,  # comment this out when training fully
            gradient_clip_val=1.0,
            enable_checkpointing=False  # this disables saving the model each epoch
        )

        # instantiate dataloaders
        # a batch_size of 8 should work fine on 16GB GPUs
        my_collator = MyCollator(self.transformer_model.tokenizer, 100)
        train_dataloader = DataLoader(MyDataset(self.transformer_model.tokenizer, train_dataset),
                                      batch_size=8,
                                      collate_fn=my_collator,
                                      shuffle=True,
                                      pin_memory=True,
                                      num_workers=4)
        validation_dataloader = DataLoader(MyDataset(self.transformer_model.tokenizer, validation_dataset),
                                           batch_size=8,
                                           collate_fn=my_collator,
                                           shuffle=False,
                                           pin_memory=True,
                                           num_workers=4)

        # call this to start training
        trainer.fit(self.transformer_model, train_dataloader, validation_dataloader)
        
        self.save(model_resource_folder)

    def predict(self, test_dataset):
        return self.transformer_model.predict(test_dataset)


class TransformerModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.transformer_model = "dumitrescustefan/bert-base-romanian-cased-v1"
        self.tokenizer = AutoTokenizer.from_pretrained(self.transformer_model, strip_accents=False)
        self.model = AutoModelForTokenClassification.from_pretrained(self.transformer_model,
                                                                     num_labels=len(bio2tags),
                                                                     from_flax=False)
        self.model_max_len = 512
        self.lr = 2e-5

        # we want to record our training loss and validation examples & loss
        # we'll hold them in these lists, and clean them after each epoch
        self.train_loss = []
        self.valid_y_hat = []
        self.valid_y = []
        self.valid_loss = []

    def forward(self, input_ids, attention_mask, labels):
        # we're just wrapping the code on the AutoModelForTokenClassification
        # it needs the input_ids, attention_mask and labels

        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)

        return output["loss"], output["logits"]

    def training_step(self, batch, batch_idx):
        # simple enough, just call forward and then save the loss
        loss, _ = self.forward(batch["input_ids"], batch["attention_mask"], batch["labels"])
        self.train_loss.append(loss.detach().cpu().numpy())
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # call forward to get loss and logits
        loss, logits = self.forward(batch["input_ids"], batch["attention_mask"],
                                    batch["labels"])  # logits is [batch_size, seq_len, num_classes]

        # let's extract our prediction and gold variables - we'll need them to evaluate our predictions
        batch_pred = torch.argmax(logits.detach().cpu(), dim=-1).tolist()  # reduce to [batch_size, seq_len] as list
        batch_gold = batch["labels"].detach().cpu().tolist()  # [batch_size, seq_len] as list
        batch_token_idx = batch["token_idx"].detach().cpu().tolist()

        # because our tokenizer can generate more than one token per word, we'll take the class predicted by the first
        # token as the class for the word. For example, if we have [Geor] [ge] with predicted classes [B-PERSON] and
        # [B-GPE] (for example), we'll assign to word George the class of [Geor], ignoring any other subsequent tokens.
        batch_size = logits.size()[0]
        for batch_idx in range(batch_size):
            pred, gold, idx = batch_pred[batch_idx], batch_gold[batch_idx], batch_token_idx[batch_idx]
            y_hat, y = [], []
            for i in range(0, max(idx) + 1):  # for each sentence, for each word in sequence
                pos = idx.index(i)  # find the position of the first ith token, and get pred and gold
                y_hat.append(pred[pos])  # save predicted class
                y.append(gold[pos])  # save gold class
            self.valid_y_hat.append(y_hat)
            self.valid_y.append(y)

        self.valid_loss.append(loss.detach().cpu().numpy())  # save our loss as well

        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        print()  # to start printing nicely on a new line
        mean_val_loss = sum(self.valid_loss) / len(self.valid_loss)  # compute average loss
        # for our evaluation, we'll need to convert class indexes to actual classes
        gold, pred = [], []
        for y, y_hat in zip(self.valid_y,
                            self.valid_y_hat):  # for each pair of predicted & gold sentences (sequences of ints)
            gold.append([bio2tags[token_id] for token_id in y
                        ])  # go, for each word in the sentence, from class id to class
            pred.append([bio2tags[token_id] for token_id in y_hat])  # same for our prediction list

        evaluator = Evaluator(gold, pred, tags=tags, loader="list")  # call the evaluator

        # let's print a few metrics
        results, results_by_tag = evaluator.evaluate()
        self.log("valid/avg_loss", mean_val_loss, prog_bar=True)
        self.log("valid/ent_type", results["ent_type"]["f1"])
        self.log("valid/partial", results["partial"]["f1"])
        self.log("valid/strict", results["strict"]["f1"])
        self.log("valid/exact", results["exact"]["f1"])

        # reset our records for a new epoch
        self.valid_y_hat = []
        self.valid_y = []
        self.valid_loss = []

    def configure_optimizers(self):
        return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-08)

    def predict(self, test_dataset):
        # we first have to process our text in the same way we did for training, so let's borrow some code from the Dataset
        input_ids, attention_mask, token_idx = [], [], []

        # let's process each word
        for i in range(len(test_dataset)):
            token_ids = self.tokenizer.encode(
                test_dataset[i],
                add_special_tokens=False)  # tokenize the word, CAREFUL as it could give you 2 or more tokens per word
            input_ids.extend(token_ids)
            token_idx.extend([i] * len(token_ids))  # save for each added token_id the same word positon i

        # the attention mask is now simply a list of 1s the length of the input_ids
        attention_mask = [1] * len(input_ids)

        # convert them to tensors; we simulate batches by placing them in [], equivalent to batch_size = 1
        input_ids = torch.tensor([input_ids],
                                 device=self.device)  # also place them on the same device (CPU/GPU) as the model
        attention_mask = torch.tensor([attention_mask], device=self.device)

        # now, we are ready to run the model, but without labels, for which we'll pass None
        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        # extract logits and move to cpu
        logits = output['logits'].cpu()  # this will be [1, seq_len, 31], for batch_size = 1, and 31 classes

        # let's extract our prediction
        prediction_int = torch.argmax(
            logits, dim=-1).squeeze().tolist()  # reduce to [seq_len] as list, as batch_size = 1 due to .squeeze()

        word_prediction_int = []
        for i in range(0, max(token_idx) + 1):  # for each word in the sentence
            pos = token_idx.index(i)  # find the position of the first ith token, and get pred and gold
            word_prediction_int.append(prediction_int[pos])  # save predicted class

        # last step, convert the ints to strings
        prediction = []
        for i in range(len(word_prediction_int)):
            prediction.append(bio2tags[word_prediction_int[i]])  # lookup in tag list

        return prediction


test()