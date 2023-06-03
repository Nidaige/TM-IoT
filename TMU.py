import numpy as np
from time import time
#import tmu
#from tmu.models.classification.vanilla_classifier import TMClassifier
import tmu as tmu
import tmu.datasets
import tmu.models.classification.vanilla_classifier as tmuclassifier
import os
#import numpy as np
import pandas
import pandas as pd
import tensorflow
import pycuda
import struct
import random
import math
#import openpyxl
import wandb
from sklearn.datasets import fetch_openml
import torch
from sklearn.metrics import f1_score, precision_score, recall_score



''' Main run begins here'''

s = 1
T = 1
clauses = 10
max_literals = 32
class_size_cutoff = 100
epochs = 1
split = 0.7
wc = True

import tmu.datasets
preprocessing_time_start = time()
#--------------------------------------------------------------------------------------------------------------
# CIC-IDS2017
'''

paths = ["Data/"+i for i in os.listdir("Data")]
CIC = tmu.datasets.CICIDS2017(split=0.7, shuffle=True, balance=True, binarize=True, bits_per_entry = 32, data_category_threshold = data_category_threshold, dataset_filepaths=paths)
dataset = CIC.retrieve_dataset(paths)
datasetName = "cicids2017"

'''
#--------------------------------------------------------------------------------------------------------------
# KDD
'''

kdd = tmu.datasets.KDD99(split=split, shuffle=True, binarize = True, balance = True, class_size_cutoff = class_size_cutoff)
dataset = kdd.retrieve_dataset()
datasetName = "kdd99"

'''  
#--------------------------------------------------------------------------------------------------------------
# NSL-KDD
'''

nsl = tmu.datasets.NSLKDD(shuffle = True, binarize = True, balance_train_set = True, balance_test_set = True, max_bits_per_literal = 32, class_size_cutoff = class_size_cutoff, limit_to_classes_in_train_set = True, limit_to_classes_in_test_set = True, dataset_directory="NSL")
dataset = nsl.retrieve_dataset()
datasetName = "nslkdd"

'''
#--------------------------------------------------------------------------------------------------------------
# UNSW-NB15
'''

paths = ["UNSW/UNSW-NB15 - CSV Files/UNSW-NB15_"+str(fi)+".csv" for fi in [1,2,3,4]]
UNSWNB15 = tmu.datasets.UNSW_NB15(split=0.7, shuffle=True, balance=True, binarize=True, bits_per_entry = 32, data_category_threshold = class_size_cutoff, paths=paths)
dataset = UNSWNB15.retrieve_dataset()
datasetName = "nb15"

'''
#--------------------------------------------------------------------------------------------------------------
# Bot IoT
'''
paths = ["Bot_IoT/All features/UNSW_2018_IoT_Botnet_Full5pc_"+str(fi)+".csv" for fi in list(range(1,5))]
BotIoT = tmu.datasets.Bot_IoT(paths = paths, split=0.7, shuffle=True, balance=False, binarize=True, bits_per_entry = 32, data_category_threshold = class_size_cutoff)
dataset = BotIoT.retrieve_dataset()
datasetName = "Bot iot"

'''

X_train = dataset["x_train"]
Y_train = dataset["y_train"]
X_test = dataset["x_test"]
Y_test = dataset["y_test"]
all_classes = dataset["target"]
print((len(X_train) == len(Y_train)) and (len(X_test) == len(Y_test))) 
preprocessing_time_end = time()
preprocessing_time = preprocessing_time_end - preprocessing_time_start
print("Preprocesing time: ", preprocessing_time, "s")
Run_Name = datasetName
Project_name = "Testing Fixed Dataloader"
print(X_train[0], Y_train[0], all_classes[Y_train[0]])
config = dict(
    Forget_rate=s,
    Threshold=T,
    Clauses=clauses,
    Max_literals=max_literals,
    Epochs=epochs,
    Weighted_Clauses = wc
)

# If you are changing dataset, change the "project" before starting new runs. "name" determines the run name, not project.
wandb.init(project=Project_name, name=Run_Name, config=config)
config = wandb.config
cm = {}
for label in all_classes:
    cm[label] = {}
for label in all_classes:
    cm[label]["class"] = label
    for label2 in all_classes:
        cm[label][label2] = 0

print("\nAccuracy over " + str(config["Epochs"]) + " epochs:\n")
if torch.cuda.is_available():
    tm = tmuclassifier.TMClassifier(number_of_clauses=2*(config["Clauses"]//2), T=config["Threshold"], s=config["Forget_rate"], max_included_literals=config["Max_literals"], platform='CUDA',
                      weighted_clauses=config["Weighted_Clauses"])
    print("RUNNING TSETLIN MACHINE ON CUDA GPU")
    print(torch.cuda.device_count())

else:
    tm = tmuclassifier.TMClassifier(number_of_clauses=2*(config["Clauses"]//2), T=config["Threshold"], s=config["Forget_rate"], max_included_literals=config["Max_literals"], platform='CPU',
                      weighted_clauses=config["Weighted_Clauses"])
    print("RUNNING TSETLIN MACHINE ON CPU")
for i in range(config["Epochs"]):
    start_training = time()
    tm.fit(X_train, Y_train)
    stop_training = time()

    start_testing = time()
    pred = tm.predict(X_test)
    result = 100 * (pred == Y_test).mean()
    loss = 100 - result
    stop_testing = time()
    traintime = stop_training - start_training
    testtime = stop_testing - start_testing
    wandb.log({"Accuracy": result, "Loss": loss, "Training duration": traintime, "Testing Duration": testtime})
    F1_score = f1_score(Y_test, pred, average = 'weighted')
    recall = recall_score(Y_test, pred, average = 'weighted')
    Precision = precision_score(Y_test, pred, average='weighted', zero_division = 1)
    wandb.log({"F1 score:":F1_score, "Recall": recall, "Precision": Precision}) 
    print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i + 1, result, traintime, testtime))
Prediction = tm.predict(X_test)
'''
Evaluation_results = calculate_multiclass_metrics(Y_test, Prediction, all_classes)
wandb.log({"Evaluation Table": Evaluation_results})
'''
for i, p in enumerate(Prediction):
    cm[all_classes[p]][all_classes[Y_test[i]]] += 1
    text = all_classes[p]
    text = text.replace(".", "_")
    text= text.replace(" ", "_")
    cm[all_classes[p]]["class"] = all_classes[p]

cmdf = pd.DataFrame.from_dict(cm).T

# create a wandb.Table() with corresponding columns
columns = all_classes
test_table = wandb.Table(data=cmdf, columns=columns)
wandb.log({"Confusion Matrix": test_table})



