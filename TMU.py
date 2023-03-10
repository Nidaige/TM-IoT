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
#import pycuda
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


def binary(num):  # converts a float to binary, 8 bits
    '''
    Function for float to binary from here: 
    https://stackoverflow.com/questions/16444726/binary-representation-of-float-in-python-bits-not-hex
    '''
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))


def shuffle_dataset(dataset):
    '''
    This function shuffles a list of tuples using a recursive method.
    '''
    print("Shuffling...")
    output_values_list = []  # list for shuffled values
    output_labels_list = []  # list for shuffled labels
    tempdata = dataset  # copy dataset for further manipulation
    while len(tempdata[0]) > 0:  # as long as there is data in the dataset, keep going
        index = math.floor(random.random() * len(tempdata[0]))  # randomly select an element by index
        output_values_list.append(
            np.array(tempdata[0][index]))  # copy the element in each slot (value, label, string label) to the output
        output_labels_list.append(tempdata[1][index])  # --||--
        for c in range(2):  # for each slot (value, label, string label):
            # temp = tempdata[c][index]  # copy as a buffer variable
            tempdata[c][index] = tempdata[c][-1]  # copy final element in array to the [index] location
            tempdata[c] = tempdata[c][
                          0:-1]  # overwrite dataset with dataset except last element (which is now copied to location[index])
    print("Shuffling done")
    return (output_values_list, output_labels_list)


def iot_data_to_binary_list(path, max_bits, database, registry):
    '''
    This function takes a path to a csv file, the number of bits to use for each dataset value, a dictionary with dataset items in lists under each category as keys, and a dictionary with the number of elements in each dataset category, and returns updated versions of the two dictionaries based on the data read from the file.
    '''
    print("Binarizing...")
    data_values = []
    data = pandas.read_csv(path)  # read data csv


    # output: dictionary with keys "x_train/x_test" for data and "y_train/y_test" for labels
    # cicids output:
    # Duplicate the registry, add new keys to it
    reg = registry
    db = database
    for new_label in list(data[' Label'].unique()):
        if new_label not in reg.keys():
            reg[new_label] = 0
            db[new_label] = []

    for num in range(len(data) - 1):  # numerically iterate through every line of data
        row = data.loc[num]  # get the data for each row
        datapoint = []  # empty list to hold the features of each row
        for item in row:  # for each value in a row
            datapoint.append(item)  # add it to the list of features for this row
        data_values.append(datapoint)  # add the final list of features for this row to the processed dataset  

    for item in data_values:  # for each dataset item
        values = item[0:-1]
        label = item[-1]
        rowie = ""  # string to temporarily hold binary representation of the data item
        for feature in values:  # for each value / feature in said item_
            rowie += str(binary(float(feature)))[
                     -max_bits:]  # concatenate the binary string for each feature to the string representing the item
        db[label].append([*rowie])
        registry[label] += 1
    print("Binarizing done")
    return (db,
            registry)  # returns tuple of binary representation of data item and its label as an integer, and the string of labels for later decoding.


def calculate_multiclass_metrics(y_true, y_pred, list_of_classes):
    recall_scores = {}
    precision_scores = {}
    specificity_scores = {}
    f1_scores = {}
    classes = {}

    n_classes = len(set(y_true))
    for i in range(n_classes):
        class_true = (y_true == i)
        class_pred = (y_pred == i)
        sum_class_true = sum(class_true)
        sum_class_pred = sum(class_pred)
        
        if sum_class_true != 0:
            recall = sum(class_true == class_pred) / sum_class_true
        else:
            recall = 0
        recall_scores[i] = recall

        if sum_class_pred != 0:
            precision = sum(class_true == class_pred) / sum(class_pred)
        else:
            precision = 0
        precision_scores[i] = precision

        not_class_true = ~class_true
        not_class_pred = ~class_pred
        sum_not_class_true = sum(not_class_true)
        
        if sum_not_class_true != 0:
            specificity = sum(not_class_true == not_class_pred) / sum(not_class_true)
        else:
            specificity = 0
        specificity_scores[i] = specificity

                    
        if precision != 0 or recall != 0:
            if (precision + recall)!= 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
        else:
            f1 = 0
        f1_scores[i] = f1
        text = list_of_classes[i]
        text = text.replace(".", "_")
        text= text.replace(" ", "_")
        classes[i] = text

    classes_df = pd.DataFrame.from_dict(classes, orient='index', columns=['class'])
    recall_df = pd.DataFrame.from_dict(recall_scores, orient='index', columns=['recall'])
    precision_df = pd.DataFrame.from_dict(precision_scores, orient='index', columns=['precision'])
    specificity_df = pd.DataFrame.from_dict(specificity_scores, orient='index', columns=['specificity'])
    f1_df = pd.DataFrame.from_dict(f1_scores, orient='index', columns=['f1_score'])
    
    result = pd.concat([classes_df, recall_df, precision_df, specificity_df, f1_df], axis=1)

    return result


def compare_two_lists(a,b):
    equal = 0
    for index, item in enumerate(a):
        if item == b[index]:
            equal += 1
    return equal

def compare_list_to_value(my_list, my_value):
    e = 0
    for item in my_list:
        if item == my_value:
            e += 1

''' Main run begins here'''

s = 150
T = 500
clauses = 25000
max_literals = 64
class_size_cutoff = 12000
epochs = 1



import tmu.datasets
preprocessing_time_start = time()

# CIC-IDS2017
'''
CIC = tmu.datasets.CICIDS2017(self, split=0.7, shuffle=True, balance=True, binarize=True, bits_per_entry = 16, max_data_entries=450000, data_category_threshold = 5000)
paths = ["Data/"+i for i in os.listdir("Data")]
dataset = CIC.retrieve_dataset(paths)
'''

# KDD
'''kdd = tmu.datasets.KDD99(split=0.7, shuffle=True, booleanize=True, max_bits_per_literal=max_literals, balance=True,
                         class_size_cutoff=500)
dataset = kdd.retrieve_dataset()'''

# NSL-KDD
'''nsl = tmu.datasets.NSLKDD(shuffle = False, booleanize = True, balance_train_set = True, balance_test_set = True, max_bits_per_literal = 32, class_size_cutoff = 30000,
                         limit_to_classes_in_train_set = True, limit_to_classes_in_test_set = True)
dataset = nsl.retrieve_dataset()'''


# UNSW-NB15
'''
UNSWNB15 = tmu.datasets.UNSW_NB15(split=0.7, shuffle=True, balance=True, binarize=True, bits_per_entry = 32, max_data_entries=450000, data_category_threshold = class_size_cutoff)
paths = ["UNSW/UNSW-NB15 - CSV Files/UNSW-NB15_"+str(fi)+".csv" for fi in [1,2,3,4]]
dataset = UNSWNB15.retrieve_dataset(paths)
'''

# Bot IoT
BotIoT = tmu.datasets.Bot_IoT(split=0.7, shuffle=True, balance=True, binarize=True, bits_per_entry = 32, max_data_entries=450000, data_category_threshold = class_size_cutoff)
#paths = ["Bot_IoT/Entire Dataset/UNSW_2018_IoT_Botnet_Dataset_"+str(fi)+".csv" for fi in list(range(19,27))]
paths = ["Bot_IoT/All features/UNSW_2018_IoT_Botnet_Full5pc_"+str(fi)+".csv" for fi in list(range(1,5))]
dataset = BotIoT.retrieve_dataset(paths)

X_train = dataset["x_train"]
Y_train = dataset["y_train"]
X_test = dataset["x_test"]
Y_test = dataset["y_test"]
all_classes = dataset["target"]
preprocessing_time_end = time()
preprocessing_time = preprocessing_time_end - preprocessing_time_start
print("Preprocesing time: ", preprocessing_time, "s")
Run_Name = "S:" + str(s) + "T:" + str(T) + "Clauses:" + str(clauses) + "Max_literals:" + str(
    max_literals) + "Epochs:" + str(epochs)+"Cutoff: "+str(class_size_cutoff)
Project_name = "UNSW-BotIoT-Sweep_2"

config = dict(
    Forget_rate=s,
    Threshold=T,
    Clauses=clauses,
    Max_literals=max_literals,
    Epochs=epochs,
    Weighted_Clauses = True
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
    tm = tmuclassifier.TMClassifier(number_of_clauses=config["Clauses"], T=config["Threshold"], s=config["Forget_rate"], max_included_literals=config["Max_literals"], platform='CUDA',
                      weighted_clauses=config["Weighted_Clauses"])
    print("RUNNING TSETLIN MACHINE ON CUDA GPU")
    print(torch.cuda.device_count())

else:
    tm = tmuclassifier.TMClassifier(number_of_clauses=config["Clauses"], T=config["Threshold"], s=config["Forget_rate"], max_included_literals=config["Max_literals"], platform='CPU',
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
print("Predictions done, calculating score---")
Total = 0
Correct = 0
# conf_matr = confusion_matrix
# For each test data item, check if correct prediction
for test_data_sample in range(len(X_test)):
    Total += 1
    if Prediction[test_data_sample] == Y_test[test_data_sample]:  # if correct guess:
        Correct += 1
