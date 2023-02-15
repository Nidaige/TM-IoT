import numpy as np
from time import time
#import tmu
#from tmu.models.classification.vanilla_classifier import TMClassifier
import tmu.tmu as tmu
import tmu.tmu.datasets
import tmu.tmu.models.classification.vanilla_classifier
import os
#import numpy as np
import pandas
#import pycuda
import pandas as pd
import tensorflow
#import pycuda
import struct
import random
import math
#import openpyxl
import wandb
from sklearn.datasets import fetch_openml


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


''' Main run begins here'''

s = 15.0
T = 5000
clauses = 2500
max_literals = 32
epochs = 25

import tmu.datasets

kdd = tmu.datasets.KDD99(split=0.7, shuffle=True, booleanize=True, max_bits_per_literal=max_literals, balance=True,
                         class_size_cutoff=500)
dataset = kdd.retrieve_dataset()
X_train = dataset["x_train"]
Y_train = dataset["y_train"]
X_test = dataset["x_test"]
Y_test = dataset["y_test"]
all_classes = dataset["target"]

Run_Name = "New_booleanizer_test: S:" + str(s) + "T:" + str(T) + "Clauses:" + str(clauses) + "Max_literals:" + str(
    max_literals) + "Epochs:" + str(epochs)
Project_name = "IoTSecurity-TMUTesting-KDD"

config = dict(
    Forget_rate=s,
    Threshold=T,
    Clauses=clauses,
    Max_literals=max_literals,
    Epochs=epochs
)

# If you are changing dataset, change the "project" before starting new runs. "name" determines the run name, not project.
wandb.init(project=Project_name, name=Run_Name, config=config)
cm = {}
for label in all_classes:
    cm[label] = {}
for label in all_classes:
    for label2 in all_classes:
        cm[label][label2] = 0

print("\nAccuracy over " + str(epochs) + " epochs:\n")
if torch.cuda.is_available():
    tm = TMClassifier(number_of_clauses=clauses, T=T, s=s, max_included_literals=max_literals, platform='CUDA',
                      weighted_clauses=True)
    print("RUNNING TSETLIN MACHINE ON CUDA GPU")
    print(torch.cuda.device_count())

else:
    tm = TMClassifier(number_of_clauses=clauses, T=T, s=s, max_included_literals=max_literals, platform='CPU',
                      weighted_clauses=True)
    print("RUNNING TSETLIN MACHINE ON CPU")
for i in range(epochs):
    start_training = time()
    tm.fit(X_train, Y_train)
    stop_training = time()

    start_testing = time()
    pred = tm.predict(X_test)
    result = 100 * (pred == Y_test).mean()
    calculate_results(pred, Y_test, all_classes)
    loss = 100 - result
    stop_testing = time()
    traintime = stop_training - start_training
    testtime = stop_testing - start_testing
    wandb.log({"Accuracy": result, "Loss": loss, "Training duration": traintime, "Testing Duration": testtime})
    print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i + 1, result, traintime, testtime))
Prediction = tm.predict(X_test)

for i, p in enumerate(Prediction):
    cm[all_classes[p]][all_classes[Y_test[i]]] += 1
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
