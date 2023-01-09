import os
import numpy as np
import pandas
import pycuda
import pandas as pd
import pycuda
import struct
from PyTsetlinMachineCUDA.tm import *
import random
import math
import openpyxl


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
    while len(tempdata[0])>0:  # as long as there is data in the dataset, keep going
        index = math.floor(random.random()*len(tempdata[0]))  # randomly select an element by index
        output_values_list.append(np.array(tempdata[0][index]))  # copy the element in each slot (value, label, string label) to the output
        output_labels_list.append(tempdata[1][index])  # --||--
        for c in range(2):  # for each slot (value, label, string label):
            #temp = tempdata[c][index]  # copy as a buffer variable
            tempdata[c][index] = tempdata[c][-1]  # copy final element in array to the [index] location
            tempdata[c] = tempdata[c][0:-1]  # overwrite dataset with dataset except last element (which is now copied to location[index])
    print("Shuffling done")
    return (output_values_list, output_labels_list)

def iot_data_to_binary_list(path, max_bits, database, registry):
    '''
    This function takes a path to a csv file, the number of bits to use for each dataset value, a dictionary with dataset items in lists under each category as keys, and a dictionary with the number of elements in each dataset category, and returns updated versions of the two dictionaries based on the data read from the file.
    '''
    print("Binarizing...")
    data_values = []
    data = pandas.read_csv(path)  # read data csv
    # Duplicate the registry, add new keys to it
    reg = registry
    db = database
    for new_label in list(data[' Label'].unique()):
        if new_label not in reg.keys():
            reg[new_label] = 0
            db [new_label] = []
    
    for num in range(len(data)-1):  # numerically iterate through every line of data
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
            rowie+=str(binary(float(feature)))[-max_bits:]  # concatenate the binary string for each feature to the string representing the item
        db[label].append([*rowie])
        registry[label] += 1
    print("Binarizing done")
    return (db, registry)  # returns tuple of binary representation of data item and its label as an integer, and the string of labels for later decoding.



''' Main run begins here'''
data_paths = ["data/Wednesday-workingHours.pcap_ISCX.csv", "data/Monday-WorkingHours.pcap_ISCX.csv"]  #  path to data files
maximum_bits = 16  # max bits to use for each dataset value
max_data = 100000  # maximum number of dataset entries to process
minimum_data_in_category = 5000  # threshold for dataset category size for it to be used in determining number of elements per dataset to use
all_data_dict = {}  # dictionary to hold all the data
class_registry = {}  # dictionary to hold the number of data in each category
all_data = [[],[]]  # list of two lists to hold the binarized data and its label respectively
smallest_count = 0  # initialize variable for size of smallest dataset category
confusion_matrix = {}  # dictionary to keep the values for the confusion matrix
# Read files and create data
for path in data_paths:
    print("Pre-processing data from", path)
    all_data_dict, class_registry = iot_data_to_binary_list(path, maximum_bits, all_data_dict, class_registry)
    
# Count the size of each category above the threshold and find the size of the smallest one
counts = []
for key in class_registry.keys():
    number_of_that_class = class_registry[key]  # get registry data for <key> category
    if number_of_that_class > minimum_data_in_category:  # if number of entries for <key> category is above threshold, use it in list of counts
        counts.append(number_of_that_class)
smallest_count = min(counts)  # get smallest count above threshold
# get that many elements from each category. If not enough elements in a given category, just take what is there.
# Limit total number of elements to the max data param set earlier
for n in range(min(math.floor(max_data/len(class_registry.keys())),smallest_count)): 
    for key in all_data_dict.keys():
        number_of_that_class = class_registry[key]
        if n < number_of_that_class:  # if there aren't enough entries in <key> category to reach n, just take what's there
            all_data[0].append(all_data_dict[key][n])
            all_data[1].append(key)
            confusion_matrix[key] = {}
# initialize confusion matrix keys
for key in confusion_matrix.keys():
    for key2 in confusion_matrix.keys():
        confusion_matrix[key][key2] = 0

print("data distribution:")
print(class_registry)
# shuffle the dataset order while keeping data entries and their labels paralell
X_all_data, Y_all_data = shuffle_dataset(all_data) 

print("Converting String labels to numbers...")
# make a set of all the labels in the dataset to translate the literal String values of the labels to integer values representing each class
labels_all_data_set = list(set(Y_all_data))  # create set of ALL string-labels
for i in range(len(Y_all_data)):  # for each element
    Y_all_data[i] = labels_all_data_set.index(Y_all_data[i])  # assign the true index to each data label Y


print("Done converting labels.")
print("Converting to numpy arrays...")
# Convert data and labels to numpy arrays for the Tsetlin Machine object
X_all_data = np.array(X_all_data).astype(float)
Y_all_data = np.array(Y_all_data)
print("Done converting to numpy arrays")
# split data into training and testing sets with the given split
count = len(X_all_data)
split = 0.7
print("Splitting into training/test with a ", 100*split,"% split...")
X_train = X_all_data[0:math.floor(count*split)]
Y_train = Y_all_data[0:math.floor(count*split)]

X_test = X_all_data[math.floor(count*split):]
Y_test = Y_all_data[math.floor(count*split):]
print("Done splitting")
print("Initializing variables and starting TM...")
# initialize hyperparameters and start the training loop. Multiple values for S and T will result in a full permutation, useful for grid search for param optimization.
S = [10]  # S-value
Clauses = 25000  # number of clauses to generate / to make each classification vote
T = [30]  # T-value
Epochs = 1000
Batch_size = 100
print("# of labels: ",len(labels_all_data_set), labels_all_data_set)
for s_ in S:
    for t_ in T:
        # Start training
        print("Running clauses:"+str(Clauses)+", T:"+str(t_)+", S:"+str(s_))  # status report to the console
        tm = MultiClassTsetlinMachine(Clauses, t_, s_, boost_true_positive_feedback=0)  # define the TM with above params
        tm.fit(X_train, Y_train, epochs = Epochs, batch_size = Batch_size)  # train the TM for 50 epochs on training data
        print("Training done, predicting...")
        # Training done, predict for each piece of test data
        Prediction = tm.predict(X_test)
        print("Predictions done, calculating score---")
        Total = 0
        Correct = 0
        conf_matr = confusion_matrix
        # For each test data item, check if correct prediction
        for test_data_sample in range(len(X_test)):
            Total += 1
            if Prediction[test_data_sample] == Y_test[test_data_sample]:  # if correct guess:
                Correct += 1
            conf_matr[labels_all_data_set[Prediction[test_data_sample]]][labels_all_data_set[Y_test[test_data_sample]]] += 1  # update confusion matrix for the prediction and truth value
        for key in conf_matr.keys():
            for key2 in conf_matr.keys():
                confusion_matrix[key][key2] = confusion_matrix[key][key2]/Total
        # Export the conusion matrix to an excel spreadsheet.
        dict1 = confusion_matrix
        df = pd.DataFrame(data=dict1, index=list(confusion_matrix.keys()))
        df.to_excel("uberbalance results/new_best/data.xlsx")
        # Print out analytics and write the results + hyperparameter configuration to a text-file. Note: It makes one file per parameter configuration.
        print("count",count)        
        print("Accuracy: ", 100*Correct/Total , "%")
        path = "uberbalance results/new_best/S-"+str(s_)+"_T-"+str(t_)+"_Clauses-"+str(Clauses)+"_Epochs-"+str(Epochs)+"_BatchSize-"+str(Batch_size)+".txt"
        file = open(path,"w")
        file.write("Results: \n")
        file.write("Dataset: Monday and Wednesday.\n")
        file.write("Params: S: "+str(s_)+ ", T:"+str(t_)+", Clauses:"+str(Clauses)+", Epochs: "+str(Epochs)+"\n")
        file.write("Accuracy: "+str(100*Correct/Total)+"% \n")
        file.close()
