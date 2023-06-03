import abc
from typing import Dict

import numpy as np

import struct
def binary(num):  # converts a float to binary, 8 bits
    '''
    Function for float to 32-bit binary from here:
    https://stackoverflow.com/questions/16444726/binary-representation-of-float-in-python-bits-not-hex
    '''
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))

class TMUDataset:

    def __init__(self):
        pass

    @abc.abstractmethod
    def booleanizer(self, name, dataset):
        raise NotImplementedError("You should override def threshold()")

    @abc.abstractmethod
    def retrieve_dataset(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError("You should override def retrieve_dataset()")

    def get(self):
        return {k: self.booleanizer(k, v) for k, v in self.retrieve_dataset().items()}

    def get_list(self):
        return list(self.get().values())

# ----------------------------------------------------------------------------------------------------------------    
    
class MNIST(TMUDataset):
    def retrieve_dataset(self) -> Dict[str, np.ndarray]:
        from keras.datasets import mnist
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        return dict(
            x_train=X_train,
            y_train=Y_train,
            x_test=X_test,
            y_test=Y_test
        )
    def booleanizer(self, name, dataset):
        if name.startswith("y"):
            return dataset
        return np.where(dataset.reshape((dataset.shape[0], 28*28)) > 75, 1, 0)

# ----------------------------------------------------------------------------------------------------------------    
    
class KDD99(TMUDataset):
    def __init__(self, split=0.7, shuffle=False, binarize = False, max_bits_per_literal = 32, balance = True, class_size_cutoff = 12000):
        self.split = split
        self.shuffle = shuffle
        self.booleanize = binarize
        self.max_bits_per_literal = max_bits_per_literal
        self.balance = balance
        self.class_size_cutoff = class_size_cutoff

    def retrieve_dataset(self) -> Dict[str, np.ndarray]:
        from sklearn.datasets import fetch_openml
        import math
        print("Loading KDDCup99 dataset...")
        kdd = fetch_openml(name='KDDCup99', version=1)
        data = kdd['data']
        labels = kdd['target']
        
        # If Shuffle flag is set, shuffle the data
        if self.shuffle:
            print("Shuffling data...")
            from random import shuffle
            temp_data = []
            temp_labels = []
            temp_storage = []
            for x in data.iterrows():
                temp_data.append(x[1])

            for y in labels.items():
                temp_labels.append(y[1])

            for i, x in enumerate(temp_data):
                temp_storage.append((temp_data[i],temp_labels[i]))
            shuffle(temp_storage)
            new_data = []
            new_labels = []
            for temp in temp_storage:
                new_data.append(temp[0])
                new_labels.append(temp[1])
            data = new_data
            labels = new_labels

        ### Booleanize ###
        if self.booleanize:
            print("Booleanizing data...")
            data_values = []
            booleanized_data = []
            
            for i, row in enumerate(data):  # numerically iterate through every line of data
                datapoint = []  # empty list to hold the features of each row
                for item in row:  # for each value in a row
                    datapoint.append(item)  # add it to the list of features for this row
                datapoint.append(labels[i])
                data_values.append(datapoint)  # add the final list of features for this row to the processed dataset
                
        
            for item in data_values:  # for each dataset item
                values = item[0:-1]
                label = item[-1]
                bindata = ""  # string to temporarily hold binary representation of the data item
                non_floatable = []
                for feature in values:  # for each value / feature in said item_
                    try:
                        floated = float(feature)
                    except:
                        if feature not in non_floatable:
                            non_floatable.append(feature)
                        floated = float(non_floatable.index(feature))
                    import struct
                    bindata += str(''.join('{:0>8b}'.format(c) for c in struct.pack('!f', floated)))[-self.max_bits_per_literal:]  # concatenate the binary string for each feature to the string representing the item
                booleanized_data.append([*bindata])
            data = booleanized_data
        else:
            data_values = []
            booleanized_data = []
            non_floatable = []
            for i, row in enumerate(data):  # numerically iterate through every line of data
                datapoint = []  # empty list to hold the features of each row
                for item in row:  # for each value in a row
                    floatable = 0.0
                    try:
                        floatable = float(item)
                        datapoint.append(item)
                    except:
                        if item not in non_floatable:
                            non_floatable.append(item)
                        floatable = float(non_floatable.index(item))
                    datapoint.append(floatable)  # add it to the list of features for this row
                data_values.append(datapoint)  # add the final list of features for this row to the processed dataset
            data = data_values
        
        ### Balance ###
        if self.balance:
            print("Balancing data...")
            registry = {}  # Dictionary to keep track of # of each data class
            database = {}  # Dictionary to keep track of the actual data for each entry in each class
            data_thresholded = []  # List to keep data after applying the threshold
            labels_thresholded = []  # List to keep labels after applying the threshold
            
            for new_label in labels:  # populate dictionaries with all keys present in dataset
                if new_label not in reg.keys():
                    registry[new_label] = 0
                    database[new_label] = []
                    
            for index, item in enumerate(data):  # populate dictionaries with data and number of entries respectively
                database[labels[index]].append(item)
                registry[labels[index]] += 1

            for n in range(self.class_size_cutoff): # Get data entries from each class until either i) the threshold is met or ii) there are no more entries in the class
                for key in database.keys():
                    if n < registry[key]:
                        data_thresholded.append(database[key][n])
                        labels_thresholded.append(key)
            data = data_thresholded
            labels = labels_thresholded
        print("Data distribution:")
        print(registry)
        
        ### Converting labels from strings to numbers (using index in set of labels to represent each label) ###
        raw_labels = list(set(labels))  # set of unique labels in dataset
        for l in range(len(labels)):
            lab = labels[l]
            ind = raw_labels.index(lab)
            labels[l] = ind
            
        ### Converting data and labels to np arrays:
        data = np.array(data)
        labels = np.array(labels)
        
        # Split data into training and testing sets with a given split
        print("Dividing data into training and test with a",self.split,"split...")
        N_split = math.floor(self.split*len(data))
        X_train = data[0:N_split]
        Y_train = labels[0:N_split]
        X_test = data[N_split:]
        Y_test = labels[N_split:]

        # Return the processed data - split into training and testing data/labels and the raw string labels
        return dict(
                x_train=X_train,
                y_train=Y_train,
                x_test=X_test,
                y_test=Y_test,
                target=raw_labels
            )

# ----------------------------------------------------------------------------------------------------------------    

class NSLKDD(TMUDataset):
    def __init__(self, dataset_directory, shuffle = True, binarize = True, balance_train_set = True, balance_test_set = True, max_bits_per_literal = 16, class_size_cutoff = 500, limit_to_classes_in_train_set = True, limit_to_classes_in_test_set = True):
        self.shuffle = shuffle
        self.booleanize = binarize
        self.balance_test_set = balance_test_set
        self.balance_train_set = balance_train_set
        self.max_bits_per_literal = max_bits_per_literal
        self.class_size_cutoff = class_size_cutoff
        self.trim_test = limit_to_classes_in_train_set
        self.trim_train = limit_to_classes_in_test_set
        self.dir = str(dataset_directory)
        
    def retrieve_dataset(self) -> Dict[str, np.ndarray]:
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []

        training_file = open(self.dir+"/KDDTrain+.txt").readlines()
        testing_file = open(self.dir+"/KDDTest+.txt").readlines()
        # Load data from training data file
        for line in training_file:
            listy = line.split(",")
            train_data.append(listy[0:-2])
            train_labels.append(listy[-2])
            
        # Load data from testing data file
        for line in testing_file:
            listy = line.split(",")
            test_data.append(listy[0:-2])
            test_labels.append(listy[-2])
        
        '''
        Note: NSLKDD has a weird issue where there are classes in both sets that do not occur in the other (classes in training that aren't in test and vice versa). Optionally these classes can be trimmed by using the "limit_to_classes_in_train_set" and "limit_to_classes_in_test_set" variables
        '''
        # Remove classes from test set that do not appear in train set
        if self.trim_test:
            new_test_data = []
            new_test_labels = []
            for index, item in enumerate(test_data):
                if test_labels[index] in train_labels:
                    new_test_data.append(item)
                    new_test_labels.append(test_labels[index])
            test_data = new_test_data
            test_labels = new_test_labels
            
        if self.trim_train:
            new_train_data = []
            new_train_labels = []
            for index, item in enumerate(train_data):
                if train_labels[index] in test_labels:
                    new_train_data.append(item)
                    new_train_labels.append(train_labels[index])
            train_data = new_train_data
            train_labels = new_train_labels
        
        # If Shuffle flag is set, shuffle the data
        if self.shuffle:
            print("Shuffling data...")
            from random import shuffle
            temp_train_data = []
            temp_train_labels = []
            temp_train_storage = []
            temp_test_data = []
            temp_test_labels = []
            temp_test_storage = []
            
            # Shuffle training data
            for x in train_data:
                temp_train_data.append(x)
            for y in train_labels:
                temp_train_labels.append(y)
            for i, x in enumerate(temp_train_data):
                temp_train_storage.append((temp_train_data[i],temp_train_labels[i]))
            shuffle(temp_train_storage)
            new_train_data = []
            new_train_labels = []
            for temp in temp_train_storage:
                new_train_data.append(temp[0])
                new_train_labels.append(temp[1])
            train_data = new_train_data
            train_labels = new_train_labels
            
            # Shuffle Test data
            for x in test_data:
                temp_test_data.append(x)
            for y in test_labels:
                temp_test_labels.append(y)
            for i, x in enumerate(temp_test_data):
                temp_test_storage.append((temp_test_data[i],temp_test_labels[i]))
            shuffle(temp_test_storage)
            new_test_data = []
            new_test_labels = []
            for temp in temp_test_storage:
                new_test_data.append(temp[0])
                new_test_labels.append(temp[1])
            test_data = new_test_data
            test_labels = new_test_labels
            
        ### Booleanize ###
        if self.booleanize:
            print("Booleanizing data...")
            ##### Training data #####
            data_values = []
            booleanized_data = []
            
            for i, row in enumerate(train_data):  # numerically iterate through every line of data
                datapoint = []  # empty list to hold the features of each row
                for item in row:  # for each value in a row
                    datapoint.append(item)  # add it to the list of features for this row
                datapoint.append(train_labels[i])
                data_values.append(datapoint)  # add the final list of features for this row to the processed dataset
                
        
            for item in data_values:  # for each dataset item
                values = item[0:-1]
                label = item[-1]
                temp_bin = ""  # string to temporarily hold binary representation of the data item
                non_floatable = []
                for feature in values:  # for each value / feature in said item_
                    try:
                        floated = float(feature)
                    except:
                        if feature not in non_floatable:
                            non_floatable.append(feature)
                        floated = float(non_floatable.index(feature))
                    import struct
                    temp_bin += str(''.join('{:0>8b}'.format(c) for c in struct.pack('!f', floated)))[-self.max_bits_per_literal:]  # concatenate the binary string for each feature to the string representing the item
                booleanized_data.append([*temp_bin])
            train_data = booleanized_data
            
            ##### Test Data #####            
            data_values = []
            booleanized_data = []
            
            for i, row in enumerate(test_data):  # numerically iterate through every line of data
                datapoint = []  # empty list to hold the features of each row
                for item in row:  # for each value in a row
                    datapoint.append(item)  # add it to the list of features for this row
                datapoint.append(test_labels[i])
                data_values.append(datapoint)  # add the final list of features for this row to the processed dataset
                
        
            for item in data_values:  # for each dataset item
                values = item[0:-1]
                label = item[-1]
                temp_bin = ""  # string to temporarily hold binary representation of the data item
                non_floatable = []
                for feature in values:  # for each value / feature in said item_
                    try:
                        floated = float(feature)
                    except:
                        if feature not in non_floatable:
                            non_floatable.append(feature)
                        floated = float(non_floatable.index(feature))
                    import struct
                    temp_bin += str(''.join('{:0>8b}'.format(c) for c in struct.pack('!f', floated)))[-self.max_bits_per_literal:]  # concatenate the binary string for each feature to the string representing the item
                booleanized_data.append([*temp_bin])
            test_data = booleanized_data
        
        else:
            ##### Training data ####
            data_values = []
            for i, row in enumerate(train_data):  # numerically iterate through every line of data
                datapoint = []  # empty list to hold the features of each row
                non_floatable = []
                for item in row:  # for each value in a row
                    floatable = 0.0
                    try:
                        floatable = float(item)
                    except:
                        if item not in non_floatable:
                            non_floatable.append(item)
                        floatable = non_floatable.index(item)
                    datapoint.append(floatable)  # add it to the list of features for this row
                data_values.append(datapoint)  # add the final list of features for this row to the processed dataset    
            train_data = data_values
            
            
            ### Test data ###
            data_values = []
            for i, row in enumerate(test_data):  # numerically iterate through every line of data
                datapoint = []  # empty list to hold the features of each row
                non_floatable = []
                for item in row:  # for each value in a row
                    floatable = 0.0
                    try:
                        floatable = float(item)
                    except:
                        if item not in non_floatable:
                            non_floatable.append(item)
                        floatable = non_floatable.index(item)
                    datapoint.append(floatable)  # add it to the list of features for this row
                data_values.append(datapoint)  # add the final list of features for this row to the processed dataset    
            test_data = data_values
            
        ### Balance ###
        if self.balance_train_set:
            print("Balancing training data...")
            registry_train = {}  # Dictionary to keep track of # of each data class
            registry_train_balanced = {}  # Dictionary to keep track of # of each data class after balancing
            database = {}  # Dictionary to keep track of the data in each class
            data_thresholded = []  # List for data entries post balancing
            labels_thresholded = []  # List for labels post balancing
            
            for new_label in train_labels:  # populate dictionaries with all keys present in dataset
                if new_label not in registry_train.keys():
                    registry_train[new_label] = 0
                if new_label not in database.keys():
                    database[new_label] = []
                    
            registry_train_balanced = registry_train  
            
            for index, item in enumerate(train_data):
                database[train_labels[index]].append(item)
                registry_train[train_labels[index]] += 1

            for n in range(self.class_size_cutoff): # get up to the cutoff of each class, taking all if there are not enough
                for key in database.keys():
                    if n < len(database[key]):  
                        try:
                            data_thresholded.append(database[key][n])
                            labels_thresholded.append(key)
                            registry_train_balanced[key] += 1
                        except:
                            print(registry_train)
                            k = {}
                            for key in database.keys():
                                k[key] = len(database[key])
                            print(k)
                            exit()
                        
            train_data = data_thresholded
            train_labels = labels_thresholded
            print("Raw Training Data Distribution:")
            print(registry_train)
            print("Balanced Training Data Distribution:")
            print(registry_train_balanced)
        
        if self.balance_test_set:
            print("Balancing test data...")
            registry_test = {}
            registry_test_balanced = {}
            database = {}
            data_thresholded = []  # List for data entries post balancing
            labels_thresholded = []  # List for labels post balancing
            
            for new_label in test_labels:  # populate dictionaries with all keys present in dataset
                if new_label not in registry_test.keys():
                    registry_test[new_label] = 0
                    database[new_label] = []
                    
            registry_test_balanced = registry_train
            for index, item in enumerate(test_data):
                database[test_labels[index]].append(item)
                registry_test[test_labels[index]] += 1

            for n in range(self.class_size_cutoff): # get up to the cutoff of each class, taking all if there are not enough
                for key in database.keys():
                    if n < registry_test[key]:  
                        data_thresholded.append(database[key][n])
                        labels_thresholded.append(key)
                        registry_test_balanced[key] += 1
                        
            test_data = data_thresholded
            test_labels = labels_thresholded
            print("Raw Training Data Distribution:")
            print(registry_test)
            print("Balanced Training Data Distribution:")
            print(registry_test_balanced)
            
        ### Converting labels from strings to numbers ###
        raw_labels = list(set(train_labels + test_labels))
        
        for l in range(len(train_labels)):
            lab = train_labels[l]
            ind = raw_labels.index(lab)
            train_labels[l] = ind
            
        for l in range(len(test_labels)):
            lab = test_labels[l]
            ind = raw_labels.index(lab)
            test_labels[l] = ind
            
        ### Converting data and labels to np arrays:
        X_train = np.array(train_data)
        Y_train = np.array(train_labels)
        X_test = np.array(test_data)
        Y_test = np.array(test_labels)        
        
        return dict(
            x_train=X_train,
            y_train=Y_train,
            x_test=X_test,
            y_test=Y_test,
            target = raw_labels
        )

# ----------------------------------------------------------------------------------------------------------------    
    
class CICIDS2017(TMUDataset):
    def __init__(self, dataset_filepaths, split=0.7, shuffle=True, balance=True, binarize=True, bits_per_entry = 16, data_category_threshold = 12000):
        self.split = split
        self.shuffle = shuffle
        self.balance = balance
        self.bits_per_entry = bits_per_entry
        self.data_category_threshold = data_category_threshold
        self.binarize = binarize
        self.paths = dataset_filepaths


    def retrieve_dataset(self) -> Dict[str, np.ndarray]:
        import math
        import numpy as np
        database = {}  # dictionary to hold all the data
        registry = {}  # dictionary to hold the number of data in each category
        registry_balanced = {}  # dictionary to hold the number of data in each category
        data_thresholded = []
        labels_thresholded = []

        # Read files and create data
        for path in self.paths:
            database, registry = self.load_and_binarize(path, self.bits_per_entry, database, registry)
        print("Data distribution:")
        print(registry)
        if self.balance:
            registry_balanced = {}
            for key in registry.keys():
                registry_balanced[key] = 0
            for n in range(self.data_category_threshold):
                for key in database.keys():
                    number_of_that_class = registry[key]
                    if n < number_of_that_class:  # if there aren't enough entries in <key> category to reach n, just take what's there
                        data_thresholded.append(database[key][n])
                        labels_thresholded.append(key)
                        registry_balanced[key] += 1
                        

        print("Data distribution after balancing:")
        print(registry_balanced)
        
        # shuffle the dataset order while keeping data entries and their labels paralell
        if self.shuffle:
            data_thresholded, labels_thresholded = self.shuffle_dataset(all_data)


            
        # make a set of all the labels in the dataset to translate the literal String values of the labels to integer values representing each class
        labels_all_data_set = list(set(labels_thresholded))  # create set of ALL string-labels
        for i in range(len(labels_thresholded)):  # for each element
            labels_thresholded[i] = labels_all_data_set.index(labels_thresholded[i])  # assign the true index to each data label Y

        # Convert data and labels to numpy arrays for the Tsetlin Machine object
        data_thresholded = np.array(data_thresholded).astype(float)
        labels_thresholded = np.array(labels_thresholded)
        # split data into training and testing sets with the given split
        count = len(data_thresholded)
        split = self.split
        N_split = math.floor(count * split)
        X_train = data_thresholded[0:N_split]
        Y_train = labels_thresholded[0:N_split]

        X_test = data_thresholded[N_split:]
        Y_test = labels_thresholded[N_split:]



        return dict(
            x_train=X_train,
            y_train=Y_train,
            x_test=X_test,
            y_test=Y_test,
            target = labels_all_data_set
        )

    def load_and_binarize(self, path, max_bits, database, registry):
        '''
        This function takes a path to a csv file, the number of bits to use for each dataset value, a dictionary with dataset items in lists under each category as keys, and a dictionary with the number of elements in each dataset category, and returns updated versions of the two dictionaries based on the data read from the file.
        '''
        import pandas
        data_values = []
        data = pandas.read_csv(path)  # read data csv
        # Duplicate the registry, add new keys to it
        reg = registry  # dictionary to keep track of the # of entries per class
        db = database  # dictionary to keep track of the data entries per class
        
        # populate the dictionaries with any labels from the new file that weren't already tracked
        for new_label in list(data[' Label'].unique()):
            if new_label not in reg.keys():
                reg[new_label] = 0
                db[new_label] = []

        # extract each line of data to a python list of elements
        for num in range(len(data) - 1):  # numerically iterate through every line of data
            row = data.loc[num]  # get the data for each row
            datapoint = []  # empty list to hold the features of each row
            for item in row:  # for each value in a row
                datapoint.append(item)  # add it to the list of features for this row
            data_values.append(datapoint)  # add the final list of features for this row to the processed dataset
    
        # Binarize data
        if self.binarize:
            for item in data_values:  # for each dataset item
                values = item[0:-1]  # extract data
                label = item[-1]  # extract label
                temp_bin = ""  # string to temporarily hold binary representation of the data item
                non_floatable = []  # list of non-floatable values
            
                # Convert to float
                for feature in values:
                    try:  
                        floated = float(feature)
                    except:
                        if feature not in non_floatable:  # if value could not be converted to float, instead use its index in a set of unfloatable values.
                            non_floatable.append(feature)
                        floated = float(non_floatable.index(feature))
                    temp_bin += str(binary(float(feature)))[ -max_bits:]  # concatenate the binary string for each feature to the string representing the item
                db[label].append([*temp_bin])
                registry[label] += 1
            return (db, registry)  # returns tuple of binary representation of data item and its label as an integer, and the string of labels for later decoding.
        
        # If not binarize:
        else:
            for item in data_values:  # for each dataset item
                values = item[0:-1]
                label = item[-1]
                temp = []  # string to temporarily hold representation of the data item
                for feature in values:  # for each value / feature in said item_
                    temp.append(feature)  # concatenate the binary string for each feature to the string representing the item
                db[label].append(temp)
                registry[label] += 1
            return (db, registry)  # returns tuple of binary representation of data item and its label as an integer, and the string of labels for later decoding.

    def binary(self, num):  # converts a float to binary, 8 bits
        '''
        Function for float to binary from here:
        https://stackoverflow.com/questions/16444726/binary-representation-of-float-in-python-bits-not-hex
        '''
        return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))

    def shuffle_dataset(self, dataset):
        '''
        This function shuffles a list of tuples using a recursive method.
        '''
        import random
        import math
        output_values_list = []  # list for shuffled values
        output_labels_list = []  # list for shuffled labels
        tempdata = dataset  # copy dataset for further manipulation
        while len(tempdata[0]) > 0:  # as long as there is data in the dataset, keep going
            index = math.floor(random.random() * len(tempdata[0]))  # randomly select an element by index
            output_values_list.append(
                np.array(
                    tempdata[0][index]))  # copy the element in each slot (value, label, string label) to the output
            output_labels_list.append(tempdata[1][index])  # --||--
            for c in range(2):  # for each slot (value, label, string label):
                # temp = tempdata[c][index]  # copy as a buffer variable
                tempdata[c][index] = tempdata[c][-1]  # copy final element in array to the [index] location
                tempdata[c] = tempdata[c][0:-1]  # overwrite dataset with dataset except last element (which is now copied to location[index])
        return (output_values_list, output_labels_list)

# ----------------------------------------------------------------------------------------------------------------

class UNSW_NB15(TMUDataset):
    def __init__(self, paths, split=0.7, shuffle=True, balance=True, binarize=True, bits_per_entry = 16, data_category_threshold = 12000):
        self.split = split
        self.shuffle = shuffle
        self.balance = balance
        self.bits_per_entry = bits_per_entry
        self.data_category_threshold = data_category_threshold
        self.binarize = binarize
        self.paths = paths


    def retrieve_dataset(self) -> Dict[str, np.ndarray]:
        import math
        import numpy as np
        maximum_bits = self.bits_per_entry  # max bits to use for each dataset value
        database = {}  # dictionary to hold all the data
        registry = {}  # dictionary to hold the number of data in each category
        registry_balanced = {}  # dictionary to hold the number of data in each category after balancing
        data_thresholded = []
        labels_thresholded = []
        
        # Read files and create data
        for path in self.paths:
            print("Loading data from", path)
            database, registry = self.load_and_binarize(path, maximum_bits, database, registry)
        print("Dataset Distribution")
        print(registry)
        if self.balance:
            registry_balanced = {}
            for key in registry.keys():
                registry_balanced[key] = 0
            for n in range(self.data_category_threshold):
                for key in database.keys():
                    number_of_that_class = registry[key]
                    if n < number_of_that_class:  # if there aren't enough entries in <key> category to reach n, just take what's there
                        data_thresholded.append(database[key][n])
                        labels_thresholded.append(key)
                        registry_balanced[key] += 1
            print("Dataset Distribution after balancing:")
            print(registry_balanced)
            data = data_thresholded
            labels = labels_thresholded
        else:
            data = []
            labels = []
            for key in database.keys():
                for item in database[key]:
                    data.append(item)
                    labels.append(key)
        # shuffle the dataset order while keeping data entries and their labels paralell
        if self.shuffle:
            X_all_data, Y_all_data = self.shuffle_dataset([data, labels])
        else:
            X_all_data = data
            Y_all_data = labels
        # make a set of all the labels in the dataset to translate the literal String values of the labels to integer values representing each class
        labels_all_data_set = list(set(Y_all_data))  # create set of ALL string-labels
        for i in range(len(Y_all_data)):  # for each element
            Y_all_data[i] = labels_all_data_set.index(Y_all_data[i])  # assign the true index to each data label Y

        # Convert data and labels to numpy arrays for the Tsetlin Machine object
        X_all_data = np.array(X_all_data).astype(float)
        Y_all_data = np.array(Y_all_data)
        # split data into training and testing sets with the given split
        count = len(X_all_data)
        split = self.split
        N_split = math.floor(count * split)

        X_train = X_all_data[0:N_split]
        Y_train = Y_all_data[0:N_split]

        X_test = X_all_data[N_split:]
        Y_test = Y_all_data[N_split:]

        return dict(
            x_train=X_train,
            y_train=Y_train,
            x_test=X_test,
            y_test=Y_test,
            target = labels_all_data_set
        )

    def load_and_binarize(self, path, max_bits, database, registry):
        '''
        This function takes a path to a csv file, the number of bits to use for each dataset value, a dictionary with dataset items in lists under each category as keys, and a dictionary with the number of elements in each dataset category, and returns updated versions of the two dictionaries based on the data read from the file.
        '''
        import pandas
        data_values = []
        data = pandas.read_csv(path)  # read data csv
        labels = data.iloc[:,-2]
        data = data.iloc[:,:-3]
        unique_labels = []
        newlabels = []
        for index, item in enumerate(labels):
            if type(item) == float:
                newb ="Normal"
            else:
                for a in ["Reconnaissance", "Shellcode", "Backdoor", "Fuzz"]:
                    if a in item:
                        newb = a
                    else:
                        newb = item
                
            if newb != "target":
                newlabels.append(newb)
            
            if labels[index] not in unique_labels:
                unique_labels.append(newb)
                
        labels = newlabels
        # Duplicate the registry, add new keys to it
        reg = registry
        db = database
        for new_label in unique_labels:
            if new_label not in reg.keys():
                reg[new_label] = 0
                db[new_label] = []

        for num in range(len(data) - 1):  # numerically iterate through every line of data
            row = data.loc[num]  # get the data for each row
            datapoint = []  # empty list to hold the features of each row
            for item in row:  # for each value in a row
                datapoint.append(item)  # add it to the list of features for this row
            data_values.append(datapoint)  # add the final list of features for this row to the processed dataset
        if self.binarize:
            for index, item in enumerate(data_values):  # for each dataset item
                values = item
                label = labels[index]
                temp_bin = ""  # string to temporarily hold binary representation of the data item
                non_floatable = []
                for feature in values:  # for each value / feature in said item_
                    try:
                        floated = float(feature)
                    except:
                        if feature not in non_floatable:
                            non_floatable.append(feature)
                        floated = float(non_floatable.index(feature))
                    import struct
                    temp_bin += str(''.join('{:0>8b}'.format(c) for c in struct.pack('!f', floated)))[-self.bits_per_entry:]  # concatenate the binary string for each feature to the string representing the item
                db[label].append([*temp_bin])
                registry[label] += 1
            return (db, registry)  # returns tuple of binary representation of data item and its label as an integer, and the string of labels for later decoding.
        else:
            for item in data_values:  # for each dataset item
                values = item[0:-1]
                label = item[-1]
                temp = []  # string to temporarily hold binary representation of the data item
                for feature in values:  # for each value / feature in said item_
                    temp.append(feature)  # concatenate the binary string for each feature to the string representing the item
                db[label].append(temp)
                registry[label] += 1
            return (db, registry)  # returns tuple of binary representation of data item and its label as an integer, and the string of labels for later decoding.

    def binary(self, num):  # converts a float to binary, 8 bits
        '''
        Function for float to binary from here:
        https://stackoverflow.com/questions/16444726/binary-representation-of-float-in-python-bits-not-hex
        '''
        return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))

    def shuffle_dataset(self, dataset):
        '''
        This function shuffles a list of tuples using a recursive method.
        '''
        import random
        import math
        output_values_list = []  # list for shuffled values
        output_labels_list = []  # list for shuffled labels
        tempdata = dataset  # copy dataset for further manipulation
        while len(tempdata[0]) > 0:  # as long as there is data in the dataset, keep going
            index = math.floor(random.random() * len(tempdata[0]))  # randomly select an element by index
            output_values_list.append(
                np.array(
                    tempdata[0][index]))  # copy the element in each slot (value, label, string label) to the output
            output_labels_list.append(tempdata[1][index])  # --||--
            for c in range(2):  # for each slot (value, label, string label):
                # temp = tempdata[c][index]  # copy as a buffer variable
                tempdata[c][index] = tempdata[c][-1]  # copy final element in array to the [index] location
                tempdata[c] = tempdata[c][0:-1]  # overwrite dataset with dataset except last element (which is now copied to location[index])
        return (output_values_list, output_labels_list)

#---------------------------------------------------------------------------------

class Bot_IoT(TMUDataset):
    def __init__(self, paths, split=0.7, shuffle=True, balance=True, binarize=True, bits_per_entry = 16, data_category_threshold = 12000):
        self.split = split
        self.shuffle = shuffle
        self.balance = balance
        self.bits_per_entry = bits_per_entry
        self.data_category_threshold = data_category_threshold
        self.binarize = binarize
        self.paths = paths

    def retrieve_dataset(self) -> Dict[str, np.ndarray]:
        import math
        import numpy as np
        maximum_bits = self.bits_per_entry  # max bits to use for each dataset value
        database = {}  # dictionary to hold all the data
        registry = {}  # dictionary to hold the number of data in each category
        registry_balanced = {}
        data_thresholded = []
        labels_thresholded = []

        # Read files and create data
        for path in self.paths:
            print("Loading data from", path)
            database, registry = self.iot_data_to_binary_list(path, maximum_bits, database, registry)
        print("Data distribution:")
        print(registry)
        
        if self.balance:
            registry_balanced = {}
            for key in registry.keys():
                registry_balanced[key] = 0
            for n in range(self.data_category_threshold):
                for key in database.keys():
                    number_of_that_class = registry[key]
                    if n < number_of_that_class:  # if there aren't enough entries in <key> category to reach n, just take what's there
                        data_thresholded.append(database[key][n])
                        labels_thresholded.append(key)
                        registry_balanced[key] += 1
            data = data_thresholded
            labels = labels_thresholded   
        else:
            data = []
            labels = []
            for key in database.keys():
                for item in database[key]:
                    data.append(item)
                    labels.append(key)
        # shuffle the dataset order while keeping data entries and their labels paralell
        if self.shuffle:
            print("shuffling...")
            X_all_data, Y_all_data = self.shuffle_dataset([data, labels])
        else:
            X_all_data = data_thresholded
            Y_all_data = labels_thresholded

        # make a set of all the labels in the dataset to translate the literal String values of the labels to integer values representing each class
        labels_all_data_set = list(set(Y_all_data))  # create set of ALL string-labels
        for i in range(len(Y_all_data)):  # for each element
            Y_all_data[i] = labels_all_data_set.index(Y_all_data[i])  # assign the true index to each data label Y

        # Convert data and labels to numpy arrays for the Tsetlin Machine object
        X_all_data = np.array(X_all_data).astype(float)
        Y_all_data = np.array(Y_all_data)
        # split data into training and testing sets with the given split
        count = len(X_all_data)
        split = self.split
        N_split = math.floor(count * split)

        X_train = X_all_data[0:N_split]
        Y_train = Y_all_data[0:N_split]

        X_test = X_all_data[N_split:]
        Y_test = Y_all_data[N_split:]

        return dict(
            x_train=X_train,
            y_train=Y_train,
            x_test=X_test,
            y_test=Y_test,
            target = labels_all_data_set
        )

    def iot_data_to_binary_list(self, path, max_bits, database, registry):
        '''
        This function takes a path to a csv file, the number of bits to use for each dataset value, a dictionary with dataset items in lists under each category as keys, and a dictionary with the number of elements in each dataset category, and returns updated versions of the two dictionaries based on the data read from the file.
        '''
        import pandas
        data_values = []
        data = pandas.read_csv(path)  # read data csv
        labels = data.iloc[:,-2]
        data = data.iloc[:,:-3]
        unique_labels = []
        newlabels = []
        for index, item in enumerate(labels):
            if type(item) == float:
                newb ="Normal"
            else:
                newb = item
            if newb != "target":
                newlabels.append(newb)
            
            if labels[index] not in unique_labels:
                unique_labels.append(newb)
                
        labels = newlabels
        # Duplicate the registry, add new keys to it
        reg = registry
        db = database
        for new_label in unique_labels:
            if new_label not in reg.keys():
                reg[new_label] = 0
                db[new_label] = []

        for num in range(len(data) - 1):  # numerically iterate through every line of data
            row = data.loc[num]  # get the data for each row
            datapoint = []  # empty list to hold the features of each row
            for item in row:  # for each value in a row
                datapoint.append(item)  # add it to the list of features for this row
            data_values.append(datapoint)  # add the final list of features for this row to the processed dataset
        if self.binarize:
            for index, item in enumerate(data_values):  # for each dataset item
                values = item
                label = labels[index]
                temp_bin = ""  # string to temporarily hold binary representation of the data item
                non_floatable = []
                for feature in values:  # for each value / feature in said item_
                    try:
                        floatable = float(feature)
                    except:
                        if feature not in non_floatable:
                            non_floatable.append(feature)
                        floatable = float(non_floatable.index(feature))
                    import struct
                    temp_bin += str(''.join('{:0>8b}'.format(c) for c in struct.pack('!f', floatable)))[-self.bits_per_entry:]  # concatenate the binary string for each feature to the string representing the item
                db[label].append([*temp_bin])
                registry[label] += 1
            return (db, registry)  # returns tuple of binary representation of data item and its label as an integer, and the string of labels for later decoding.
        else:
            for item in data_values:  # for each dataset item
                values = item[0:-1]
                label = item[-1]
                temp = []  # string to temporarily hold binary representation of the data item
                for feature in values:  # for each value / feature in said item_
                    temp.append(feature)  # concatenate the binary string for each feature to the string representing the item
                db[label].append(temp)
                registry[label] += 1
            return (db, registry)  # returns tuple of binary representation of data item and its label as an integer, and the string of labels for later decoding.

    def binary(self, num):  # converts a float to binary, 8 bits
        '''
        Function for float to binary from here:
        https://stackoverflow.com/questions/16444726/binary-representation-of-float-in-python-bits-not-hex
        '''
        return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))

    def shuffle_dataset(self, dataset):
        '''
        This function shuffles a list of tuples using a recursive method.
        '''
        import random
        import math
        output_values_list = []  # list for shuffled values
        output_labels_list = []  # list for shuffled labels
        tempdata = dataset  # copy dataset for further manipulation
        while len(tempdata[0]) > 0:  # as long as there is data in the dataset, keep going
            index = math.floor(random.random() * len(tempdata[0]))  # randomly select an element by index
            output_values_list.append(
                np.array(
                    tempdata[0][index]))  # copy the element in each slot (value, label, string label) to the output
            output_labels_list.append(tempdata[1][index])  # --||--
            for c in range(2):  # for each slot (value, label, string label):
                # temp = tempdata[c][index]  # copy as a buffer variable
                tempdata[c][index] = tempdata[c][-1]  # copy final element in array to the [index] location
                tempdata[c] = tempdata[c][0:-1]  # overwrite dataset with dataset except last element (which is now copied to location[index])
        return (output_values_list, output_labels_list)
