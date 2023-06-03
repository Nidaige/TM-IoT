# TM-IoT
This is the code repository for the master's Thesis "Advancing IoT Security with Tsetlin Machines: A Resource-Efficient Anomaly Detection Approach"
by Henning Blomfeldt Thorsen and Ole Gunvaldsen.

# About the repository
The code used to run the different tests is found in TMU.py
The file contains the function calls to load each dataset, as well as the training/testing loop.
To run a specific one, uncomment the relevant lines of code, specify your hyperparameters, and run.
The code also logs to Weights and Biases.

The code used to preprocess each dataset is found in datasets.py
The code for each dataset is also made part of CAIR's own tmu library at https://github.com/cair/tmu/tree/main/tmu/data


# Running the code


## Step 1: Install tmu
The install instructions can be found at https://github.com/cair/tmu, but considering that the original implementation used the code structure of CAIR's tmu from February 2023, an older version might be required.

## Step 2: Install tensorflow, pycuda, pytorch, pandas
This step should be fairly simple, use pip install in your chosen environment

## Step 3: Install and log into Weights and Biases
First install weights and biases with pip install wandb
Next, log in to wandb with "wandb login <your wandb API token>".
You'll find your personal login token under user settings on your W&B account.
  
## Step 4: Download the relevant datasets
### KDD99 
  KDD99 is hosted by openML and does not require a manual download
### NSL-KDD: 
  Download link is found at https://www.unb.ca/cic/datasets/nsl.html. You will be required to give some personal information, but this is only so UNB (University of New Brunswick) can see who has accessed their data.
  The current version of THIS code requires that you unzip the dataset file in a folder called "NSL"
### CIC-IDS2017: 
  Download link is found at https://www.unb.ca/cic/datasets/ids-2017.html. You will be required to give some personal information, but this is only so UNB (University of New Brunswick) can see who has accessed their data.
  The current version of THIS code allows you to re-define the path where this dataset is stored, so no specific folder is required. If you want to use it as is, extract into a folder called "Data".
### UNSW-NB15: 
  Download link is found at https://research.unsw.edu.au/projects/unsw-nb15-dataset. 
  The current version of THIS code allows you to re-define the path where this dataset is stored, so no specific folder is required. If you want to use it as is, extract into a folder called "UNSW".
### UNSW-Bot-IoT: 
  Download link is found at https://research.unsw.edu.au/projects/bot-iot-dataset.
  The current version of THIS code allows you to re-define the path where this dataset is stored, so no specific folder is required. If you want to use it as is, extract into a folder called "Bot_IoT".
  
## Step 5: Editing TMU.py before running
  There are a few changes that are necessary before running the code.
  1. Setting the hyperparameters. Most of the parameters are self-explanatory, but specifically wc is "weighted clauses", a feature of tmu.
  2. Choosing the dataset. The code is only set up to run one dataset, so remove the ''' before and after the dataset you want to run.
  
At this point the code should be able to run.
