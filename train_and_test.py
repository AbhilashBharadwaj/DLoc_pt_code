#!/usr/bin/python
'''
Script for both training and evaluating the DLoc network
Automatically imports the parameters from params.py.
For further details onto which params file to load
read the README in `params_storage` folder.
'''

import warnings

import torch

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
import trainer
from data_loader import load_data
from Generators import *
from joint_model import Enc_2Dec_Network, Enc_Dec_Network
from modelADT import ModelADT
from params import *
from utils import *

torch.manual_seed(0)
np.random.seed(0)
import scipy.io

'''
Defining the paths from where to Load Data.
Assumes that the data is stored in a subfolder called data in the current data folder
'''
BASE_PATH = "/media/datadisk_2/loc_data/wifi/quantenna/datasets_release/"

def choose_dataset(opt_exp):
    dataset_mappings = {
        ######################################Final Simple Space Results################################################
        "rw_to_rw_atk": {
            "train": ["dataset_July18"],
            "test": ["dataset_test_July18"]
        },
        ######################################Final Complex Space Results################################################
        "rw_to_rw": {
            "train": ["dataset_jacobs_July28", "dataset_jacobs_July28_2"],
            "test": ["dataset_test_jacobs_July28", "dataset_test_jacobs_July28_2"]
        },
        #########################################Generalization across Scenarios###########################################
        "rw_to_rw_env2": {
            "train": ["dataset_jacobs_July28", "dataset_jacobs_July28_2", "dataset_jacobs_Aug16_3", "dataset_jacobs_Aug16_4_ref"],
            "test": ["dataset_test_jacobs_Aug16_1"]
        },
        "rw_to_rw_env3": {
            "train": ["dataset_jacobs_July28", "dataset_jacobs_July28_2", "dataset_jacobs_Aug16_1", "dataset_jacobs_Aug16_4_ref"],
            "test": ["dataset_test_jacobs_Aug16_3"]
        },
        "rw_to_rw_env4": {
            "train": ["dataset_jacobs_July28", "dataset_jacobs_July28_2", "dataset_jacobs_Aug16_1", "dataset_jacobs_Aug16_3"],
            "test": ["dataset_test_jacobs_Aug16_4_ref"]
        },
        ######################################Generalization Across Bandwidth##########################################
        ########## Unclear and need to be discussed ###########
        # "rw_to_rw_40": {
        #     "train": ["dataset_jacobs_July28_40"], 
        #     "test": ["dataset_test_jacobs_July28_40"]  
        # },
        # "rw_to_rw_20": {
        #     "train": ["dataset_jacobs_July28_20"],  
        #     "test": ["dataset_test_jacobs_July28_20"]   
        # },
        ######################################Generalization Across Space##########################################
        "data_segment": {
            "train": ["dataset_jacobs_July28", "dataset_jacobs_July28_2"],
            "test": ["dataset_test_jacobs_July28", "dataset_test_jacobs_July28_2"]
        },
    
    }

    if opt_exp.data in dataset_mappings:
        dataset_info = dataset_mappings[opt_exp.data]
        train_paths = [BASE_PATH + name + '.mat' for name in dataset_info["train"]]
        test_paths = [BASE_PATH + name + '.mat' for name in dataset_info["test"]]

        return train_paths, test_paths
    else:
        raise ValueError("Invalid experiment option provided")


if "data" in opt_exp:
    trainpath, testpath = choose_dataset(opt_exp)
    print(f'Experiment {opt_exp.data} started')

'''
Loading Training and Evaluation Data into their respective Dataloaders
'''
# load traning data
B_train,A_train,labels_train = load_data(trainpath[0])

for i in range(len(trainpath)-1):
    f,f1,l = load_data(trainpath[i+1])
    B_train = torch.cat((B_train, f), 0)
    A_train = torch.cat((A_train, f1), 0)
    labels_train = torch.cat((labels_train, l), 0)

labels_train = torch.unsqueeze(labels_train, 1)

train_data = torch.utils.data.TensorDataset(B_train, A_train, labels_train)
train_loader =torch.utils.data.DataLoader(train_data, batch_size=opt_exp.batch_size, shuffle=True)

print(f"A_train.shape: {A_train.shape}")
print(f"B_train.shape: {B_train.shape}")
print(f"labels_train.shape: {labels_train.shape}")
print('# training mini batch = %d' % len(train_loader))

# load testing data
B_test,A_test,labels_test = load_data(testpath[0])

for i in range(len(testpath)-1):
    f,f1,l = load_data(testpath[i+1])
    B_test = torch.cat((B_test, f), 0)
    A_test = torch.cat((A_test, f1), 0)
    labels_test = torch.cat((labels_test, l), 0)

labels_test = torch.unsqueeze(labels_test, 1)

# create data loader
test_data = torch.utils.data.TensorDataset(B_test, A_test, labels_test)
test_loader =torch.utils.data.DataLoader(test_data, batch_size=opt_exp.batch_size, shuffle=False)
print(f"A_test.shape: {A_test.shape}")
print(f"B_test.shape: {B_test.shape}")
print(f"labels_test.shape: {labels_test.shape}")
print('# testing mini batch = %d' % len(test_loader))
print('Test Data Loaded')

'''
Initiate the Network and build the graph
'''

# init encoder
enc_model = ModelADT()
enc_model.initialize(opt_encoder)
enc_model.setup(opt_encoder)

# init decoder1
dec_model = ModelADT()
dec_model.initialize(opt_decoder)
dec_model.setup(opt_decoder)

if opt_exp.n_decoders == 2:
    # init decoder2
    offset_dec_model = ModelADT()
    offset_dec_model.initialize(opt_offset_decoder)
    offset_dec_model.setup(opt_offset_decoder)

    # join all models
    print('Making the joint_model')
    joint_model = Enc_2Dec_Network()
    joint_model.initialize(opt_exp, enc_model, dec_model, offset_dec_model, gpu_ids=opt_exp.gpu_ids)

elif opt_exp.n_decoders == 1:
    # join all models
    print('Making the joint_model')
    joint_model = Enc_Dec_Network()
    joint_model.initialize(opt_exp, enc_model, dec_model, gpu_ids=opt_exp.gpu_ids)

else:
    print('Incorrect number of Decoders specified in the parameters')
    # return -1
    exit(0)

if opt_exp.isFrozen:
    enc_model.load_networks(opt_encoder.starting_epoch_count)
    dec_model.load_networks(opt_decoder.starting_epoch_count)
    if opt_exp.n_decoders == 2:
        offset_dec_model.load_networks(opt_offset_decoder.starting_epoch_count)

# train the model
'''
Trainig the network
'''
trainer.train(joint_model, train_loader, test_loader)

'''
Model Evaluation at the best epoch
'''

epoch = "best"  # int/"best"/"last"
# load network
enc_model.load_networks(epoch, load_dir=eval_name)
dec_model.load_networks(epoch, load_dir=eval_name)
if opt_exp.n_decoders == 2:
    offset_dec_model.load_networks(epoch, load_dir=eval_name)
    joint_model.initialize(opt_exp, enc_model, dec_model, offset_dec_model, gpu_ids = opt_exp.gpu_ids)
elif opt_exp.n_decoders == 1:
    joint_model.initialize(opt_exp, enc_model, dec_model, gpu_ids = opt_exp.gpu_ids)

# pass data through model
total_loss, median_error = trainer.test(joint_model, 
    test_loader, 
    save_output=True,
    save_dir=eval_name,
    save_name=f"decoder_test_result_epoch_{epoch}",
    log=False)
print(f"total_loss: {total_loss}, median_error: {median_error}")
