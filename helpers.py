import os
import numpy as np
import scipy.stats as st
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from config import cfg
from Data_process import *
from evaluation_CD import *
from network_built import *


def load_data(file_path, file_name, start=None, end=None):
    processor = Data_process_mat(file_path, file_name, start=start, end=end)
    return processor.get_seg_data_and_label_M()

def create_data_loader(data1, data2, labels, batch_size):
    tensor_creator = Data_tensor_TRO(data1, data2, labels)
    return tensor_creator.__getitem__(batch_size=batch_size)

s1, s2, s3 = cfg.segments
def load_test_data(link):
    file_name = cfg.test_files['base_name'].format(
        link=link, stiff_level=cfg.test_files['stiff_level']
    )
    data, labels = load_data(cfg.signal_path, file_name)
    return data[:, s1:s3, :], labels


def initialize_model():
    model = AttDilatedT_New_S(
        input_dim=cfg.input_dim, 
        num_classes=cfg.num_classes
    ).to(cfg.device)
    
    if os.path.exists('MAD_CNN.pth'):
        model.load_state_dict(torch.load('MAD_CNN.pth'))
        model.eval()
    
    return model