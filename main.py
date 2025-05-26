import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.metrics import roc_curve, auc
from torch import nn, optim
from config import cfg
from helpers import *

torch.cuda.empty_cache()

def main():
    # Initialize the model
    model = initialize_model()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Load training and testing data
    train_loader = load_training_data() # Training collision data
    test_C1_tensor, test_C2_tensor, test_C_labels = load_testing_collision_data() # Testing collision data
    test_data_free, test_label_free = load_testing_free_motion_data() # Testing free motion data 

    # Train the model
    trainer = train_and_evaluation(
    model, cfg.device, train_loader, None, None, 
    optimizer, criterion, cfg.num_epochs
    )
    trainer.train_S()

    # Save the trained model
    torch.save(model.state_dict(), 'MAD_CNN.pth') 

    # Evaluation
    evaluate_model(model, test_C1_tensor, test_C2_tensor, test_C_labels, test_data_free, test_label_free, optimizer, criterion)


def load_training_data():
    # Load and process training data
    train_C2_data, train_C2_labels = load_data(cfg.signal_path, cfg.train_files['C2'])
    train_C1_data, train_C1_labels = load_data(
        cfg.signal_path, cfg.train_files['C1'], start=0, end=1000*60*2
    )

    # Combine and segment data
    s1, s2, s3 = cfg.segments
    train_data = {
        'data1': np.concatenate([train_C1_data[:, s1:s2, :], train_C2_data[:, s1:s2, :]]),
        'data2': np.concatenate([train_C1_data[:, s2:s3, :], train_C2_data[:, s2:s3, :]]),
        'labels': np.concatenate([train_C1_labels, train_C2_labels])
    }

    train_loader, _, _, _ = create_data_loader(
        train_data['data1'], train_data['data2'], train_data['labels'], cfg.batch_size
    )
    return train_loader


def load_testing_collision_data():
    # Load collision data for testing
    test_C2_data, test_C2_labels = load_test_data(2)
    test_C1_data, test_C1_labels = load_test_data(1)


    # Process collision data
    test_collision = {
        'data1': np.concatenate([test_C1_data[:, s1:s2, :], test_C2_data[:, s1:s2, :]]),
        'data2': np.concatenate([test_C1_data[:, s2:s3, :], test_C2_data[:, s2:s3, :]]),
        'labels': np.concatenate([test_C1_labels, test_C2_labels])
    }

    _, test_C1_tensor, test_C2_tensor, test_C_labels = create_data_loader(
        test_collision['data1'], test_collision['data2'], test_collision['labels'], cfg.batch_size
    )
    return test_C1_tensor, test_C2_tensor, test_C_labels


def load_testing_free_motion_data():
    # Load free motion data for testing
    test_data_free, test_label_free = load_data(
        cfg.signal_path, 
        cfg.test_files['free_name'].format(stiff_level=cfg.test_files['stiff_level'])
    )
    return test_data_free, test_label_free


def evaluate_model(model, dataC1, dataC2, labelsC, test_data_free, test_label_free, optimizer, criterion):
    # Evaluate the model on collision data and free motion data
    evaluator = train_and_evaluation(
        model, cfg.device, None, None, labelsC, optimizer, criterion, cfg.num_epochs
    )
    DFn, DD, DD_each, pred_out, prob, runtime = evaluator.test_Collison_data_TRO(dataC1, dataC2, CF_len=cfg.CF_len)

    # Free motion evaluation
    FPn_total = 0

    # Process in intervals
    free_intervals = [(0,5), (5,10), (10,15)]
    free_results = [process_free_motion(interval, test_data_free, test_label_free) for interval in free_intervals]

    for (_, d1, d2, lbl), interval in zip(free_results, free_intervals):
        FPn, pred, prob = train_and_evaluation(
            model, cfg.device, _, _, lbl, optimizer, criterion, cfg.num_epochs
        ).test_free_motion_data_TRO(d1, d2, CF_len=cfg.CF_len)
        
        FPn_total += FPn

    #print the results
    print(f"DFn: {DFn} \nDD: {DD}")
    print(f"FPn: {FPn_total}")
    print(f"Runing time: {runtime}")


def process_free_motion(interval, test_data_free, test_label_free):
    # Process free motion data in specified intervals
    start, end = interval
    data = test_data_free[1000*60*start:1000*60*end]
    s1, s2, s3 = cfg.segments
    return create_data_loader(
        data[:, s1:s2, :], data[:, s2:s3, :], 
        test_label_free[1000*60*start:1000*60*end], cfg.batch_size
    )


if __name__ == "__main__":
    main()