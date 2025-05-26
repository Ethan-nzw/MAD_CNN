import os
import torch

class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Define the hyperparameters
    input_dim = 2 # (2 signals:(joint torque, joint velocity))
    num_classes = 2 # The number of classes in the output (collision or no collision)
    learning_rate = 0.001 # The learning rate
    num_epochs = 30 # The number of epochs
    batch_size = 1000 # The batch size
    CF_len = 15 # The continuous filter length, default is 0

    # Data Segmentation
    segments = (0, 2, 4)

    @property
    def signal_path(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(script_dir, "signal")
    
    # File Names
    train_files = {
        'C2': 'Training_collision_2nd_link_highest_stiff_no_load_2mins.mat',
        'C1': 'Training_collision_1st_link_highest_stiff_no_load_5mins.mat'
    }
    
    test_files = {
        'stiff_level': '3rd',
        'base_name': 'Testing_5mins_C{link}_no_load_{stiff_level}_stiff_clear_label.mat',
        'free_name': 'Testing_freemotion_{stiff_level}_stiff_no_load_15mins.mat'
    }

cfg = Config()