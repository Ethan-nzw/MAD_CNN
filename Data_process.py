import torch
import numpy as np
from scipy import io as scio
from os.path import dirname, join as pjoin
from scipy.signal import savgol_filter
from tqdm import tqdm

# Data extraction form .mat file
class Data_process_mat():
    def __init__(self, data_path, data_name, start=0, end=-1):
        self.data_path = data_path # data path
        self.data_name = data_name # data name with .mat
        self.start = start # data length
        self.end = end  # data length

    def data_extract(self):
        data_dir = pjoin(dirname(scio.__file__), self.data_path)
        mat_fname_free = pjoin(data_dir, self.data_name)
        data_read_training_free = scio.loadmat(mat_fname_free)
        data = data_read_training_free['Expri_Data']
        data = data[self.start:self.end, :]
        return data

    def signal_extract(self):
        data = self.data_extract() # data extraction
        ts = 0.001 # sampling time

        tau_1 = data[:, 0] # tau_1
        tau_2 = data[:, 1] # tau_2

        link_1_P = data[:, 2] # link_1_P
        link_2_P = data[:, 3] # link_2_P
        Collision_label = data[:, 8] # Collision_label

        l1_vel = savgol_filter(np.gradient(link_1_P) / ts, 21, 3) # link_1_P velocity
        l2_vel = savgol_filter(np.gradient(link_2_P) / ts, 21, 3) # link_2_P velocity

        l1_ref = data[:, 6] # link_1_P reference
        l2_ref = data[:, 7] # link_2_P reference

        err_l1 = l1_ref - link_1_P # link_1_P error
        err_l2 = l2_ref - link_2_P # link_2_P error

        v_err_l1 = savgol_filter(np.gradient(err_l1) / ts, 21, 3) # link_1_P error velocity
        v_err_l2 = savgol_filter(np.gradient(err_l2) / ts, 21, 3) # link_2_P error velocity

        return tau_1, tau_2, l1_vel, l2_vel, err_l1, err_l2, v_err_l1, v_err_l2, Collision_label

    # The collision index which includes the start and end index of collision
    def collision_index_get(self, label_data):
        diff_label = np.diff(label_data)
        # index_label = np.empty(shape=[2, 0])
        index_label_1 = np.asarray(np.where(diff_label > 0)) + 1
        index_label_2 = np.asarray(np.where(diff_label < 0))
        # index_label = np.asarray([index_label_1, index_label_2])
        index_label = np.concatenate((index_label_1, index_label_2), axis=0)
        return index_label

    def signal_normalization(self, signal):

        # normalize to 0 to 1
        min = np.amin(signal)
        max = np.amax(signal)
        signal_normalized = (signal - min) / (max - min)

        # normalize to -1 to 1
        # signal_normalized = (signal_normalized - 0.5) * 2

        # signal_normalized = (signal - np.mean(signal)) / np.std(signal)
        return signal_normalized

    def data_segmentation_TRO(self, data, data_label, segment_size, slice_size):

        data = self.signal_normalization(data)

        N_samples = data.shape[0]
        N_signals = data.shape[1]
        Training_size = N_samples - segment_size + 1

        slice_seg = segment_size // slice_size

        data_segmented = np.ndarray((Training_size, N_signals, slice_seg))
        Y_label = np.ndarray(shape=(Training_size,))


        # Training_slice_data = np.zeros(shape=(Training_size, slice_size + 1))

        data_interval = np.zeros(shape=(slice_seg + 1))

        for samples in tqdm(range(Training_size), desc='Data Segmentation',unit='samples'):

            label_period = data_label[samples:(samples + segment_size)]
            Y_label[samples] = label_period[-1]

            for n_signal in range(N_signals):
                data_slice = data[samples:(samples + segment_size), n_signal]

               # Calculate the index of the last point in the original data
                last_index = len(data_slice)
                # Calculate the index of the first point to extract
                start_index = last_index % slice_size
                # Extract one point from every interval
                data_interval = data_slice[last_index::-slice_size]

                # Training_slice_data[samples, :] = data_interval
                data_segmented[samples, n_signal, :] = np.flip(data_interval)
        return data_segmented, Y_label


    def get_seg_data_and_label_M(self):
        # Extract the signals and labels
        tau_1_test, tau_2_test, vel_1_test, vel_2_test, err_P1, err_P2, err_vel1, err_vel2, Collision_label = self.signal_extract()
        # Concatenate the signals, here we use the tau_1 and tau_2 as the input
        data = np.concatenate((tau_1_test[:, np.newaxis], vel_1_test[:, np.newaxis], tau_2_test[:, np.newaxis], vel_2_test[:, np.newaxis]), axis=1)
        # data = np.concatenate((tau_1_test[:, np.newaxis], vel_2_test[:, np.newaxis], err_P1[:, np.newaxis], err_P2[:, np.newaxis]), axis=1) # [4 3 0 5]
        # data = np.concatenate((vel_1_test[:, np.newaxis], err_P1[:, np.newaxis], err_vel1[:, np.newaxis], tau_2_test[:, np.newaxis]), axis=1)  # [2 4 6 1]
        # data segment and label segment
        data_segmentation, label_segmentation = self.data_segmentation_TRO(data, Collision_label, segment_size=110, slice_size=10)

        return data_segmentation, label_segmentation


class Data_tensor_TRO():
    def __init__(self, data1, data2, label):
        self.data1 = data1
        self.data2 = data2
        self.label = label

    def __getitem__(self, batch_size):
        data_tensor1 = torch.from_numpy(self.data1).to(torch.float32)
        data_tensor2 = torch.from_numpy(self.data2).to(torch.float32)


        seed = 0
        torch.manual_seed(seed)
        label_tensor = torch.randint(0, 2, (data_tensor1.size(0),))
        label_tensor = torch.zeros_like(label_tensor)
        for i in range(data_tensor1.size(0)):
            if self.label[i] == 1:
                label_tensor[i] = 1
            else:
                label_tensor[i] = 0
        # generate the dataset and dataloader, data_tensor is the data, label_tensor is the groundtruth labels
        dataset = torch.utils.data.TensorDataset(data_tensor1, data_tensor2, label_tensor)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        return data_loader, data_tensor1, data_tensor2, label_tensor





