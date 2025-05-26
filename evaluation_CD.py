# evaluate the collision detection performance based on
# Detection Delay (DD) in ms, Detection Failure number (DFn) and False Positive number (FPn)

import numpy as np

class Eva_CD():
    def __init__(self, True_label, Predict_label, CF_len):
        self.True_label = True_label
        self.Predict_label = Predict_label
        self.CF_len = CF_len


        index_true = np.array([])
        diff_label = np.diff(self.True_label)
        index_true_0 = np.asarray(np.where(diff_label > 0)) + 1  # collision start
        index_true_1 = np.asarray(np.where(diff_label < 0))  # collision end
        self.index_true = np.concatenate((index_true_0, index_true_1), axis=0)

        diff_label_pre = np.diff(np.double(self.Predict_label))
        index_pre_0 = np.asarray(np.where(diff_label_pre > 0)) + 1  # collision start
        index_pre_1 = np.asarray(np.where(diff_label_pre < 0))  # collision end


        if index_pre_1[:,0] < index_pre_0[:,0]:
            index_pre_0 = np.insert(index_pre_0, 0, 0, axis=1)
        if index_pre_0[:,-1] > index_pre_1[:,-1]:
            index_pre_1 = np.insert(index_pre_1, -1, self.True_label.shape[0]-1, axis=1)

        index_pre = np.concatenate((index_pre_0, index_pre_1), axis=0)

        ## use the continuous filter
        y_pred_copy = self.Predict_label
        for i in range(index_pre.shape[1]):
            if index_pre[1, i] - index_pre[0, i] < self.CF_len:
                y_pred_copy[index_pre[0, i]:index_pre[1, i]+1,] = 0
        self.Predict_label_CF = y_pred_copy

    def DD_DFn_FPn(self):
        # get all the delay duration
        sub_true_pre = self.True_label - self.Predict_label_CF

        sub_M_P = sub_true_pre
        for i in range(sub_M_P.shape[0]):
            if sub_M_P[i] == -1:
                sub_M_P[i] = 0
        # index_sub = np.empty((0, 2))
        diff_sub_true_pre = np.diff(sub_M_P)
        index_sub_0 = np.asarray(np.where(diff_sub_true_pre > 0)) + 1  # collision start
        index_sub_1 = np.asarray(np.where(diff_sub_true_pre < 0))  # collision end
        index_sub = np.concatenate((index_sub_0, index_sub_1), axis=0)

        # create an empty index_delay array
        # index_delay = np.zeros(2,index_true.shape[1])
        index_delay = np.zeros((2, self.index_true.shape[1]))
        # loop through index and index_sub arrays to find matching collision start times
        for i in range(self.index_true.shape[1]):
            for j in range(index_sub.shape[1]):
                if self.index_true[0, i] == index_sub[0, j]:
                    # index_delay.append([index_true[i, 0], index_sub[j, 1] + 1])
                    index_delay[0, i] = self.index_true[0, i]
                    index_delay[1, i] = index_sub[1, j] + 1
                    break


        # calculate the Detection Failure number (DFn)
        duration_collision_pre = index_delay[1, :] - index_delay[0, :]
        duration_true = self.index_true[1, :] - self.index_true[0, :]
        index_delay_copy = index_delay
        num_FN = 0
        for i in range(duration_true.shape[0]):
            if duration_collision_pre[i] == (duration_true[i] + 1):
                num_FN += 1
                index_delay_copy[:, i] = 0

        # calculate Detection Delay (DD)
        delay_each = index_delay_copy[1, :] - index_delay_copy[0, :]
        index_zero = np.where(delay_each == 0)
        delay_each = np.delete(delay_each, index_zero)
        delay_average = np.mean(delay_each)

        ## calculate the False Positive number (FPn) for the collision data
        # sub_true_pre_FP = sub_true_pre
        # for ii in range(sub_true_pre_FP.shape[0]):
        #     if sub_true_pre[ii] == 1:
        #         sub_true_pre_FP[ii] = 0
        #
        # diff_sub_true_pre_FP = np.diff(sub_true_pre_FP)
        # # index_FP = np.zeros((diff_sub_true_pre_FP.shape[1], 2), dtype=int)
        # index_FP_0 = np.asarray(np.where(diff_sub_true_pre_FP < 0)) + 1  # collision start
        # index_FP_1 = np.asarray(np.where(diff_sub_true_pre_FP > 0))  # collision end
        # index_FP = np.concatenate((index_FP_0, index_FP_1), axis=0)
        #
        # for ii in range(index_true.shape[1]):
        #     for jj in range(index_FP.shape[1]):
        #         if index_FP[0, jj] == index_true[1, ii] + 1:
        #             index_FP[:,jj] = 0
        #
        # index_cross = np.where(index_FP[0, :] == 0)
        # num_FP = index_FP.shape[1] - index_cross.shape[1]

        return num_FN, delay_average, delay_each

    def FPn_for_free_motion_data(self):
        # get all the duration for all False Positive prediction
        sub_true_pre = self.Predict_label_CF - self.True_label
        # index_FP = []
        diff_sub_true_pre = np.diff(sub_true_pre)
        index_FP_0 = np.asarray(np.where(diff_sub_true_pre > 0)) + 1  # FP start
        index_FP_1 = np.asarray(np.where(diff_sub_true_pre < 0))  # FP end
        # index_FP_1 = index_FP_1[:, 1:]
        # index_FP = np.concatenate((index_FP_0, index_FP_1), axis=0)

        # get the number of False Positives
        num_FP = index_FP_0.shape[1]

        return num_FP



