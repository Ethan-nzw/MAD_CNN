import torch
import torch.nn as nn
import torch.optim as optim
from Data_process import *
from evaluation_CD import *
import torch
import torch.nn as nn
import timeit

#%% Concatenated network
class ConvBlock_S(nn.Module):

    def __init__(self, input_dim):
        super(ConvBlock_S, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=3, stride=1, padding=4, dilation=4) # dilation=4, padding=4
        self.batch1 = nn.BatchNorm1d(num_features=16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=8, dilation=8) #dilation=8, padding=8
        self.batch2 = nn.BatchNorm1d(num_features=32)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(in_features=32 * 2, out_features=32)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.pool1(self.act(self.batch1(self.conv1(x))))
        x = self.pool2(self.act(self.batch2(self.conv2(x))))
        x = x.view(-1, 32 * 2)
        x = self.act(self.fc1(x))
        x = x.unsqueeze(0) # Add batch dimension for attention mechanism

        return x

class AttDilatedT_New_S(nn.Module):

    def __init__(self, input_dim, num_classes):
        super(AttDilatedT_New_S, self).__init__()
        self.conv = ConvBlock_S(input_dim)
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=1)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)
        self.act = nn.GELU()

    def forward(self, xin_1, xin_2):
        x1 = self.conv(xin_1)
        x2 = self.conv(xin_2)
        x = torch.cat((x1, x2), dim=2)
        x = self.act(x)
        x, _ = self.attention(x, x, x) # Apply attention mechanism
        x = self.act(x)
        x = x.view(-1, 64)
        x = self.fc2(x)

        return x


#%% train and evaluation
l2_regularization = 0.001
class train_and_evaluation:
    def __init__(self, model, device, train_loader, test_data, test_label, optimizer, criterion, num_epochs):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_data = test_data
        self.test_label = test_label
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs

    def train(self):
        self.model.train()
        all_losses = []  # Create a list to store all the losses
        for epoch in range(self.num_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)

                # Add regularization term to the loss
                for param in self.model.parameters():
                    loss += l2_regularization * torch.norm(param, 2)

                loss.backward()
                self.optimizer.step()

                if batch_idx % 100 == 0:
                    print(f'Batch {batch_idx}, Loss {loss.item():.4f}')
            all_losses.append(loss.item())

            print(f"Epoch {epoch + 1}, Loss {loss.item():.4f}")

        return all_losses



    def train_S(self):
        self.model.train()
        all_losses = []  # Create a list to store all the losses
        for epoch in range(self.num_epochs):
            for batch_idx, (data1, data2, target) in enumerate(self.train_loader):
                data1, data2, target = data1.to(self.device), data2.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data1, data2)
                loss = self.criterion(output, target)

                # Add regularization term to the loss
                for param in self.model.parameters():
                    loss += l2_regularization * torch.norm(param, 2)

                loss.backward()
                self.optimizer.step()

                if batch_idx % 100 == 0:
                    print(f'Batch {batch_idx}, Loss {loss.item():.4f}')
            all_losses.append(loss.item())

            print(f"Epoch {epoch + 1}, Loss {loss.item():.4f}")

        return all_losses


    def test_Collison_data(self, CF_len):
        self.model.eval()
        with torch.no_grad():
            data = self.test_data.to(self.device)

            start = timeit.default_timer()

            output = self.model(data)
            probs = nn.functional.softmax(output, dim=1)
            predicted_output = (probs[:, 1] > 0.5).long()

            stop = timeit.default_timer()
            execution_time = stop - start

            runing_time = execution_time / len(self.test_data)

            predicted_output, target = predicted_output.to('cpu'), self.test_label.to('cpu')
            torch.cuda.empty_cache()
            Performance = Eva_CD(target.numpy(), predicted_output.numpy(), CF_len=CF_len)
            DFn, DD, DD_each = Performance.DD_DFn_FPn()


        return DFn, DD, predicted_output, probs, runing_time



    def test_free_motion_data(self, CF_len):
        self.model.eval()
        with torch.no_grad():
            data = self.test_data.to(self.device)
            output = self.model(data)
            torch.cuda.empty_cache()

            probs = nn.functional.softmax(output, dim=1)
            predicted_output = (probs[:, 1] > 0.5).long()

            predicted_output, target = predicted_output.to('cpu'), self.test_label.to('cpu')

            Performance = Eva_CD(target.numpy(), predicted_output.numpy(), CF_len=CF_len)
            FPn = Performance.FPn_for_free_motion_data()

        return FPn, predicted_output, probs



    def test_Collison_data_TRO(self, test_data1, test_data2, CF_len):
        self.model.eval()
        with torch.no_grad():
            data1 = test_data1.to(self.device)
            data2 = test_data2.to(self.device)
            start = timeit.default_timer()
            output = self.model(data1, data2)



            probs = nn.functional.softmax(output, dim=1)
            predicted_output = (probs[:, 1] > 0.5).long()

            stop = timeit.default_timer()
            execution_time = stop - start

            runing_time = execution_time / len(data1)

            # probs = output.squeeze(1)
            # predicted_output = (probs > 0.5).long()

            # _, predicted_output = torch.max(output.data, 1)

            predicted_output, target = predicted_output.to('cpu'), self.test_label.to('cpu')
            torch.cuda.empty_cache()
            Performance = Eva_CD(target.numpy(), predicted_output.numpy(), CF_len=CF_len)
            DFn, DD, DD_each = Performance.DD_DFn_FPn()


        return DFn, DD, DD_each, predicted_output, probs, runing_time

    def test_free_motion_data_TRO(self, test_data1, test_data2, CF_len):
        self.model.eval()
        with torch.no_grad():
            data1 = test_data1.to(self.device)
            data2 = test_data2.to(self.device)
            output = self.model(data1, data2)
            torch.cuda.empty_cache()

            probs = nn.functional.softmax(output, dim=1)
            predicted_output = (probs[:, 1] > 0.5).long()

            predicted_output, target = predicted_output.to('cpu'), self.test_label.to('cpu')

            Performance = Eva_CD(target.numpy(), predicted_output.numpy(), CF_len=CF_len)
            FPn = Performance.FPn_for_free_motion_data()

        return FPn, predicted_output, probs