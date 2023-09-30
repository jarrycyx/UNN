import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from mmd import MMD_loss
from xcorr import xcorr_score

def test_model_MSE(test_loader, model, summary_writer, device):
    model.eval()
    test_losses = []
    with torch.no_grad():
        for seq, target in test_loader:
            seq = torch.Tensor(seq)
            target = torch.Tensor(target).squeeze().to(device)
            output = model(seq)
            mse = np.mean((np.array(output.cpu()) - np.array(target.cpu())) ** 2)
            test_losses.append(mse)
    print('Test MSE: ', np.mean(test_losses))
    summary_writer.add_scalar('Test model MSE', np.mean(test_losses))
    return np.mean(test_losses)

class Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout):
        super(Classifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.test_data = None

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out
    
    def train_classifier(self, real_data, generate_data, seq_len, device, summary_writer, num_epochs=5, learning_rate=0.001):
        if len(real_data.shape) == 2:
            real_data = torch.Tensor(real_data)
            real_data = real_data.unfold(0, seq_len, 1)
        elif len(real_data.shape) == 3:
            real_data_list = []
            for i in range(real_data.shape[0]):
                real_data_list.append(torch.Tensor(real_data[i]).unfold(0, seq_len, 1))
            real_data = torch.cat(real_data_list)
        if len(generate_data.shape) == 3:
            generate_data_list = []
            for i in range(generate_data.shape[0]):
                generate_data_list.append(torch.Tensor(generate_data[i]).unfold(0, seq_len, 1))
            generate_data = torch.cat(generate_data_list)
        elif len(generate_data.shape) == 2:
            generate_data = torch.Tensor(generate_data).unfold(0, seq_len, 1)

        real_label = torch.ones(real_data.shape[0])
        generate_label = torch.zeros(generate_data.shape[0])
        real_set = torch.utils.data.TensorDataset(real_data, real_label)
        generate_set = torch.utils.data.TensorDataset(generate_data, generate_label)
        
        train_size = int(0.8 * len(real_set))
        test_size = len(real_set) - train_size
        real_train, real_test = torch.utils.data.random_split(real_set, [train_size, test_size])
        train_size = int(0.8 * len(generate_set))
        test_size = len(generate_set) - train_size
        generate_train, generate_test = torch.utils.data.random_split(generate_set, [train_size, test_size])
        
        train_dataset = torch.utils.data.ConcatDataset([real_train, generate_train])
        test_dataset = torch.utils.data.ConcatDataset([real_test, generate_test])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
        self.test_data = real_test
        self.seq_len = seq_len
        
        self = self.to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            self.train()
            for i, (X, y) in enumerate(train_loader):
                X = X.permute(0, 2, 1).to(device)
                X = X.cuda()
                y = y.cuda()
                y_pred = self(X)
                loss = criterion(y_pred.squeeze(), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i % 10 == 0:
                    # print(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.6f}")
                    summary_writer.add_scalar('train_loss', loss.item(), epoch*len(train_loader)+i)
            self.eval()
            with torch.no_grad():
                for i, (X, y) in enumerate(test_loader):
                    X = X.permute(0, 2, 1).to(device)
                    X = X.cuda()
                    y = y.cuda()
                    y_pred = self(X)
                    loss = criterion(y_pred.squeeze(), y)
                    summary_writer.add_scalar('test_loss', loss.item(), epoch*len(test_loader)+i)
        torch.save(self.state_dict(), '/ssd/0/wzq/unnset/unnset/classifier.pth')

    def test_by_classify(self, generate_data, summary_writer, device):
        if len(generate_data.shape) == 2:
            generate_data = torch.Tensor(generate_data)
            generate_data = generate_data.unfold(0, self.seq_len, 1)
            generate_label = torch.zeros(generate_data.shape[0])
        elif len(generate_data.shape) == 3:
            generate_data_list = []
            for i in range(generate_data.shape[0]):
                generate_data_list.append(torch.Tensor(generate_data[i]).unfold(0, self.seq_len, 1))
            generate_data = torch.cat(generate_data_list)
            generate_label = torch.zeros(generate_data.shape[0])
        dataset = torch.utils.data.TensorDataset(generate_data, generate_label)
        test_size = len(self.test_data)
        dataset, _ = torch.utils.data.random_split(dataset, [test_size, len(dataset) - test_size])
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        real_data_loader = torch.utils.data.DataLoader(self.test_data, batch_size=32, shuffle=True)
        self = self.to(device)
        self.eval()
        acc = []
        y_pred_list = []
        y_list = []
        with torch.no_grad():
            for i, (X, y) in enumerate(test_loader):
                X = X.permute(0, 2, 1).to(device)
                X = X.cuda()
                y = y.cuda()
                y_pred = self(X)
                y_pred = y_pred.cpu().detach().numpy()
                y = y.cpu().detach().numpy()
                y_pred_list.append(y_pred.squeeze())
                y_list.append(y.squeeze())
                y_pred = np.where(y_pred > 0.3, 1, 0)
                accuracy = np.mean(y_pred == y)
                acc.append(accuracy)
                
            for i, (X, y) in enumerate(real_data_loader):
                X = X.permute(0, 2, 1).to(device)
                X = X.cuda()
                y = y.cuda()
                y_pred = self(X)
                y_pred = y_pred.cpu().detach().numpy()
                y = y.cpu().detach().numpy()
                y_pred_list.append(y_pred.squeeze())
                y_list.append(y.squeeze())
                y_pred = np.where(y_pred > 0.3, 1, 0)
                accuracy = np.mean(y_pred == y)
                acc.append(accuracy)
        accuracy = np.mean(acc)
        y_pred_list = np.concatenate(y_pred_list)
        y_list = np.concatenate(y_list)
        auc_score = roc_auc_score(y_list, y_pred_list)
        print('Test accuracy: ', accuracy)
        print('Test auc: ', auc_score)
        summary_writer.add_scalar('Test accuracy', accuracy)
        summary_writer.add_scalar('Test auc', auc_score)
        return auc_score
    
def generate_score(generate_data, data_ori, summary_writer):
    mmd_fn = MMD_loss()
    if len(generate_data.shape) == 3:
        generate_data = generate_data.reshape(-1, generate_data.shape[-1])
    if len(data_ori.shape) == 3:
        data_ori = data_ori.reshape(-1, data_ori.shape[-1])
    len_max = min(len(generate_data), int(len(data_ori)/2), 1000)
    generate_data = generate_data[:len_max]
    data_ori_first = data_ori[:len_max]
    data_ori_compare = data_ori[len_max:2*len_max]
    MMD = mmd_fn(torch.Tensor(data_ori_first).cuda(), torch.Tensor(generate_data).cuda()).item()
    MMD_self = mmd_fn(torch.Tensor(data_ori_first).cuda(), torch.Tensor(data_ori_compare).cuda()).item()
    xcorr = xcorr_score(data_ori_first, generate_data)
    summary_writer.add_scalar('xcorr', xcorr)
    summary_writer.add_scalar('MMD', MMD)
    summary_writer.add_scalar('MMD_self', MMD_self)
    return MMD, xcorr

def calculate_average_mse(model, testloader, n, device):
    mse_loss = nn.MSELoss()  

    model_seq = None
    i = 0
    mse_list = []
    target_list = []
    outputs = []
    with torch.no_grad():
        for inputs, targets in testloader:

            if i==0:
                model_seq = inputs.to(device)
            output = model(model_seq)
            outputs.append(output.unsqueeze(1))  
            model_seq = torch.cat((model_seq, output.unsqueeze(1)), dim=1) 
            target_list.append(targets.unsqueeze(1))
            i+=1
            if i == n:
                mse = mse_loss(torch.cat(outputs, dim=1), torch.cat(target_list,dim=1).to(device))
                mse_list.append(mse)
                i = 0
                target_list = []
                outputs = []
    average_mse = torch.mean(torch.Tensor(mse_list))
    return average_mse
