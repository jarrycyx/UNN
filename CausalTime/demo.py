import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import load_data_h5py
from train import train, stable_train
from torch.utils.tensorboard import SummaryWriter
import os
from visualization import plot_model_prediction, plot_generate_data
from models import shareLSTM, shareMLP, shareTransformer, Residual_model, NF_ResidualTransformerModel
from generate import generate
import sys
from os.path import join as opj
from os.path import dirname as opd
sys.path.append(opj(opd(__file__), ".."))

if __name__ == '__main__':
    data_path = './datasets/'
    batch_size = 32
    input_size = 36
    hidden_size = 128
    num_layers = 2
    num_heads = 4
    dropout = 0.1
    seq_length = 20
    num_epochs = 10
    learning_rate = 0.0001
    n_epochs = 1
    flow_length = 4
    gen_n = 20
    save_path = './outputs/air_quality/'
    log_dir = "./outputs/air_quality/log/"
    task = 'air_quality'
    data_path = './datasets/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(log_dir):
       os.makedirs(log_dir)
    summary_writer = SummaryWriter(log_dir=log_dir)
    train_loader, test_loader, val_loader, X, data_ori, mask = load_data_h5py(data_path + task + '/', batch_size, 20, data_type='pm2.5',test_size=0.2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = Residual_model(input_size, hidden_size, mask, num_layers, 'decoder', hidden_size, dropout, type = 'LSTM').to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(base_model.parameters(), lr=learning_rate)
    optimizer.zero_grad()

    train(base_model.full_model, optimizer, criterion, train_loader, val_loader, device, save_path+'full/', n_epochs, summary_writer)
    train(base_model.masked_model, optimizer, criterion, train_loader, val_loader, device, save_path+'masked/', n_epochs, summary_writer)
    model = NF_ResidualTransformerModel(base_model, input_size*2, input_size*2, hidden_size, mask, num_layers, flow_length)
    model.train_NF(train_loader, 5, summary_writer)
    torch.save(model.state_dict(), save_path + 'NF.pth')
    generated_data = generate(model, test_loader, radom=False, batch_size=batch_size, gen_length=20, save_path=save_path, device = device, n = 500)
    leq_length = 5
    plot_generate_data(generated_data[:,:,:input_size], data_ori, save_path, input_size, leq_length, summery_writer=summary_writer)