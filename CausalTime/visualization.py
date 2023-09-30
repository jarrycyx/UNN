import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

def plot_model_prediction(model, train_loader, len,  save, name = 'test', path = None, mask = None, summery_writer = None):
    model.eval()
    targets = []
    preds = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    n_nodes = model.input_size
    batch_size = train_loader.batch_size
    n = 0
    if mask is not None:
        mask = torch.Tensor(mask).to(device)
    with torch.no_grad():
        for seq, target in train_loader:
            n += 1
            seq = torch.Tensor(seq)
            target = torch.Tensor(target).squeeze()
            seq = seq.to(device)
            if mask is not None:
                masks = mask.unsqueeze(0)
                masks = masks.repeat(seq.shape[0], 1, 1)
                pred = model(seq,masks).squeeze(1)
            else:
                pred = model(seq)
            targets.append(target)
            print(target.shape, pred.shape)
            preds.append(pred)
            if n*batch_size > len:
                break
    targets = torch.cat(targets, dim=0).cpu().detach().numpy()
    preds = torch.cat(preds, dim=0).cpu().detach().numpy()


    plt.figure(figsize=(20, 10))
    plt.plot(targets[0], label='true')
    plt.plot(preds[0], label='pred')
    plt.legend()
    plt.show()
    # check path
    summery_writer.add_figure(name + '_prediction_single', plt.gcf())

    if save:
        if path is None:
            path = os.getcwd()
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path +'/' + name + '_prediction_single.png')
    plt.clf()
    plt.figure(figsize=(20, 10))
    targets = targets.T
    preds = preds.T

    plt.plot(targets[0], label='true')
    plt.plot(preds[0], label='pred')
    plt.legend()
    summery_writer.add_figure(name + '_prediction_batch', plt.gcf())
    if save:
        if path is None:
            path = os.getcwd()
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path +'/' + name + '_prediction_batch.png')
    
    return targets, preds

def plot_generate_data(generated_data, ori_data, save_path, input_size, seq_length, summery_writer = None):
    if len(generated_data.shape) == 3:
        generated_data_seq = []
        for i in range(generated_data.shape[0]):
            els = len(generated_data[i]) % seq_length
            if els != 0:
                generated_data_single = generated_data[i][:-els]
            else:
                generated_data_single = generated_data[i]
            generated_data_seq.append(generated_data_single.reshape(-1, seq_length * input_size))
        generated_data = torch.cat(generated_data_seq, dim=0)
        els = len(ori_data) % seq_length
        if els != 0:
            ori_data = ori_data[:-els]
        ori_data = ori_data.reshape(-1, seq_length * input_size)

    else:
        els = len(ori_data) % seq_length
        if els != 0:
            ori_data = ori_data[:-els]
        ori_data = ori_data.reshape(-1, seq_length * input_size)
        els = len(generated_data) % seq_length
        if els != 0:
            generated_data = generated_data[:-els]
        generated_data = generated_data.reshape(-1, seq_length * input_size)

    if generated_data.shape[0] > ori_data.shape[0]:
        generated_data = generated_data[:ori_data.shape[0]]

    else:
        ori_data = ori_data[:generated_data.shape[0]]

    generated_data = generated_data.cpu().detach().numpy()

    tsne = TSNE(n_components=2, random_state=42)
    pca = PCA(n_components=2)

    marge_data = np.concatenate((generated_data, ori_data), axis=0)
    labels = np.concatenate((np.ones(len(generated_data)), np.zeros(len(ori_data))), axis=0)
    marge_data_tsne = tsne.fit_transform(marge_data)
    marge_data_PCA = pca.fit_transform(marge_data)

    plt.clf()
    plt.scatter(marge_data_tsne[labels == 0, 0], marge_data_tsne[labels == 0, 1], color='#ff2424', label='generated data', alpha=0.5, s=5)
    plt.scatter(marge_data_tsne[labels == 1, 0], marge_data_tsne[labels == 1, 1], color='#0a0aff', label='original data', alpha=0.5, s=5)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('t-SNE Visualization')
    plt.show()
    plt.savefig(save_path + '/t-SNE Visualization.png')
    if summery_writer is not None:
        summery_writer.add_figure('t-SNE Visualization', plt.gcf())


    plt.clf()
    plt.scatter(marge_data_PCA[labels == 0, 0], marge_data_PCA[labels == 0, 1], color='#ff2424', label='generated data', alpha=0.5, s=5)
    plt.scatter(marge_data_PCA[labels == 1, 0], marge_data_PCA[labels == 1, 1], color='#0a0aff', label='original data', alpha=0.5, s=5)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('PCA Visualization')
    plt.show()
    plt.savefig(save_path + '/PCA Visualization.png')
    if summery_writer is not None:
        summery_writer.add_figure('PCA Visualization', plt.gcf())


    
    
    


