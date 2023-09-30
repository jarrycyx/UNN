import h5py
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.manifold import MDS
from geopy import distance

def load_dataset(data_ori, batch_size, seq_len, test_size=0.2, scaler_type='minmax'):
    
    original_shape = data_ori.shape
    
    mask = np.isnan(data_ori)
    data = np.ma.masked_array(data_ori, mask)
    data_interp = pd.DataFrame(data).interpolate().values
    data_ori = data_interp
    data_ori = np.nan_to_num(data_ori)

    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    data_ori = scaler.fit_transform(data_ori.reshape(-1, 1)).squeeze()
    data_ori = data_ori.reshape(original_shape)

    data_slices = []
    for i in range(0, len(data_ori) - seq_len, 1):
        data_slices.append(data_ori[i:i+seq_len])

    tensor_data = torch.from_numpy(np.array(data_slices)).float()

    X = tensor_data[:-1]

    y = []
    for i in range(1, len(X)):
        y.append(X[i][0])
    y.append(torch.from_numpy(data_ori[len(X)]).float())
    y = torch.stack(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.7, shuffle=False)
    val_data = TensorDataset(X_val, y_val)
    test_data = TensorDataset(X_test, y_test)
    return train_data, test_data, val_data, X, data_ori

def load_data(data_ori, batch_size, seq_len, test_size=0.2, scaler_type='minmax'):
    
    original_shape = data_ori.shape
    
    mask = np.isnan(data_ori)
    data = np.ma.masked_array(data_ori, mask)
    data_interp = pd.DataFrame(data).interpolate().values
    data_ori = data_interp
    data_ori = np.nan_to_num(data_ori)

    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    data_ori = scaler.fit_transform(data_ori.reshape(-1, 1)).squeeze()
    data_ori = data_ori.reshape(original_shape)

    data_slices = []
    for i in range(0, len(data_ori) - seq_len, 1):
        data_slices.append(data_ori[i:i+seq_len])

    tensor_data = torch.from_numpy(np.array(data_slices)).float()

    X = tensor_data[:-1]

    y = []
    for i in range(1, len(X)):
        y.append(X[i][0])
    y.append(torch.from_numpy(data_ori[len(X)]).float())
    y = torch.stack(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.7, shuffle=True)
    val_data = TensorDataset(X_val, y_val)
    test_data = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader, X, data_ori


def load_graph(path,threshold = 0.25, save_graph = False, save_map = False):
    """
    Load the graph data
    :param path: the path of the graph data
    :return: the graph data
    """
    with h5py.File(path, "r") as f:
        locations = f["stations"]["block0_values"][:]
    dist_matrix = np.zeros((len(locations), len(locations))) 
    for i in range(len(locations)):
        for j in range(i+1, len(locations)):
            dist = distance.distance(locations[i], locations[j]).km  
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist

    def greater_than_thresh(arr: np.ndarray, thresh: float) -> np.ndarray:
        output = np.zeros_like(arr)
        output[arr > thresh] = 1
        return output

    def distance_conversion(dist):
        min_dist = np.min(dist)
        max_dist = np.max(dist)

        normalized_dist = (dist - min_dist) / (max_dist - min_dist)

        for i in range(len(normalized_dist)):
            normalized_dist[i][i] = 1
        for i in range(len(normalized_dist)):
            for j in range(len(normalized_dist)):
                if normalized_dist[i][j] != 0:
                    normalized_dist[i][j] = 1/normalized_dist[i][j]
                else:
                    normalized_dist[i][j] = 1

        return greater_than_thresh(normalize_distance(np.power(normalized_dist, 1/3)), threshold)

    def normalize_distance(dist):
        min_dist = np.min(dist)
        max_dist = np.max(dist)
        normalized_dist = (dist - min_dist) / (max_dist - min_dist)
        for i in range(len(normalized_dist)):
            normalized_dist[i][i] = 1
        return normalized_dist
    mask = distance_conversion(dist_matrix)
    filename = os.path.basename(path)
    file_without_extension, extension = os.path.splitext(filename)
    if save_graph:
        folder_path ="{}/{}".format(r"./output", file_without_extension)
        #folder_path = 'D:\study\progect\UNN\dataset\code\output\\' + file_without_extension
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.imshow(mask, cmap='Blues')
        plt.colorbar()
        #plt.show()
        plt.savefig(folder_path + '/graph.png')
    if save_map:
        mds = MDS(n_components=2, random_state=42)
        mds_coords = mds.fit_transform(dist_matrix)

        plt.figure(figsize=(8, 6))
        plt.scatter(mds_coords[:, 0], mds_coords[:, 1])
        for i in range(dist_matrix.shape[0]):
            plt.annotate(str(i), (mds_coords[i, 0], mds_coords[i, 1]))
        plt.xlabel("MDS 1")
        plt.ylabel("MDS 2")
        plt.title("MDS of city distances")
        #plt.show()
        plt.savefig(folder_path + '/map.png')
    return mask

def load_medical_data(data, batch_size, seq_len, test_size=0.2):
    
    data = np.array(data)[:200]
    data_oris = []
    train_sets = []
    test_sets = []
    val_sets = []
    Xs = []
    for i in range(len(data)):
        data_single = np.array(data[i])
        train_loader, test_loader, val_loader, X, data_ori = load_dataset(data_single, batch_size, seq_len, test_size=0.2)
        data_oris.append(data_ori)
        train_sets.append(train_loader)
        test_sets.append(test_loader)
        val_sets.append(val_loader)
        Xs.append(X)
    train_set_all = torch.utils.data.ConcatDataset(train_sets)
    test_set_all = torch.utils.data.ConcatDataset(test_sets)
    val_set_all = torch.utils.data.ConcatDataset(val_sets)
    train_loader_all = DataLoader(train_set_all, batch_size=batch_size, shuffle=True)
    test_loader_all = DataLoader(test_set_all, batch_size=batch_size, shuffle=True)
    val_loader_all = DataLoader(val_set_all, batch_size=batch_size, shuffle=False)

    X_all = torch.cat(Xs)
    data_ori_all = np.concatenate(data_oris)
    return train_loader_all, test_loader_all, val_loader_all, X_all, data_ori_all



def load_data_h5py(data_path, batch_size, seq_len, data_type='pm2.5', test_size=0.2, scaler_type='minmax', threshold = 0.25, n_max = 100):
    if data_type == 'pm2.5':
        data_np = np.load(data_path + 'data.npy')
        train_loader, test_loader, val_loader, X, data_np = load_data(data_np, batch_size, seq_len, test_size=0.2)
        mask = np.load(data_path + 'graph.npy') 
    elif data_type == 'traffic':
        data_np = np.load(data_path + 'data.npy')
        if n_max < data_np.shape[1]:
            data_np = data_np[:,:n_max]
        train_loader, test_loader, val_loader, X, data_np = load_data(data_np, batch_size, seq_len, test_size=0.2)
        mask = np.load(data_path + 'graph.npy')
        if n_max < mask.shape[0]:
            mask = mask[:n_max, :n_max]
    elif data_type == 'finance':
        data_np = np.load(data_path + 'data.npy')
        data_np = data_np[:,:n_max]
        train_loader, test_loader, val_loader, X, data_np = load_data(data_np, batch_size, seq_len, test_size=0.2)
        mask = np.load(data_path + 'graph.npy', allow_pickle=True)
    elif data_type == 'medical':
        data_np = np.load(data_path + 'data.npy')
        mask = np.load(data_path + 'graph.npy', allow_pickle=True)
        train_loader, test_loader, val_loader, X, data_np = load_medical_data(data_np, batch_size, seq_len, test_size=0.2)
    else:
        print('data type error!')
    for x, y in train_loader:
        print(f'In loader:X_shape:{x.shape},y_shape:{y.shape}') 
        break
    return train_loader, test_loader, val_loader, X, data_np, mask


