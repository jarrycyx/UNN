import torch
import numpy as np
import os
import random

def generate(model, test_loader, radom, batch_size, gen_length, save_path, device, variance=0.001,radom_seed = 42, n = 100, residual=False):

    all_batches = list(test_loader)

    def generate_data(model, gen_length, seed_ori, device):
        model.eval()
        model = model.to(device)
        data = []
        seed = seed_ori.to(device)
        for i in range(gen_length):
            seed = seed.to(device)
            output = model(seed).detach()
            
            if output.shape[1] != seed.shape[2]:
                seed = torch.cat((seed[:, 1:, :], output.unsqueeze(1)[:,:,:seed.shape[2]]), dim=1)
                if i == 0:
                    data.append(torch.cat((seed, torch.zeros_like(seed).to(device)), dim=2))
            else:
                seed = torch.cat((seed[:, 1:, :], output.unsqueeze(1)), dim=1)
            data.append(output.unsqueeze(1))
        return torch.cat(data, dim=1)
    
    def generate_data_radom(model, gen_length, seed_ori, device, variance=0.1):
        model.eval()
        model = model.to(device)
        data = []
        seed = seed_ori.to(device)
        def generate_random_data(size, mean, variance):
            random_data = torch.normal(mean, variance, size)
            return random_data

        size = seed_ori.shape
        mean = 0 
        for i in range(gen_length):
            seed = seed.to(device)
            output = model(seed + generate_random_data(size, mean, variance).to(device)).detach()
            if output.shape[1] != seed.shape[2]:
                seed = torch.cat((seed[:, 1:, :], output.unsqueeze(1)[:,:,:seed.shape[2]]), dim=1)
                if i == 0:
                    data.append(torch.cat((seed, torch.zeros_like(seed).to(device)), dim=2))
                else:
                    data.append(output.unsqueeze(1))
            else:
                seed = torch.cat((seed[:, 1:, :], output.unsqueeze(1)), dim=1)
                data.append(output.unsqueeze(1))
        return torch.cat(data, dim=1)
    
    generated_datas = []
    max_random = min(n//batch_size + 1, len(all_batches))
    seed_list = random.sample(all_batches, max_random - 1)
    
    if radom:
        for i in range(max_random - 1):
            seed = seed_list[i][0]
            generated_data = generate_data_radom(model, gen_length, seed, device, variance)
            generated_datas.append(generated_data)
    else:
        for i in range(max_random - 1):
            seed = seed_list[i][0]
            generated_data = generate_data(model, gen_length, seed, device)
            generated_datas.append(generated_data)            
    generated_datas = torch.cat(generated_datas, dim=0)
    linked_datas = generated_datas.reshape(-1, generated_datas.shape[2])

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.save(save_path + '/generated_datas.npy', generated_datas.cpu().detach().numpy())
    np.save(save_path + '/linked_datas.npy', linked_datas.cpu().detach().numpy())
    return generated_datas

