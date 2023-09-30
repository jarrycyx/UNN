import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt



class shareTransformer(nn.Module):
    def __init__(self, input_size, hidden_dim, mask, num_layers, share_type, hidden_size, dropout, full_mask=False):
        super(shareTransformer, self).__init__()
        for i in range(mask.shape[0]):
            mask[i][i] = 1
        # Encoder
        self.input_size = input_size
        self.output_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.share_type = share_type
        self.mask_origin = mask
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(input_size, hidden_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_heads = 4
        if full_mask:
            self.mask = np.ones((input_size, input_size))
        else:
            self.mask = mask

        if share_type == 'full':
            
            self.encoder = nn.ModuleList([nn.TransformerEncoder(
                nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_dim),
                num_layers)
                for _ in range(input_size)
            ])

            # Decoder
            self.decoder = nn.ModuleList([nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )
                                          for _ in range(input_size)
            ])
        
        elif share_type == 'encoder':
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_dim),
                num_layers)

            # Decoder
            self.decoder = nn.ModuleList([nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )
                              for _ in range(input_size)
            ])

        elif share_type == 'decoder':
            self.encoder = nn.ModuleList([nn.TransformerEncoder(
                nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_dim),
                num_layers)
                for _ in range(input_size)
            ])

            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )
        self.mask = torch.from_numpy(self.mask).type(torch.DoubleTensor).to(self.device)
    def forward(self, x, y = None):
        mask = self.mask
        if y is not None:
            if np.random.rand() < 0.3:
                x.to(self.device)
                y.to(self.device)
                x = torch.cat((x[:,0:-1,:],y.unsqueeze(1)), dim=1).to(self.device)
        x = x.permute(1, 0, 2) 
        head_outputs = []
        for i in range(self.output_size):
            x = x.to(self.device)
            masked_x = x * mask[i]
            masked_x = masked_x.type(torch.FloatTensor).to(self.device)
            masked_x = self.embedding(masked_x).to(self.device)
            if self.share_type == 'full':
                masked_x = self.encoder[i](masked_x)
                masked_x = masked_x.permute(1, 0, 2)
                head_outputs.append(self.decoder[i](masked_x[:, -1]))
            elif self.share_type == 'encoder':
                masked_x = self.encoder(masked_x)
                masked_x = masked_x.permute(1, 0, 2)
                head_outputs.append(self.decoder[i](masked_x[:, -1]))
            elif self.share_type == 'decoder':
                masked_x = self.encoder[i](masked_x)
                masked_x = masked_x.permute(1, 0, 2)
                head_outputs.append(self.decoder(masked_x[:, -1]))
            else:
                raise ValueError('share_type must be one of full, encoder, decoder')
        outputs = torch.cat(head_outputs, dim=1)
        return outputs

    def set_mask(self, mask):
        self.mask = mask
        self.mask = torch.from_numpy(self.mask).type(torch.DoubleTensor).to(self.device)


class shareMLP(nn.Module):
    def __init__(self, input_size, hidden_dim, mask, num_layers, share_type, hidden_size, dropout, full_mask=False):
        super(shareMLP, self).__init__()
        for i in range(mask.shape[0]):
            mask[i][i] = 1
        # Encoder
        self.input_size = input_size
        self.output_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.share_type = share_type
        self.mask_origin = mask
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding = nn.Linear(input_size, hidden_size)

        if full_mask:
            self.mask = np.ones((input_size, input_size))
        else:
            self.mask = mask

        if share_type == 'full':
            
            self.encoder = nn.ModuleList([
                nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size))
                for _ in range(input_size)
            ])

            # Decoder
            self.decoder = nn.ModuleList([nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )
                                          for _ in range(input_size)
            ])
        
        elif share_type == 'encoder':
            self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size))

            # Decoder
            self.decoder = nn.ModuleList([nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )
                              for _ in range(input_size)
            ])

        elif share_type == 'decoder':
            self.encoder = nn.ModuleList([
                nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size))
                for _ in range(input_size)
            ])

            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )
        self.mask = torch.from_numpy(self.mask).type(torch.DoubleTensor).to(self.device)
    def forward(self, x, y = None):
        mask = self.mask
        if y is not None:
            if np.random.rand() < 0.3:
                x.to(self.device)
                y.to(self.device)
                x = torch.cat((x[:,0:-1,:],y.unsqueeze(1)), dim=1).to(self.device)
        x = x.permute(1, 0, 2) 
        head_outputs = []
        for i in range(self.output_size):
            x = x.to(self.device)
            masked_x = x * mask[i]
            masked_x = masked_x.type(torch.FloatTensor).to(self.device)
            masked_x = self.embedding(masked_x).to(self.device)
            if self.share_type == 'full':
                masked_x = self.encoder[i](masked_x)
                masked_x = masked_x.permute(1, 0, 2)
                head_outputs.append(self.decoder[i](masked_x[:, -1]))
            elif self.share_type == 'encoder':
                masked_x = self.encoder(masked_x)
                masked_x = masked_x.permute(1, 0, 2)
                head_outputs.append(self.decoder[i](masked_x[:, -1]))
            elif self.share_type == 'decoder':
                masked_x = self.encoder[i](masked_x)
                masked_x = masked_x.permute(1, 0, 2)
                head_outputs.append(self.decoder(masked_x[:, -1]))
            else:
                raise ValueError('share_type must be one of full, encoder, decoder')
        outputs = torch.cat(head_outputs, dim=1)
        return outputs


    def set_mask(self, mask):
        self.mask = mask
        self.mask = torch.from_numpy(self.mask).type(torch.DoubleTensor).to(self.device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class shareLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, mask, num_layers, share_type, hidden_size, dropout, full_mask=False):
        super(shareLSTM, self).__init__()
        for i in range(mask.shape[0]):
            mask[i][i] = 1

        self.input_size = input_size
        self.output_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.share_type = share_type
        self.mask_origin = mask
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding = nn.Linear(input_size, hidden_size)

        if full_mask:
            self.mask = np.ones((input_size, input_size))
        else:
            self.mask = mask

        if share_type == 'full':
            
            self.encoder = nn.ModuleList([
                nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=False)
                for _ in range(input_size)
            ])

            self.decoder = nn.ModuleList([nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )
                                          for _ in range(input_size)
            ])
        
        elif share_type == 'encoder':
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=False)

            self.decoder = nn.ModuleList([nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )
                              for _ in range(input_size)
            ])

        elif share_type == 'decoder':
            self.encoder = nn.ModuleList([
                nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=False)
                for _ in range(input_size)
            ])

            self.decoder = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )
        self.mask = torch.from_numpy(self.mask).type(torch.DoubleTensor).to(self.device)

    def forward(self, x, y = None):
        mask = self.mask
        if y is not None:
            if np.random.rand() < 0.3:
                x = x.to(self.device)
                y.to(self.device)
                x = torch.cat((x[:,0:-1,:],y.unsqueeze(1)), dim=1).to(self.device)
        x = x.permute(1, 0, 2)
        head_outputs = []
        for i in range(self.output_size):
            x = x.to(self.device)
            masked_x = x * mask[i]
            masked_x = masked_x.type(torch.FloatTensor).to(self.device)
            masked_x = self.embedding(masked_x).to(self.device)
            if self.share_type == 'full':
                masked_x, _ = self.encoder[i](masked_x)
                masked_x = masked_x.permute(1, 0, 2)
                head_outputs.append(self.decoder[i](masked_x[:, -1]))
            elif self.share_type == 'encoder':
                masked_x, _ = self.encoder(masked_x)
                masked_x = masked_x.permute(1, 0, 2)
                head_outputs.append(self.decoder[i](masked_x[:, -1]))
            elif self.share_type == 'decoder':
                masked_x, _ = self.encoder[i](masked_x)
                masked_x = masked_x.permute(1, 0, 2)
                head_outputs.append(self.decoder(masked_x[:, -1]))
            else:
                raise ValueError('share_type must be one of full, encoder, decoder')
        outputs = torch.cat(head_outputs, dim=1)
        return outputs

    def set_mask(self, mask):
        self.mask = mask
        self.mask = torch.from_numpy(self.mask).type(torch.DoubleTensor).to(self.device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Residual_model(nn.Module):
    def __init__(self, input_size, hidden_dim, mask, num_layers, share_type, hidden_size, dropout, type = 'LSTM'): 
        super(Residual_model, self).__init__()
        self.input_size = input_size
        self.output_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.share_type = share_type
        self.mask_origin = mask
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding = nn.Linear(input_size, hidden_size)
        self.type = type
        self.full_mask = np.ones((input_size, input_size))
        self.mask = mask
        self.expand_mask = self.expand_matrix(mask)
        if type == 'LSTM':
            self.full_model = shareLSTM(input_size, hidden_dim, self.full_mask, num_layers, share_type, hidden_size, dropout, full_mask=True)
            self.masked_model = shareLSTM(input_size, hidden_dim, mask, num_layers, share_type, hidden_size, dropout, full_mask=False)
        elif type == 'MLP':
            self.full_model = shareMLP(input_size, hidden_dim, self.full_mask, num_layers, share_type, hidden_size, dropout, full_mask=True)
            self.masked_model = shareMLP(input_size, hidden_dim, mask, num_layers, share_type, hidden_size, dropout, full_mask=False)
        elif type == 'Transformer':
            self.full_model = shareTransformer(input_size, hidden_dim, self.full_mask, num_layers, share_type, hidden_size, dropout, full_mask=False)
            self.masked_model = shareTransformer(input_size, hidden_dim, mask, num_layers, share_type, hidden_size, dropout, full_mask=False)
        else:
            raise ValueError('type must be one of LSTM, MLP, Transformer')
        self.last_residual = None
    def forward(self, x):
        full_output = self.full_model(x)
        masked_output = self.masked_model(x)
        residual = full_output - masked_output
        if self.last_residual is None:
            self.last_residual = residual
            return torch.zeros_like(torch.cat((masked_output, residual), dim=1))
        else:
            self.last_residual = residual
            outputs = torch.cat((masked_output, self.last_residual), dim=1)
            return outputs
    
    def expand_matrix(self, matrix):
        n = matrix.shape[0]    
        expanded_matrix = np.ones((2*n, n))
        expanded_matrix[:n, :] = matrix
        expandeded_matrix = np.zeros((2*n, 2*n))
        expandeded_matrix[:2*n, :n] = expanded_matrix[:, :]
        for i in range(n):
            expandeded_matrix[n, n+i] = 1
        return expandeded_matrix
    
    def set_mask(self, mask):
        self.mask = mask
        self.mask = torch.from_numpy(self.mask).type(torch.DoubleTensor).to(self.device)
        self.full_model.set_mask(mask)
        self.masked_model.set_mask(mask)

class NormalizingFlows(nn.Module):
    def __init__(self, input_dim, hidden_dims, flow_length):
        super(NormalizingFlows, self).__init__()
        self.transforms = nn.ModuleList()
        for _ in range(flow_length):
            transform = nn.Sequential(
                nn.Linear(input_dim, hidden_dims),
                nn.ReLU(),
                nn.Linear(hidden_dims, input_dim)
            )
            self.transforms.append(transform)

    def forward(self, x):
        log_det_J = 0
        for transform in self.transforms:
            x = x + transform(x)
            log_det_J += torch.sum(torch.log(torch.abs(1 + transform(x))), dim=1)
        return x, log_det_J

class NF_ResidualTransformerModel(nn.Module):
    def __init__(self, base_model, input_size, output_size, hidden_size, mask, num_layers, flow_length):
        super(NF_ResidualTransformerModel, self).__init__()
        
        for i in range(mask.shape[0]):
            mask[i][i] = 1
          
        self.base_model = base_model
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size * 2
        self.mask_origin = mask
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mask = self.expand_matrix(mask)
        self.flow_length = flow_length
        self.mean_noise = torch.zeros((input_size))
        self.val_noise = torch.zeros((input_size))
        self.flows = NormalizingFlows(input_size, hidden_size, flow_length)
        self.visual_mean = torch.zeros((input_size))
    def forward(self, x):
        random_noise = torch.randn_like(self.base_model(x)) * self.val_noise.to(self.device)
        NF_noise, _ = self.flows(random_noise)
        return self.base_model(x.to(self.device)).to(self.device) + NF_noise.to(self.device)
    def expand_matrix(self, matrix):
        n = matrix.shape[0]    
        expanded_matrix = np.ones((2*n, n))
        expanded_matrix[:n, :] = matrix
        expandeded_matrix = np.zeros((2*n, 2*n))
        expandeded_matrix[:2*n, :n] = expanded_matrix[:, :]
        for i in range(n):
            expandeded_matrix[n, n+i] = 1
        return expandeded_matrix
    
    def train_NF(self, dataloader, n_epochs, summary_writer):
        optimizer = optim.Adam(self.flows.parameters(), lr=0.001)

        criterion = nn.MSELoss()
        trained_model = self.base_model
        trained_model.to(self.device)
        self.flows.to(self.device)

        for epoch in range(n_epochs):
            running_loss = 0.0
            trained_model.train()
            self.flows.train()
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                optimizer.zero_grad()
                inputs = torch.Tensor(inputs).to(self.device)
                targets = torch.Tensor(targets).to(self.device)
                transformed_inputs = trained_model(inputs)
                if transformed_inputs.shape != targets.shape:
                    targets = torch.cat((targets, torch.zeros_like(targets)), dim=1)


                outputs, _ = self.flows(transformed_inputs - targets)
                loss = criterion(outputs + transformed_inputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {running_loss/len(dataloader):.4f}")
            summary_writer.add_scalar('NF_noise_train_loss', running_loss/len(dataloader), epoch)

    def visualize_NF(self, num_samples, save_path, summary_writer):
        self.flows.eval()
        self.flows.to(self.device)
        input_dim = self.input_size
        z = torch.randn(num_samples, input_dim).to(self.device)
        generated_samples,_ = self.flows(z) 
        generated_samples = generated_samples.cpu()
        print(generated_samples.shape)
        plt.clf()
        plt.title('Generated residual samples')
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 5))
        for i in range(3):
            for j in range(3):
                axes[i, j].hist(generated_samples[:, i*3+j].detach().numpy(), bins=100, density=True)
                axes[i, j].set_title(f'feature {i*3+j}')
        plt.savefig(save_path + '/residual_samples.png')
        summary_writer.add_figure('residual_samples', fig)
        self.visual_mean = generated_samples.mean(dim=0)
        Mean = generated_samples.mean(dim=0)
        self.mean_noise = Mean
        print(f"Mean: {Mean}")
        Var = generated_samples.var(dim=0)
        self.val_noise = Var
        print(f"Var: {Var}")
        return (Mean, Var)

