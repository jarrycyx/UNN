import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import shutil
import os

def train_CUTS(model, mask, optimizer, criterion, train_loader, val_loader, device, save_path, n_epochs, summary_writer=None):
    model.to(device)
    val_losses = []
    min_loss = np.inf
    mask = torch.Tensor(mask).to(device)

    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        i = 0
        for seq, target in train_loader:
            seq = torch.Tensor(seq).to(device)
            target = torch.Tensor(target).squeeze().to(device)
            print(seq.shape, target.shape)

            masks = mask.to(device)
            masks = masks.unsqueeze(0)
            masks = masks.repeat(seq.shape[0], 1, 1)

            output = model(seq, masks).to(device)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                # print(f"Epoch {epoch+1}/{n_epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.6f}")
                summary_writer.add_scalar('train_loss', loss.item(), epoch*len(train_loader)+i)
            i += 1

            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            for seq, target in val_loader:
                seq = torch.Tensor(seq).to(device)
                
                target = torch.Tensor(target).squeeze().to(device)
                masks = mask.to(device)
                masks = masks.unsqueeze(0)
                masks = masks.repeat(seq.shape[0], 1, 1)
                output = model(seq, masks).to(device)
                loss = criterion(output, target)

                val_losses.append(loss.item())
                if loss.item() < min_loss:
                    min_loss = loss.item()
                    # 清空文件夹
                    shutil.rmtree(save_path, ignore_errors=True)
                    torch.save(model.state_dict(), save_path + f'epoch_{epoch+1}_loss_{loss.item()}.pth')

        print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {np.mean(train_losses)}')
        print(f'Epoch {epoch+1}/{n_epochs}, Val Loss: {np.mean(val_losses)}')
        summary_writer.add_scalar('train_loss_epoch', np.mean(train_losses), epoch)
        summary_writer.add_scalar('val_loss', np.mean(val_losses), epoch)
        summary_writer.flush()
    torch.save(model.state_dict(), save_path + f'epoch_{epoch+1}_loss_{loss.item()}.pth')
    
    return model

def train(model, optimizer, criterion, train_loader, val_loader, device, save_path, n_epochs, summary_writer=None):
    model.to(device)
    # check path
    if not os.path.exists(save_path + 'model'):
        os.makedirs(save_path + 'model')  
    val_losses = []
    min_loss = np.inf
    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        i = 0
        for seq, target in train_loader:

            seq = torch.Tensor(seq).to(device)
            target = torch.Tensor(target).squeeze().to(device)

            output = model(seq).to(device)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                # print(f"Epoch {epoch+1}/{n_epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.6f}")
                summary_writer.add_scalar('train_loss', loss.item(), epoch*len(train_loader)+i)
            i += 1

            train_losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            for seq, target in val_loader:
                seq = torch.Tensor(seq)
                target = torch.Tensor(target).squeeze().to(device)

                output = model(seq)
                loss = criterion(output, target)

                val_losses.append(loss.item())
                if loss.item() < min_loss:
                    min_loss = loss.item()
                    shutil.rmtree(save_path + 'model/', ignore_errors=True)
                    if not os.path.exists(save_path + 'model'):
                        os.makedirs(save_path + 'model')  
                    torch.save(model.state_dict(), save_path + f'model/epoch_{epoch+1}_loss_{loss.item()}.pth')

        print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {np.mean(train_losses)}')
        print(f'Epoch {epoch+1}/{n_epochs}, Val Loss: {np.mean(val_losses)}')
        summary_writer.add_scalar('train_loss_epoch', np.mean(train_losses), epoch)
        summary_writer.add_scalar('val_loss', np.mean(val_losses), epoch)
        summary_writer.flush()
    torch.save(model.state_dict(), save_path + f'epoch_{epoch+1}_loss_{loss.item()}.pth')
    return model

def stable_train(model, optimizer, criterion, train_loader, val_loader, device, save_path, n_epochs, summary_writer=None):
    model.to(device)
    if not os.path.exists(save_path + 'model'):
        os.makedirs(save_path + 'model')  
    val_losses = []
    min_loss = np.inf
    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        i = 0
        for seq, target in train_loader:
            seq = torch.Tensor(seq).to(device)
            target = torch.Tensor(target).squeeze().to(device)

            output = model(seq, target) 
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.6f}")
                summary_writer.add_scalar('train_loss', loss.item(), epoch*len(train_loader)+i)
            i += 1

            train_losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            for seq, target in val_loader:
                seq = torch.Tensor(seq)
                target = torch.Tensor(target).squeeze().to(device)

                output = model(seq, target)
                loss = criterion(output, target)

                val_losses.append(loss.item())
                if loss.item() < min_loss:
                    min_loss = loss.item()
                    shutil.rmtree(save_path + 'model/', ignore_errors=True)
                    if not os.path.exists(save_path + 'model'):
                        os.makedirs(save_path + 'model')  
                    torch.save(model.state_dict(), save_path + f'model/epoch_{epoch+1}_loss_{loss.item()}.pth')

        print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {np.mean(train_losses)}')
        print(f'Epoch {epoch+1}/{n_epochs}, Val Loss: {np.mean(val_losses)}')
        summary_writer.add_scalar('train_loss_epoch', np.mean(train_losses), epoch)
        summary_writer.add_scalar('val_loss', np.mean(val_losses), epoch)
        summary_writer.flush()
    torch.save(model.state_dict(), save_path + f'epoch_{epoch+1}_loss_{loss.item()}.pth')
    return model

def train_using_residual(model, optimizer, criterion, train_loader, val_loader, device, save_path, n_epochs, summary_writer=None):
    full_model = model.full_model
    masked_model = model.masked_model
    full_model.to(device)
    masked_model.to(device)

    train(full_model, optimizer, criterion, train_loader, val_loader, device, save_path + 'full/', n_epochs, summary_writer)
    train(masked_model, optimizer, criterion, train_loader, val_loader, device, save_path + 'masked/', n_epochs, summary_writer)
    return model
