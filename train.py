import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
from sklearn.manifold import TSNE

from func import save_model, make_z
from model import VAE

torch.manual_seed(999)

class Trainer:
    def __init__(self, config, device, mode, continuous) -> None:
        self.device = device
        self.continuous = continuous
        self.config = config
        self.mode = mode
        self.denoising = self.config.denoising
        self.log_interval = self.config.log_interval

        self.epochs = self.config.epochs
        self.offset = 0
        self.lr = self.config.lr
        self.data_path = self.config.data_path
        self.val_propotion = self.config.val_propotion
        self.batch_size = self.config.batch_size
        self.model_name = self.config.model_name
        self.model_path = os.path.join('./model', self.model_name, self.model_name + '.pt')

        if self.denoising:
            self.noise_std = self.config.noise_std
            self.noise_mean = self.config.noise_mean
        
        # model

        self.model = VAE(self.config)
        self.model.to(self.device)

        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # data (MNIST)
        self.dataloaders = {}
        if mode == 'train':

            train_data = datasets.MNIST(root=self.data_path,
                                download=True, train=True,
                                transform=transforms.ToTensor())
            data_size = len(train_data)

            self.train_data, self.valid_data = random_split(train_data, [data_size - int(data_size*self.val_propotion), int(data_size * self.val_propotion)])
    
            self.dataloaders['train'] = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
            self.dataloaders['valid'] = DataLoader(self.valid_data, batch_size=self.batch_size, shuffle=False)

            if self.continuous:
                self.check_point = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(self.check_point['model_state'])
                self.optimizer.load_state_dict(self.check_point['optimmizer_state'])
                self.offset = self.check_point['epoch']

        if mode == 'test':
            self.test_data = datasets.MNIST(root=self.data_path,
                                train=False,
                                download=True,
                                transform=transforms.ToTensor())

            self.dataloaders['test'] = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

            self.check_point = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(self.check_point['model_state'])

        # loss
        self.bce_loss = nn.BCELoss(reduction='sum')
        self.criterion = self.vae_loss_fn
        
        
    def vae_loss_fn(self, y, x, mean, log_var):
        
        # regularization term
        loss1 = 0.5 * torch.sum((torch.square(mean) + torch.exp(log_var) - log_var -1))

        # reconstruction term
        loss2 = self.bce_loss(y, x)

        return loss1 + loss2


    def train(self):

        start_time = time.time()

        best_val_loss = float('inf')
        best_val_epoch = self.offset if self.continuous else 0

        train_losses = []
        valid_losses = []

        for epoch in range(self.offset, self.offset + self.epochs):
            print(f"epoch : {epoch+1}")
            self.model.train()
            total_loss = 0.
            for i, batch in tqdm(enumerate(self.dataloaders['train']), desc='training', total=len(self.dataloaders['train'])):
                x, _ = batch[0], batch[1]
                self.optimizer.zero_grad()

                if self.denoising:
                    noise = torch.randn(x.size())*self.noise_std + self.noise_mean
                    noise = noise.to(self.device)
                    x = x.to(self.device)
                    noise_x = x + noise
                    logits, mean, log_var = self.model(noise_x)
                
                else:
                    x = x.to(self.device)
                    logits, mean, log_var = self.model(x)

                loss = self.criterion(logits, x, mean, log_var)
                loss.backward()
                self.optimizer.step()

                total_loss += (loss.item() * x.size(0))

                if i % self.log_interval == 0:
                    print("Train epoch: {} [{}/{}({:.0f}%)]\t Step loss {:.6f}\n".format(epoch+1, i, len(self.dataloaders['train']), 100*i/len(self.dataloaders['train']), loss.item()))

            
            valid_loss = self.valid(epoch)
            print("====> Epoch: {}\t train loss: {:.4f}\t valid loss: {:.4f}\t elapsed time: {} s".format(epoch+1, total_loss/len(self.dataloaders['train'].dataset), valid_loss, time.time()-start_time))

            train_losses.append(total_loss)
            valid_losses.append(valid_loss)

            if best_val_loss > valid_loss:
                print('save model...\n')
                best_val_epoch = epoch+1
                best_val_loss = valid_loss
                save_model(self.model_path, best_val_epoch, best_val_loss, self.model.state_dict(), self.optimizer.state_dict())
        
        print(f"Best Epoch: {best_val_epoch}\nBest Val Loss: {best_val_loss:.4f}")

        return {
            'train_losses' : train_losses,
            'valid_losses' : valid_losses
        }


    def valid(self, epoch):
        self.model.eval()
        total_loss = 0.
        
        for i, batch in tqdm(enumerate(self.dataloaders['valid']), desc='validating', total=len(self.dataloaders['valid'])):
            x, _ = batch[0], batch[1]
            self.optimizer.zero_grad()

            if self.denoising:
                noise = torch.randn(x.size())*self.noise_std + self.noise_mean
                noise = noise.to(self.device)
                x = x.to(self.device)
                noise_x = x + noise
                logits, mean, log_var = self.model(noise_x)

            else:
                x = x.to(self.device)
                logits, mean, log_var = self.model(x)

            loss = self.criterion(logits, x, mean, log_var)

            total_loss += (loss.item() * x.size(0))

            if i % self.log_interval == 0:
                print("Valid epoch: {}\n [{}/{}({:.0f}%)]\t Step loss {:.6f}".format(epoch+1, i, len(self.dataloaders['valid']), 100*i/len(self.dataloaders['valid']), loss.item()))

        return total_loss/len(self.dataloaders['valid'].dataset)


    def test(self, img_num=9):
        self.model.eval()
        total_loss = 0.
        
        X, noise_X, outputs, total_mean, total_log_var, total_label = [], [], [], [], [], []
        for batch in tqdm(self.dataloaders['test'], desc='testing', total=len(self.dataloaders['test'])):
            x, label = batch[0], batch[1].to(self.device)
            self.optimizer.zero_grad()

            if self.denoising:
                noise = torch.randn(x.size())*self.noise_std + self.noise_mean
                noise = noise.to(self.device)
                x = x.to(self.device)
                noise_x = x + noise
                logits, mean, log_var = self.model(noise_x)
                X.append(x.detach().cpu())
                noise_X.append(noise_x.detach().cpu())
                
            else:
                x = x.to(self.device)
                logits, mean, log_var = self.model(x)
                X.append(x.detach().cpu())
            
            outputs.append(logits.detach().cpu())
            total_mean.append(mean.detach().cpu())
            total_log_var.append(log_var.detach().cpu())
            total_label.append(label.detach().cpu())
            
            loss = self.criterion(logits, x, mean, log_var)
            total_loss += (loss.item() * x.size(0))

        print('Test loss: {:.4f}'.format(total_loss/len(self.dataloaders['test'].dataset)))

        X = torch.cat(tuple(X), dim=0)
        outputs = torch.cat(tuple(outputs), dim=0)
        total_mean = torch.cat(tuple(total_mean), dim=0)
        total_log_var = torch.cat(tuple(total_log_var), dim=0)
        total_label = torch.cat(tuple(total_label), dim=0)


        plt.figure(figsize=(6, 3*img_num))
        if self.denoising:
            noise_X = torch.cat(tuple(noise_X), dim=0)
            plt.figure(figsize=(9, 3*img_num))

        # save img
        for j in range(img_num):
            idx = np.random.randint(len(outputs))

            if self.denoising:
                # origin, noise, output
                origin = X[idx].squeeze().numpy()
                noise = noise_X[idx].squeeze().numpy()
                output = outputs[idx].squeeze().numpy()
                
                plt.subplot(img_num, 3, (j*3)+1)
                plt.imshow(origin, cmap='gray')
                plt.subplot(img_num, 3, (j*3)+2)
                plt.imshow(noise, cmap='gray')
                plt.subplot(img_num, 3, (j*3)+3)
                plt.imshow(output, cmap='gray')
                
            else:
                # origin, output
                origin = X[idx].squeeze()
                output = outputs[idx].squeeze()

                plt.subplot(img_num, 2, (j*2)+1)
                plt.imshow(origin, cmap='gray')
                plt.subplot(img_num, 2, (j*2)+2)
                plt.imshow(output, cmap='gray')
        
        plt.savefig(os.path.join('./result', self.model_name, self.model_name))

        # manifold
        z = make_z(total_mean, total_log_var)
        tsne = TSNE()

        data_2D = tsne.fit_transform(z)
        # min-max normalization
        data_2D = (data_2D-data_2D.min()) / (data_2D.max() - data_2D.min())

        plt.figure(figsize=(8, 6))
        
        base = plt.cm.get_cmap('jet')
        c_list = base(np.linspace(0,1,10))
        cmap_name = base.name + str(10)
        plt.scatter(data_2D[:, 0], data_2D[:, 1], c=total_label, cmap=base.from_list(cmap_name, c_list, 10))
        plt.colorbar(ticks=range(10))
        axes = plt.gca()
        axes.set_xlim([0, 1])
        axes.set_ylim([0, 1])
        plt.grid(True)
        
        plt.savefig(os.path.join('./result', self.model_name, self.model_name + '_2D'))

        # latent vector
        # 1 -> 7

        first_ids = np.where(total_label==1)
        second_ids = np.where(total_label==7)

        start, end = np.mean(z[first_ids], axis=0), np.mean(z[second_ids], axis=0)

        z_data = []
        for a in (np.linspace(0, 1, 20)):
            z_data.append(start*(1-a) + end*a)
        
        z_data = torch.from_numpy(np.array(z_data)).to(self.device)
        output = self.model.decoder(z_data)
        output = output.view(20, 1, 28, 28).detach().cpu()
        
        plt.figure(figsize=(5,5))
        for i in range(20):
            plt.imshow(output[i].permute(1,2,0))
            plt.savefig(os.path.join('./result', self.model_name, "res"+str(i)))

        # save gif

        img_list = os.listdir(os.path.join('./result', self.model_name))
        img_list = [os.path.join('./result', self.model_name, x) for x in img_list if x.startswith('res')]
        img_list = sorted(img_list, key = lambda x:int(x[x.rfind('/')+4: x.rfind('.')]))

        imgs = [Image.open(x) for x in img_list]
        im = imgs[0]
        im.save(os.path.join('./result', self.model_name, 'out.gif'), save_all=True, append_images=imgs[1:], loop=0, duration=300)


        # rm res png
        for png_img in img_list:
            os.remove(png_img)