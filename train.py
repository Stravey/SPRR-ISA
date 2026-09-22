from model.unet_model import UNet
from OIR import clean_ddr_illumination
import torch.nn.functional as F
from extra import ImageFolderDataset
from utils.dataset import FundusSeg_Loader
from torch import optim
import torch.nn as nn
import random
import torch
import numpy as np
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import os
import argparse
import torch.nn.functional as F

parser = argparse.ArgumentParser(
    description='Train Retinex+Swap with cleaned DDR illumination'
)
parser.add_argument('dataset_name', choices=['drive', 'stare', 'chase', 'rimone', 'refuge', 'refuge2'])
parser.add_argument('run_num', type=int)
parser.add_argument('--gpu', type=int, default=0, help='GPU index to use (default: 0)')
args = parser.parse_args()

dataset_name = args.dataset_name
run_num = args.run_num
gpu_id = args.gpu


class TeeLogger:
    """Write console output to both the terminal and a log file."""
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.terminal.flush()
        self.log_file.flush()
        return len(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return self.terminal.isatty()

if dataset_name == "drive":
    train_data_path = "/home/data/guo/dataset/drive/train/"
    valid_data_path = "/home/data/guo/dataset/drive/test/"
    N_epochs = 2500
    lr_decay_step = [2400]
    lr_init = 0.001
    batch_size = 1
    test_epoch = 5
    dataset_mean=[0.4969, 0.2702, 0.1620]
    dataset_std=[0.3479,0.1896,0.1075]
    early_epoch = 400

if dataset_name == "stare":
    train_data_path = "/home/data/guo/dataset/stare/train/"
    valid_data_path = "/home/data/guo/dataset/stare/test/"
    N_epochs = 2500
    lr_decay_step = [2400]
    lr_init = 0.001
    batch_size = 1
    test_epoch = 5
    dataset_mean=[0.5889, 0.3272, 0.1074]
    dataset_std=[0.3458,0.1844,0.1104]
    early_epoch = 400

if dataset_name == "chase":
    train_data_path = "/home/data/guo/dataset/chase_db1/train/"
    valid_data_path = "/home/data/guo/dataset/chase_db1/test/"
    N_epochs = 2500
    lr_decay_step = [2400]
    lr_init = 0.001
    batch_size = 1
    test_epoch = 5
    dataset_mean=[0.4416, 0.1606, 0.0277]
    dataset_std=[0.3530,0.1407,0.0366]
    early_epoch = 400

if dataset_name == "rimone":
    train_data_path = "/home/data/guo/dataset/oc/rimone/train/"
    valid_data_path = "/home/data/guo/dataset/oc/rimone/test/"
    N_epochs = 2500
    lr_decay_step = [2400]
    lr_init = 0.0001
    batch_size = 8
    test_epoch = 2 
    dataset_mean = [0.3383, 0.1164, 0.0465] # In use
    dataset_std = [0.1849, 0.0913, 0.0441]
    early_epoch = 400

if dataset_name == "refuge":
    train_data_path = "/home/data/guo/dataset/oc/refuge/train/"
    valid_data_path = "/home/data/guo/dataset/oc/refuge/train_valid/"
    N_epochs = 2500
    lr_decay_step = [2400]
    lr_init = 0.0001
    batch_size = 8
    test_epoch = 2
    dataset_mean = [0.4237, 0.2414, 0.1182] # In Use
    dataset_std  = [0.1996, 0.1206, 0.0712]
    early_epoch = 400

if dataset_name == "refuge2":
    train_data_path = "/home/data/guo/dataset/oc/refuge/valid_train/"
    valid_data_path = "/home/data/guo/dataset/oc/refuge/valid_test/"
    N_epochs = 2500
    lr_decay_step = [2000]
    lr_init = 0.0001
    batch_size = 8
    test_epoch = 2
    dataset_mean = [0.5984, 0.4048, 0.3161] # In Use
    dataset_std  = [0.2416, 0.1871, 0.1442]
    early_epoch = 400

def train_net(net, device, run_num, epochs=N_epochs, batch_size=batch_size, lr=lr_init):
    train_dataset = FundusSeg_Loader(train_data_path, 1, dataset_name, dataset_mean, dataset_std)
    valid_dataset = FundusSeg_Loader(valid_data_path, 0, dataset_name, dataset_mean, dataset_std)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=2, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False)
    print('Train images: %s' % len(train_loader.dataset))
    print('Valid images: %s' % len(valid_loader.dataset))

    extra_path = "/home/data/guo/dataset/lesion/ddr_512/train/img"
    extra_dataset = ImageFolderDataset(extra_path)
    extra_loader = torch.utils.data.DataLoader(extra_dataset, batch_size=batch_size, shuffle=False, num_workers=2,drop_last=True)

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=lr_decay_step,gamma=0.1)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float('inf')
    best_epoch = 10
    # 为 extra_loader 建立迭代器
    extra_iter = iter(extra_loader)
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        net.train()
        train_loss = 0
        for i, (image, label, filename, raw_height, raw_width) in enumerate(train_loader):
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            if random.random() > 0.5:
                 # ==== 从 extra_loader 取一批 ====
                try:
                    extra_sample = next(extra_iter)
                except StopIteration:
                    extra_iter = iter(extra_loader)
                    extra_sample = next(extra_iter)
                image = random_domain_retinex_swap_extra(
                    image,
                    extra_sample,
                )

            pred = net(image)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

        # Validation
        # epoch != test_epoch
        if ((epoch+1) % test_epoch == 0):
            net.eval()
            val_loss = 0
            for i, (image, label, filename, raw_height, raw_width) in enumerate(valid_loader):
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                pred = net(image)
                loss = criterion(pred, label)
                val_loss = val_loss + loss.item()
            if val_loss < best_loss:
                best_loss = val_loss
                snapshot_name = '{}_b{}.pth'.format(dataset_name, run_num)
                torch.save(net.state_dict(), os.path.join('snapshot', snapshot_name))
                print('saving model............................................')
                best_epoch = epoch
            if (epoch - best_epoch) > early_epoch:
                print('Early Stopping ............................................')
                return
        
            print('Loss/valid', val_loss / i)
            sys.stdout.flush()

        scheduler.step()

if __name__ == "__main__":
    log_dir = os.path.join('logs', dataset_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'seed_{}.log'.format(run_num))
    log_file = open(log_path, mode='a', encoding='utf-8', buffering=1)
    sys.stdout = TeeLogger(sys.stdout, log_file)
    sys.stderr = TeeLogger(sys.stderr, log_file)

    print('\n' + '=' * 80)
    print('Log file:', os.path.abspath(log_path))
    print('Experiment: Retinex+Swap with cleaned DDR illumination')

    random.seed(run_num) 
    np.random.seed(run_num)
    torch.manual_seed(run_num)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available, but GPU training was requested.')
    if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
        raise ValueError('Invalid GPU index {}. Available GPU indices: 0 to {}.'.format(
            gpu_id, torch.cuda.device_count() - 1))
    torch.cuda.set_device(gpu_id)
    torch.cuda.manual_seed(run_num)
    torch.cuda.manual_seed_all(run_num)
    device = torch.device('cuda:{}'.format(gpu_id))
    print('Using GPU {}: {}'.format(gpu_id, torch.cuda.get_device_name(device)))
    net = UNet(n_channels=3, n_classes=1)
    net.to(device=device)

    training_start_time = time.time()
    print('Training start time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(training_start_time)))
    train_net(net, device, run_num)
    training_end_time = time.time()
    print('Training end time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(training_end_time)))
