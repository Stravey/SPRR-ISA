import argparse
import copy
import os
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from model.unet_model import UNet
from utils.dataset import FundusSeg_Loader
from utils.eval_metrics import perform_metrics, cal_f1


warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).resolve().parent
DATASETS = ['drive', 'stare', 'chase', 'rimone', 'refuge', 'refuge2']

parser = argparse.ArgumentParser(description='Predict Retinex+Swap segmentation model')
parser.add_argument('source_domain', choices=DATASETS)
parser.add_argument('target_domain', choices=DATASETS)
parser.add_argument('run_spe', type=int)
parser.add_argument('--gpu', type=int, default=0, help='GPU index to use (default: 0)')
parser.add_argument(
    '--snapshot-dir',
    type=Path,
    default=PROJECT_DIR / 'snapshot',
    help='Directory containing source_bseed.pth weight files.',
)
parser.add_argument(
    '--results-dir',
    type=Path,
    default=PROJECT_DIR / 'results',
    help='Root directory used to save prediction results.',
)
args = parser.parse_args()

source_domain = args.source_domain
target_domain = args.target_domain
run_spe = args.run_spe
gpu_id = args.gpu

model_path = args.snapshot_dir / (source_domain + '_b' + str(run_spe) + '.pth')
save_path = args.results_dir / (
    source_domain + '_to_' + target_domain + '_b' + str(run_spe)
)


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


if source_domain == 'drive':
    dataset_mean = [0.4969, 0.2702, 0.1620]
    dataset_std = [0.3479, 0.1896, 0.1075]

if source_domain == 'stare':
    dataset_mean = [0.5889, 0.3272, 0.1074]
    dataset_std = [0.3458, 0.1844, 0.1104]

if source_domain == 'chase':
    dataset_mean = [0.4416, 0.1606, 0.0277]
    dataset_std = [0.3530, 0.1407, 0.0366]

if source_domain == 'rimone':
    dataset_mean = [0.3383, 0.1164, 0.0465]
    dataset_std = [0.1849, 0.0913, 0.0441]

if source_domain == 'refuge':
    dataset_mean = [0.4237, 0.2414, 0.1182]
    dataset_std = [0.1996, 0.1206, 0.0712]

if source_domain == 'refuge2':
    dataset_mean = [0.5984, 0.4048, 0.3161]
    dataset_std = [0.2416, 0.1871, 0.1442]

if target_domain == 'drive':
    test_data_path = '/home/data/guo/dataset/drive/test/'

if target_domain == 'chase':
    test_data_path = '/home/data/guo/dataset/chase_db1/test/'

if target_domain == 'stare':
    test_data_path = '/home/data/guo/dataset/stare/test/'

if target_domain == 'rimone':
    test_data_path = '/home/data/guo/dataset/oc/rimone/test/'

if target_domain == 'refuge':
    test_data_path = '/home/data/guo/dataset/oc/refuge/train_valid/'

if target_domain == 'refuge2':
    test_data_path = '/home/data/guo/dataset/oc/refuge/valid_test/'


if __name__ == '__main__':
    os.makedirs(save_path, exist_ok=True)
    log_path = save_path / 'predict.log'
    log_file = open(log_path, mode='w', encoding='utf-8', buffering=1)
    sys.stdout = TeeLogger(sys.stdout, log_file)
    sys.stderr = TeeLogger(sys.stderr, log_file)

    print('Log file:', log_path.resolve())
    print('Source domain:', source_domain)
    print('Target domain:', target_domain)
    print('Seed:', run_spe)
    print('Model path:', model_path.resolve())
    print('Save path:', save_path.resolve())

    with torch.no_grad():
        test_dataset = FundusSeg_Loader(
            test_data_path,
            0,
            target_domain,
            dataset_mean,
            dataset_std,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
        )
        print('Testing images: %s' % len(test_loader.dataset))

        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available, but GPU prediction was requested.')
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise ValueError(
                'Invalid GPU index {}. Available GPU indices: 0 to {}.'.format(
                    gpu_id, torch.cuda.device_count() - 1
                )
            )
        torch.cuda.set_device(gpu_id)
        device = torch.device('cuda:{}'.format(gpu_id))
        print('Using GPU {}: {}'.format(gpu_id, torch.cuda.get_device_name(device)))

        net = UNet(n_channels=3, n_classes=1)
        net.to(device=device)
        print('Loading model {}'.format(model_path.resolve()))
        net.load_state_dict(torch.load(model_path, map_location=device),strict=False)

        net.eval()
        pre_stack = []
        label_stack = []
        saved_images = 0

        for image, label, filename, raw_height, raw_width in test_loader:
            image = image.cuda().float()
            label = label.cuda().float()

            image = image.to(device=device, dtype=torch.float32)
            pred = net(image)
            pred = torch.sigmoid(pred)
            pred = pred[:, :, :raw_height, :raw_width]
            label = label[:, :, :raw_height, :raw_width]
            pred = pred.cpu().numpy().astype(np.double)[0][0]
            label = label.cpu().numpy().astype(np.double)[0][0]

            pre_stack.append(pred)
            label_stack.append(label)

            pred = pred * 255
            save_filename = save_path / (filename[0] + '.png')
            if not cv2.imwrite(str(save_filename), pred):
                raise IOError('Failed to save prediction image: {}'.format(save_filename))
            saved_images += 1

        print('Evaluating...')
        label_stack = np.stack(label_stack, axis=0)
        pre_stack = np.stack(pre_stack, axis=0)
        label_stack = label_stack.reshape(-1)
        pre_stack = pre_stack.reshape(-1)

        if (
            target_domain == 'rimone'
            or target_domain == 'refuge'
            or target_domain == 'refuge2'
        ):
            f1 = cal_f1(pre_stack, label_stack)
            print('F1-score: {}'.format(f1))
        else:
            # precision, sen, spec, f1, acc, roc_auc, pr_auc = perform_metrics(
            #     pre_stack, label_stack
            # )
            # print(
            #     'Precision: {} Sen: {} Spec:{} F1-score: {} Acc: {} '
            #     'ROC_AUC: {} PR_AUC: {}'.format(
            #         precision, sen, spec, f1, acc, roc_auc, pr_auc
            #     )
            # )
            f1 = cal_f1(pre_stack, label_stack)
            print('F1-score: {}'.format(f1))

        print('Saved prediction images:', saved_images)
