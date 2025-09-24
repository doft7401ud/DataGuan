import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from torch import nn
from torch.utils.data import ConcatDataset
from load_dataset import ds_smarteye
from model import *
from runner import train_model, evaluate_model
from datetime import datetime
import warnings
import sys
import logging
import argparse

# import shap

def main(epochs,lr):
    model = LSTMModel().to('cuda')
    torch.manual_seed(3407)
    json_path = "./label_2.json"

    train_datasets = []
    test_datasets = []

    for participant_id in range(1, 44):
        train_dataset = ds_smarteye(json_path, participant_id, train=True,)
        # test_dataset = ds_smarteye(json_path, participant_id, train=False)
        train_datasets.append(train_dataset)
        # test_datasets.append(test_dataset)

    # 合并所有参与者的数据集
    combined_train_dataset = ConcatDataset(train_datasets)

    train_dataloader = torch.utils.data.DataLoader(
        combined_train_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # 1e-3 for LSTM
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-9)

    for participant_id in range(1, 44):
        if participant_id in [19, 32, 2, 18, 22, 8, 25, 9, 11, 28, 24, 17, 6]: # delete 
            continue
        test_dataset = ds_smarteye(json_path, participant_id, train=False)
        test_datasets.append(test_dataset)
    combined_test_dataset = ConcatDataset(test_datasets)
    test_dataloader = torch.utils.data.DataLoader(combined_test_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=4)
    train_model(model, train_dataloader, optimizer, scheduler, test_dataloader, participant_id, epochs=epochs)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=9e-4, help="Learning rate")
    args = parser.parse_args()

    epochs = args.epochs
    lr = args.lr
    cur_date = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 配置日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 创建文件 Handler（日志文件）
    log_filename = f"./work_dir/logs/train_{cur_date}_lr{lr}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)

    # 创建控制台 Handler（终端输出）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # 只让 INFO 及以上日志出现在终端

    # 设置日志格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加 Handler 到 Logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 运行主函数
    main(epochs, lr)

    # log_dir = f'./work_dir/logs/'
    # os.makedirs(log_dir, exist_ok=True)

    # log_file_path = os.path.join(log_dir, f'{cur_date}_epochs{epochs}_lr{lr}.txt')

    # with open(log_file_path, 'w') as f:
    #     old_stdout = sys.stdout  # 备份原始标准输出
    #     sys.stdout = f 
    #     f.write(f"Start Training: {cur_date}\n")
    #     f.write(f"Parameters: epochs={epochs}, lr={lr}\n")
    #     # 捕获 main 的输出
    #     try:
    #         main(epochs,lr)
    #         f.write("Training completed successfully.\n")
    #     except Exception as e:
    #         f.write(f"Training failed with error: {str(e)}\n")
