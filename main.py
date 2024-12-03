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

# import shap

def main(epochs,lr):
    model = LSTMModel().to('cuda')
    torch.manual_seed(3407)
    json_path = ".\\label_2.json"

    train_datasets = []
    test_datasets = []

    for participant_id in range(1, 44):
        # TODO: 7 8 9 10 11 12 16 21 only
        train_dataset = ds_smarteye(json_path, participant_id, train=True,)
        # test_dataset = ds_smarteye(json_path, participant_id, train=False)
        train_datasets.append(train_dataset)
        # test_datasets.append(test_dataset)

    # 合并所有参与者的数据集
    combined_train_dataset = ConcatDataset(train_datasets)
    # combined_test_dataset = ConcatDataset(test_datasets)

    train_dataloader = torch.utils.data.DataLoader(
        combined_train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn,
    )
    # test_dataloader = torch.utils.data.DataLoader(
    #     combined_test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn
    # )

    # criterion = focal_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # 1e-3 for LSTM
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-9)

    # for epoch in range(epochs):
        # train_model(model, train_dataloader, criterion, optimizer, test_dataloader, participant_id)

    for participant_id in range(1, 44):
        if participant_id in [15,20]: # delete No.15 & 20
            continue
        test_dataset = ds_smarteye(json_path, participant_id, train=False)
        test_datasets.append(test_dataset)
    combined_test_dataset = ConcatDataset(test_datasets)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn
        )
        # print(f'Now Participant {participant_id} is testing:')
        # evaluate_model(model, test_dataloader, participant_id, single=True)
    test_dataloader = torch.utils.data.DataLoader(
    combined_test_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    train_model(model, train_dataloader, optimizer, scheduler, test_dataloader, participant_id, epochs=epochs)
        # evaluate_model(model, test_dataloader, participant_id, criterion)

    # explainer = shap.DeepExplainer(model, X_train)
    # shap_values = explainer.shap_values(X_test)
    # shap.summary_plot(shap_values, X_test)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    epochs = 10
    lr = 3e-4
    cur_date = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    log_dir = f'./work_dir/logs/'
    os.makedirs(log_dir, exist_ok=True)

    log_file_path = os.path.join(log_dir, f'{cur_date}_epochs{epochs}_lr{lr}.txt')

    with open(log_file_path, 'w') as f:
        old_stdout = sys.stdout  # 备份原始标准输出
        sys.stdout = f 
        f.write(f"Start Training: {cur_date}\n")
        f.write(f"Parameters: epochs={epochs}, lr={lr}\n")
        # 捕获 main 的输出
        try:
            main(epochs,lr)
            f.write("Training completed successfully.\n")
        except Exception as e:
            f.write(f"Training failed with error: {str(e)}\n")
