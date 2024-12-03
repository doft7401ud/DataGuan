import json
import torch
from torch.utils.data import Dataset
import pandas as pd

class ds_smarteye(Dataset):
    def __init__(self, json_path, participant_id, train=True, split_ratio=0.9):
        """
        json_path: 存放标签和数据的 JSON 文件路径
        participant_id: 实验者编号，例如 'Participant_1'
        train: 如果为 True，则加载训练数据，否则加载测试数据
        split_ratio: 用于划分训练集和测试集的比例，默认 90% 为训练集
        """
        self.participant_id = f"Participant_{participant_id}"  # 例如 'Participant_1'
        self.data = []
        self.labels = []
        
        # 加载 JSON 文件中的数据和标签
        with open(json_path, 'r') as f:
            self.data_dict = json.load(f)
        
        # 获取该实验者对应的实验数据
        participant_data = self.data_dict.get(self.participant_id, [])
        
        # 读取数据和对应的标签
        for experiment_info in participant_data:
            data = pd.read_csv(f'.\\final\\{experiment_info['experiment']}')
            columns_to_keep = [
                '6', 
                '7', 
                '8', 
                '9', 
                '10', 
                '11', 
                '15', 
                '20',
                'Blink',
                'EyelidOpeningChange',
                'EyelidOpening',
                'Fixationrecalculated(20240104)',
                'GazeDirectionX',
                'GazeDirectionY',
                'GazeObjectsrecalculated(Reidentified+gapfilled)',
                'movingGazeRatioeachLaneChange',
                'PupilDiameterChange',
                'PupilDiameter',
                'Saccaderecalculated(20231231)'
            ]
            # 选择指定的列
            data = data[columns_to_keep]
            label = experiment_info['label']
            label = 1 if label else 0  # 将 true/false 转换为 1/0 标签
            self.data.append(data)
            self.labels.append(label)

        # 根据 split_ratio 划分数据
        split_index = int(len(self.data) * split_ratio)
        if train:
            self.data = self.data[:split_index]  # 训练集
            self.labels = self.labels[:split_index]  # 对应训练集的标签
        else:
            self.data = self.data[split_index:]  # 测试集
            self.labels = self.labels[split_index:]  # 对应测试集的标签

        # 打印训练集或测试集的总数
        # print(f"Loaded {len(self.data)} {'training' if train else 'testing'} samples from Participant {participant_id}.")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        seq_length = len(data)
        
        # 如果 data 是 DataFrame，转换为 numpy 数组
        if isinstance(data, pd.DataFrame):
            data = data.values  # 将 DataFrame 转换为 numpy 数组
        
        # 转换为 PyTorch 张量
        data_tensor = torch.tensor(data, dtype=torch.float) # (sequence_length, 1)
        label_tensor = torch.tensor(label, dtype=torch.long)  # 二分类标签 (1 或 0)
        # print(data_tensor.shape)
        return {
            'data': data_tensor,  # (sequence_length, 1)
            'label': label_tensor,  # 二分类标签
            'seq_length': seq_length  # 原始序列长度
        }

# 使用示例：加载单个实验者的数据
if __name__ == '__main__':
    json_path = ".\\label_2.json"
    participant_id = 1  # 实验者编号

    train_dataset = ds_smarteye(json_path, participant_id, train=True)
    test_dataset = ds_smarteye(json_path, participant_id, train=False)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)
    for batch in train_dataloader:
        print(batch)