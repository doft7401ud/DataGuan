import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torchvision.ops import sigmoid_focal_loss

def train_model(model, train_dataloader, optimizer, scheduler, test_dataloader, participant_id, epochs=10):
    model.train()
    cur_date = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'./work_dir/tensorboards/{cur_date}_epochs{epochs}_lr{optimizer.param_groups[0]["lr"]}'
    writer = SummaryWriter(log_dir)
    
    for epoch in range(epochs):
        model.train()  # 训练模式
        running_loss = 0.0
        
        for batch in train_dataloader:
            inputs = batch['data'].to('cuda')
            seq_lengths = batch['seq_lengths']
            labels = batch['label'].float().to('cuda')
            
            # 前向传播
            outputs = model(inputs, seq_lengths)
            outputs = outputs.view(-1)
            loss = sigmoid_focal_loss(
                outputs,         # 输入 logits
                labels,          # 目标标签
                alpha=0.25,      # 平衡因子，默认值可调整
                gamma=2.0,       # 调整难易样本的参数，默认值可调整
                reduction='mean' # 损失的聚合方式
            )
            
            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        scheduler.step()
        # 记录训练损失
        writer.add_scalar('Loss/train', running_loss / len(train_dataloader), epoch)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_dataloader):.4f}")
        
        # 训练一个 epoch 后立即进行评估
        evaluate_model(model, test_dataloader, participant_id, writer, epoch)
    
    writer.close()

def evaluate_model(model, test_dataloader, participant_id, writer, epoch, single=False):
    model.eval()  # 评估模式
    total = 0
    correct = 0

    TP = 0  
    FP = 0  
    TN = 0  
    FN = 0  
    running_loss = 0.0  # 用于记录评估时的损失

    with torch.no_grad():
        for batch in test_dataloader:
            inputs = batch['data'].to('cuda')
            seq_lengths = batch['seq_lengths']
            labels = batch['label'].float().to('cuda')

            outputs = model(inputs, seq_lengths)
            outputs = outputs.view(-1)  # 调整形状
            labels = labels.view(-1)    # 确保形状匹配
            loss = sigmoid_focal_loss(outputs, labels, alpha=0.25, gamma=2.0, reduction='mean')
            running_loss += loss.item()  # 记录评估损失

            # 将 logits 转换为概率
            probabilities = torch.sigmoid(outputs)
            predicted = probabilities >= 0.5

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 统计TP、FP、TN、FN
            TP += ((predicted == 1) & (labels == 1)).sum().item()
            FP += ((predicted == 1) & (labels == 0)).sum().item()
            TN += ((predicted == 0) & (labels == 0)).sum().item()
            FN += ((predicted == 0) & (labels == 1)).sum().item()

    # 计算评估的平均损失
    eval_loss = running_loss / len(test_dataloader)
    writer.add_scalar('Loss/eval', eval_loss, epoch)  # 记录评估损失到TensorBoard

    accuracy = correct / total
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    if single:
        print(f'{participant_id} test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}')
    else:
        print(f'Overall accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}')
