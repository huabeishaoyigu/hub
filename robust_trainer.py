import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, ReduceLROnPlateau
import os
import matplotlib.pyplot as plt

class RobustTrainer:
    """
    支持对抗训练 (Adversarial Training) 的训练框架
    基于 Goodfellow 的理论：将对抗样本纳入训练集
    """
    def __init__(self, model_name='resnet18', dataset='CIFAR10', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dataset_name = dataset
        self.model = self._get_model(model_name).to(self.device)
        self.loader = self._get_data()
        
    def _get_model(self, name):
        if self.dataset_name == 'MNIST':
            from collections import OrderedDict
            return nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(1, 20, 5)), ('relu1', nn.ReLU()), ('pool1', nn.MaxPool2d(2)),
                ('conv2', nn.Conv2d(20, 50, 5)), ('relu2', nn.ReLU()), ('pool2', nn.MaxPool2d(2)),
                ('flatten', nn.Flatten()), ('fc1', nn.Linear(800, 500)), ('relu3', nn.ReLU()),
                ('fc2', nn.Linear(500, 10))
            ]))
        else:
            model = models.__dict__[name]()
            if hasattr(model, 'fc'):
                model.fc = nn.Linear(model.fc.in_features, 10)
            elif hasattr(model, 'classifier'):
                if isinstance(model.classifier, nn.Sequential):
                    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 10)
            return model

    def _get_data(self):
        if self.dataset_name == 'MNIST':
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        else:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=128, shuffle=True)

    def train(self, epochs=10, adv_train=False, epsilon=0.1, 
              lr_scheduler='cosine', initial_lr=0.1):
        """
        核心训练逻辑 - 添加动态学习率
        :param adv_train: 是否开启对抗训练
        :param lr_scheduler: 学习率调度器类型 ('cosine', 'step', 'plateau', 'none')
        :param initial_lr: 初始学习率
        """
        optimizer = optim.SGD(self.model.parameters(), lr=initial_lr, 
                             momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        # 选择学习率调度器
        if lr_scheduler == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
            print(f"使用Cosine退火学习率: {initial_lr} -> 1e-4")
        elif lr_scheduler == 'step':
            scheduler = MultiStepLR(optimizer, milestones=[epochs//2, epochs*3//4], gamma=0.1)
            print(f"使用Step学习率: milestones at {[epochs//2, epochs*3//4]}")
        elif lr_scheduler == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                         patience=2, verbose=True)
            print(f"使用ReduceLROnPlateau: 基于验证损失调整")
        else:
            scheduler = None
            print(f"使用固定学习率: {initial_lr}")
        
        print(f"开始训练 | 模式: {'对抗训练' if adv_train else '标准训练'} | 设备: {self.device}")
        
        # 记录训练历史
        history = {
            'loss': [],
            'acc': [],
            'lr': []
        }
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for i, (images, labels) in enumerate(self.loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # 如果是对抗训练，实时生成攻击样本
                if adv_train:
                    images.requires_grad = True
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    self.model.zero_grad()
                    loss.backward()
                    
                    # FGSM 扰动
                    adv_images = images + epsilon * images.grad.sign()
                    images = torch.clamp(adv_images, 0, 1).detach()
                    images.requires_grad = False

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # 计算准确率
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                if i % 100 == 99:  # 每100个batch打印一次
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch [{epoch+1}/{epochs}] Batch [{i+1}] "
                          f"Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}% "
                          f"LR: {current_lr:.6f}")

            epoch_acc = 100. * correct / total
            epoch_loss = total_loss / len(self.loader)
            
            # 记录历史
            history['loss'].append(epoch_loss)
            history['acc'].append(epoch_acc)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # 更新学习率
            if scheduler is not None:
                if lr_scheduler == 'plateau':
                    scheduler.step(epoch_loss)  # 基于损失调整
                else:
                    scheduler.step()  # 基于epoch调整
            
            print(f"Epoch [{epoch+1}/{epochs}] Completed | "
                  f"Avg Loss: {epoch_loss:.4f} | "
                  f"Acc: {epoch_acc:.2f}% | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存模型
        os.makedirs('./models', exist_ok=True)
        if adv_train:
            model_type = 'robust'
        else:
            model_type = 'std'
        
        model_name = f"{self.dataset_name}_{model_type}_lr{lr_scheduler}"
        save_path = f"./models/{model_name}.pth"
        torch.save(self.model.state_dict(), save_path)
        print(f"模型训练完成并保存至: {save_path}")
        
        # 绘制训练曲线
        self.plot_training_history(history, model_name)
        
        return history
    
    def plot_training_history(self, history, model_name):
        """绘制训练历史曲线"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 损失曲线
        axes[0].plot(history['loss'], 'b-', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].grid(True, alpha=0.3)
        
        # 准确率曲线
        axes[1].plot(history['acc'], 'g-', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training Accuracy')
        axes[1].grid(True, alpha=0.3)
        
        # 学习率曲线
        axes[2].plot(history['lr'], 'r-', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f'Training History: {model_name}', fontsize=14)
        plt.tight_layout()
        
        # 保存图像
        os.makedirs('./training_plots', exist_ok=True)
        plot_path = f'./training_plots/{model_name}_history.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"训练曲线已保存至: {plot_path}")

if __name__ == "__main__":
    # 使用示例：执行对抗训练，使用Cosine退火学习率
    trainer = RobustTrainer(dataset='MNIST')
    
    # 正确调用 - 启用动态学习率
    trainer.train(
        epochs=50, 
        adv_train=True, 
        epsilon=0.2,
        lr_scheduler='cosine',  # 使用余弦退火
        initial_lr=0.1          # 初始学习率
    )