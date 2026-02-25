import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class AdversarialAttacker:
    """
    专业级对抗攻击套件
    支持：FGSM (单步), I-FGSM (迭代), 定向/非定向攻击
    """
    def __init__(self, model, device, config=None):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        # 默认配置
        self.config = config if config else {
            'epsilon': 0.1,
            'alpha': 0.01,
            'num_iter': 10,
            'mean': [0.485, 0.456, 0.406],  # ImageNet 均值
            'std': [0.229, 0.224, 0.225]    # ImageNet 标准差
        }

    def _normalize(self, x):
        """对输入张量进行标准化"""
        if x.shape[1] == 1: 
            # MNIST 数据集，不需要ImageNet标准化
            return x
        else:
            # CIFAR10或其他彩色图像数据集
            mean = torch.tensor(self.config['mean']).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor(self.config['std']).view(1, 3, 1, 1).to(self.device)
            return (x - mean) / std

    def fgsm_attack(self, images, labels, epsilon=None, target_labels=None):
        """
        Fast Gradient Sign Method (FGSM)
        :param target_labels: 如果提供，执行定向攻击 (Targeted Attack)
        """
        eps = epsilon if epsilon is not None else self.config['epsilon']
        images = images.clone().detach().to(self.device)
        labels = labels.to(self.device)
        
        images.requires_grad = True

        outputs = self.model(self._normalize(images))
        
        if target_labels is not None:
            # 定向攻击：减小与目标标签的损失
            target_labels = target_labels.to(self.device)
            loss = F.cross_entropy(outputs, target_labels)
            self.model.zero_grad()
            loss.backward()
            grad = images.grad.data
            adv_images = images - eps * grad.sign()
        else:
            # 非定向攻击：增大与原始标签的损失
            loss = F.cross_entropy(outputs, labels)
            self.model.zero_grad()
            loss.backward()
            grad = images.grad.data
            adv_images = images + eps * grad.sign()

        adv_images = torch.clamp(adv_images, 0, 1).detach()
        return adv_images

    def i_fgsm_attack(self, images, labels, epsilon=None, alpha=None, num_iter=None):
        """Basic Iterative Method (I-FGSM / BIM)"""
        eps = epsilon if epsilon is not None else self.config['epsilon']
        alp = alpha if alpha is not None else self.config['alpha']
        iters = num_iter if num_iter is not None else self.config['num_iter']
        
        adv_images = images.clone().detach().to(self.device)
        ori_images = images.clone().detach().to(self.device)

        for _ in range(iters):
            adv_images.requires_grad = True
            outputs = self.model(self._normalize(adv_images))
            loss = F.cross_entropy(outputs, labels.to(self.device))
            
            self.model.zero_grad()
            loss.backward()
            
            grad = adv_images.grad.data
            adv_images = adv_images + alp * grad.sign()
            
            # 限制扰动在 epsilon 的 L-infinity 球内
            delta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
            adv_images = torch.clamp(ori_images + delta, 0, 1).detach()
            
        return adv_images

    def evaluate(self, data_loader, attack_type='fgsm', epsilon=0.1):
        """全面评估模型的鲁棒性"""
        success_count = 0
        total_count = 0
        l_inf_dist = []

        print(f"正在执行 {attack_type.upper()} 攻击评估 (ε={epsilon})...")
        
        # 先计算原始准确率（仅对原本预测正确的样本进行攻击成功率统计）
        original_correct = 0
        for images, labels in data_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(self._normalize(images))
                preds = outputs.max(1)[1]
                original_correct += preds.eq(labels).sum().item()
        
        # 重新遍历数据加载器进行攻击评估
        for images, labels in tqdm(data_loader, desc=f"{attack_type}攻击进度"):
            images, labels = images.to(self.device), labels.to(self.device)
            
            with torch.no_grad():
                # 检查原始预测是否正确
                orig_outputs = self.model(self._normalize(images))
                orig_preds = orig_outputs.max(1)[1]
                correct_mask = orig_preds.eq(labels)
            
            # 只对原始预测正确的样本进行攻击
            correct_indices = correct_mask.nonzero(as_tuple=True)[0]
            if len(correct_indices) == 0:
                continue
                
            correct_images = images[correct_indices]
            correct_labels = labels[correct_indices]
            
            # 生成攻击样本
            if attack_type == 'fgsm':
                adv_images = self.fgsm_attack(correct_images, correct_labels, epsilon=epsilon)
            else:
                adv_images = self.i_fgsm_attack(correct_images, correct_labels, epsilon=epsilon)

            with torch.no_grad():
                outputs = self.model(self._normalize(adv_images))
                preds = outputs.max(1)[1]
                
                # 统计攻击成功率（原本正确但现在错误的）
                attack_success_mask = preds != correct_labels
                success_count += attack_success_mask.sum().item()
                total_count += correct_labels.size(0)
                
                # 计算 L-inf 距离
                dist = torch.norm((adv_images - correct_images).view(correct_images.shape[0], -1), 
                                 p=float('inf'), dim=1)
                l_inf_dist.extend(dist.cpu().numpy())

        if total_count > 0:
            attack_success_rate = success_count / total_count
            avg_l_inf = np.mean(l_inf_dist) if l_inf_dist else 0
            print(f"\n[攻击结果]")
            print(f"原始准确率: {original_correct/len(data_loader.dataset):.2%}")
            print(f"攻击成功率: {attack_success_rate:.2%} (在原始正确的样本上)")
            print(f"扰动强度: 平均 L∞ 距离: {avg_l_inf:.4f}")
            print(f"鲁棒准确率: {(1 - attack_success_rate) * 100:.2f}%")
        else:
            print("警告: 没有找到原始预测正确的样本进行攻击评估。")