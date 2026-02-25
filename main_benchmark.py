import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import os

# 确保可以导入 adversarial_suite
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ==========================================
# 1. 模型结构定义 (必须与训练时完全一致)
# ==========================================
def get_mnist_model():
    from collections import OrderedDict
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(1, 20, 5)), ('relu1', nn.ReLU()), ('pool1', nn.MaxPool2d(2)),
        ('conv2', nn.Conv2d(20, 50, 5)), ('relu2', nn.ReLU()), ('pool2', nn.MaxPool2d(2)),
        ('flatten', nn.Flatten()), ('fc1', nn.Linear(800, 500)), ('relu3', nn.ReLU()),
        ('fc2', nn.Linear(500, 10))
    ]))

# ==========================================
# 2. 攻击可视化函数
# ==========================================
def save_attack_visualization(model, device, data_loader, attacker, epsilon, save_path="result.png"):
    """
    捕获攻击成功与失败的案例，并保存为并排对比图
    """
    model.eval()
    success_sample = None
    failure_sample = None

    print(f"正在寻找可视化案例 (Epsilon: {epsilon})...")
    
    # 创建保存可视化样本的目录
    os.makedirs('./visualizations', exist_ok=True)
    
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        
        # 使用 I-FGSM 攻击来生成可视化样本
        adv_images = attacker.i_fgsm_attack(images, labels, epsilon=epsilon)
        
        with torch.no_grad():
            # 原始预测
            orig_outputs = model(images)
            orig_preds = orig_outputs.max(1)[1]
            
            # 对抗预测
            adv_outputs = model(adv_images)
            adv_preds = adv_outputs.max(1)[1]
            
            for i in range(images.size(0)):
                # 只分析原始预测正确的样本
                if orig_preds[i] == labels[i]:
                    # 案例 A: 攻击成功 (预测改变)
                    if adv_preds[i] != labels[i] and success_sample is None:
                        success_sample = {
                            'original': images[i].cpu(),
                            'adversarial': adv_images[i].cpu(),
                            'label': labels[i].item(),
                            'orig_pred': orig_preds[i].item(),
                            'adv_pred': adv_preds[i].item()
                        }
                    # 案例 B: 攻击失败 (预测依然正确)
                    elif adv_preds[i] == labels[i] and failure_sample is None:
                        failure_sample = {
                            'original': images[i].cpu(),
                            'adversarial': adv_images[i].cpu(),
                            'label': labels[i].item(),
                            'orig_pred': orig_preds[i].item(),
                            'adv_pred': adv_preds[i].item()
                        }
        
        if success_sample and failure_sample:
            break

    # 绘图逻辑
    if success_sample and failure_sample:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # 第一行：展现攻击如何"欺骗"了模型
        axes[0, 0].imshow(success_sample['original'].squeeze(), cmap='gray')
        axes[0, 0].set_title(f"Original\nLabel: {success_sample['label']}, Pred: {success_sample['orig_pred']}")
        
        axes[0, 1].imshow(success_sample['adversarial'].squeeze(), cmap='gray')
        axes[0, 1].set_title(f"Attack Success\nPred: {success_sample['adv_pred']}")
        
        # 第二行：展现模型的"鲁棒性"（防御成功）
        axes[1, 0].imshow(failure_sample['original'].squeeze(), cmap='gray')
        axes[1, 0].set_title(f"Original\nLabel: {failure_sample['label']}, Pred: {failure_sample['orig_pred']}")
        
        axes[1, 1].imshow(failure_sample['adversarial'].squeeze(), cmap='gray')
        axes[1, 1].set_title(f"Attack Failure\nPred: {failure_sample['adv_pred']}")
        
        for ax in axes.flatten():
            ax.axis('off')
            ax.grid(False)
        
        plt.suptitle(f'Adversarial Attack Visualization (ε={epsilon})', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"对比图已成功保存至: {save_path}")
        
        # 打印样本信息
        print(f"\n成功案例: 标签 {success_sample['label']} → 预测从 {success_sample['orig_pred']} 变为 {success_sample['adv_pred']}")
        print(f"失败案例: 标签 {failure_sample['label']} → 预测保持为 {failure_sample['adv_pred']}")
    else:
        print("提示：未能同时找到成功和失败的样本。可能 epsilon 太小导致全部防御，或太大导致全部沦陷。")

# ==========================================
# 3. 主程序入口
# ==========================================
def main():
    # --- 配置参数 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model_path = "./models/MNIST_robust.pth"  # 必须先运行训练脚本生成此文件
    epsilon_value = 0.2                       # 攻击强度 (0.0 到 1.0)
    
    # --- 数据准备 ---
    print("加载测试数据集...")
    test_dataset = datasets.MNIST(
        './data', 
        train=False, 
        download=True, 
        transform=transforms.ToTensor()
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    
    # --- 模型加载 ---
    print(f"加载模型: {model_path}")
    model = get_mnist_model().to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"状态: 已加载稳健模型权重 {model_path}")
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 {model_path}")
        print("请先运行 robust_trainer.py 训练模型:")
        print("python robust_trainer.py")
        return
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return

    # --- 导入攻击套件 ---
    try:
        from adversarial_suite import AdversarialAttacker
    except ImportError:
        print("错误: 无法导入 adversarial_suite。请确保 adversarial_suite.py 在同一目录下。")
        return

    # --- 初始化攻击套件 ---
    config = {'epsilon': epsilon_value, 'alpha': 0.01, 'num_iter': 10}
    attacker = AdversarialAttacker(model, device, config=config)

    # --- 评估干净样本的准确率 ---
    print("\n" + "="*50)
    print("评估干净样本的准确率...")
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    print(f"干净准确率: {100.*correct/total:.2f}%")

    # --- 执行定量评估 ---
    print("\n" + "="*50)
    print("1. 正在评估 FGSM (单步攻击)...")
    attacker.evaluate(test_loader, attack_type='fgsm', epsilon=epsilon_value)
    
    print("\n" + "="*50)
    print("2. 正在评估 I-FGSM (迭代攻击)...")
    attacker.evaluate(test_loader, attack_type='i-fgsm', epsilon=epsilon_value)
    print("="*50)

    # --- 执行定性可视化 ---
    print("\n" + "="*50)
    print("生成攻击可视化图...")
    save_attack_visualization(
        model=model, 
        device=device, 
        data_loader=test_loader, 
        attacker=attacker, 
        epsilon=epsilon_value, 
        save_path="./visualizations/attack_visualization.png"
    )

if __name__ == "__main__":
    main()