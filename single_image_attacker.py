import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class SingleImageAttacker:
    """
    Single Image Adversarial Attack Tool
    """
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path).to(self.device)
        self.model.eval()
        
        # MNIST标准化参数
        self.mnist_mean = 0.1307
        self.mnist_std = 0.3081
        
        # 为测试集图片使用的预处理
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((self.mnist_mean,), (self.mnist_std,))
        ])
        
        # 为用户上传图片使用的预处理
        self.user_transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            self._auto_preprocess,  # 自动预处理
            transforms.Normalize((self.mnist_mean,), (self.mnist_std,))
        ])
        
    def _auto_preprocess(self, tensor):
        """
        自动预处理用户上传的图片
        1. 自动反转颜色（如果是白底黑字）
        2. 归一化到[0, 1]
        3. 确保对比度合适
        """
        # 自动颜色反转
        if tensor.mean() > 0.5:
            tensor = 1 - tensor
        
        # 增强对比度
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val - min_val > 0.01:  # 避免除零
            tensor = (tensor - min_val) / (max_val - min_val)
        
        # 二值化阈值处理
        threshold = 0.3
        tensor = torch.where(tensor > threshold, torch.tensor(1.0), tensor)
        
        return tensor
        
    def _load_model(self, model_path):
        """Load trained model"""
        from collections import OrderedDict
        model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 20, 5)), ('relu1', nn.ReLU()), ('pool1', nn.MaxPool2d(2)),
            ('conv2', nn.Conv2d(20, 50, 5)), ('relu2', nn.ReLU()), ('pool2', nn.MaxPool2d(2)),
            ('flatten', nn.Flatten()), ('fc1', nn.Linear(800, 500)), ('relu3', nn.ReLU()),
            ('fc2', nn.Linear(500, 10))
        ]))
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Model loaded successfully: {model_path}")
            return model
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise
    
    def load_test_image(self, image_path):
        """加载MNIST测试集图片"""
        try:
            from torchvision import datasets
            test_dataset = datasets.MNIST('./data', train=False, download=True, 
                                         transform=transforms.ToTensor())
            
            # 按标签选择图片
            selected_digit = int(input("Enter digit (0-9): ").strip())
            indices = [i for i, (_, label) in enumerate(test_dataset) if label == selected_digit]
            
            if not indices:
                print(f"No images found for digit {selected_digit}")
                return None, None
            
            idx = indices[0]
            image_tensor, label = test_dataset[idx]
            image_tensor = image_tensor.unsqueeze(0)
            
            # 应用标准化
            image_tensor = transforms.Normalize((self.mnist_mean,), (self.mnist_std,))(image_tensor)
            
            original_image = Image.fromarray((image_tensor.squeeze().numpy() * 255).astype(np.uint8))
            
            print(f"✅ Loaded test image: digit {selected_digit}")
            print(f"  Tensor shape: {image_tensor.shape}")
            print(f"  Pixel range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
            
            return image_tensor, original_image
            
        except Exception as e:
            print(f"❌ Test image loading failed: {e}")
            return None, None
    
    def load_user_image(self, image_path):
        """加载并预处理用户上传的图片"""
        try:
            if not os.path.exists(image_path):
                print(f"❌ Image not found: {image_path}")
                return None, None
            
            # 打开图片
            image = Image.open(image_path)
            print(f"📷 Original image: {image_path}")
            print(f"  Mode: {image.mode}, Size: {image.size}")
            
            # 保存原始图片用于显示
            original_image = image.copy()
            
            # 应用用户预处理流程
            image_tensor = self.user_transform(image).unsqueeze(0)
            
            # 调试信息
            print(f"🔄 Preprocessed image:")
            print(f"  Tensor shape: {image_tensor.shape}")
            print(f"  Pixel range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
            print(f"  Mean value: {image_tensor.mean():.3f}")
            print(f"  Std value: {image_tensor.std():.3f}")
            
            # 显示预处理后的图片
            self._show_preview(original_image, image_tensor)
            
            return image_tensor, original_image
            
        except Exception as e:
            print(f"❌ Image loading failed: {e}")
            return None, None
    
    def _show_preview(self, original_image, processed_tensor):
        """显示预处理前后的对比"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            
            # 原始图片
            axes[0].imshow(original_image, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # 处理后的图片
            processed_np = processed_tensor.squeeze().cpu().numpy()
            axes[1].imshow(processed_np, cmap='gray')
            axes[1].set_title('Processed for Model')
            axes[1].axis('off')
            
            plt.suptitle('Image Preprocessing Preview', fontsize=12)
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(2)
            plt.close()
        except:
            pass  # 如果显示失败，继续执行
    
    def predict(self, image_tensor, verbose=True):
        """模型预测，添加调试信息"""
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            
            if verbose:
                print(f"📊 Model input analysis:")
                print(f"  Shape: {image_tensor.shape}")
                print(f"  Range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
                print(f"  Mean: {image_tensor.mean():.3f}, Std: {image_tensor.std():.3f}")
            
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            if verbose:
                print(f"\n🎯 Prediction results:")
                for i, prob in enumerate(probabilities.cpu().numpy()[0]):
                    if prob > 0.01:  # 只显示概率大于1%的
                        print(f"  Digit {i}: {prob:.2%}")
                print(f"  ➤ Predicted: {predicted.item()} (confidence: {confidence.item():.2%})")
            
            return predicted.item(), confidence.item(), probabilities.cpu().numpy()[0]
    
    def fgsm_attack(self, image_tensor, target_label=None, epsilon=0.2):
        """FGSM attack - 修复设备不匹配问题"""
        image = image_tensor.clone().detach().to(self.device)
        image.requires_grad = True
        
        # Original prediction
        with torch.no_grad():
            orig_outputs = self.model(image)
            orig_pred = torch.argmax(orig_outputs, 1).item()
        
        print(f"🎯 Original prediction: {orig_pred}")
        
        # Compute gradient
        outputs = self.model(image)
        
        if target_label is not None:
            # Targeted attack
            target = torch.tensor([target_label], dtype=torch.long).to(self.device)
            loss = F.cross_entropy(outputs, target)
            print(f"🎯 Targeted attack: trying to change prediction to {target_label}")
        else:
            # Untargeted attack
            orig_label = torch.tensor([orig_pred], dtype=torch.long).to(self.device)
            loss = -F.cross_entropy(outputs, orig_label)
            print(f"🎯 Untargeted attack: trying to change prediction from {orig_pred}")
        
        self.model.zero_grad()
        loss.backward()
        
        # Generate adversarial example
        grad = image.grad.data
        grad_norm = grad.abs().max().item()
        print(f"📈 Gradient norm: {grad_norm:.4f}")
        
        if target_label is not None:
            adv_image = image - epsilon * grad.sign()
        else:
            adv_image = image + epsilon * grad.sign()
        
        # 修复：确保所有张量在同一个设备上
        min_val = image_tensor.min().item() if image_tensor.is_cuda else image_tensor.min()
        max_val = image_tensor.max().item() if image_tensor.is_cuda else image_tensor.max()
        
        # 使用数值而不是张量进行clamp
        adv_image = torch.clamp(adv_image, min_val, max_val).detach()
        
        return adv_image, orig_pred
    
    def i_fgsm_attack(self, image_tensor, target_label=None, epsilon=0.2, alpha=0.01, num_iter=10):
        """I-FGSM attack - 修复设备不匹配问题"""
        adv_image = image_tensor.clone().detach().to(self.device)
        orig_image = image_tensor.clone().detach().to(self.device)
        
        # Original prediction
        with torch.no_grad():
            orig_outputs = self.model(adv_image)
            orig_pred = torch.argmax(orig_outputs, 1).item()
        
        print(f"🎯 Original prediction: {orig_pred}")
        print(f"🔄 Starting I-FGSM attack with {num_iter} iterations...")
        
        for i in range(num_iter):
            adv_image.requires_grad = True
            outputs = self.model(adv_image)
            
            if target_label is not None:
                target = torch.tensor([target_label], dtype=torch.long).to(self.device)
                loss = F.cross_entropy(outputs, target)
            else:
                orig_label = torch.tensor([orig_pred], dtype=torch.long).to(self.device)
                loss = -F.cross_entropy(outputs, orig_label)
            
            self.model.zero_grad()
            loss.backward()
            
            grad = adv_image.grad.data
            if target_label is not None:
                adv_image = adv_image - alpha * grad.sign()
            else:
                adv_image = adv_image + alpha * grad.sign()
            
            # Limit perturbation range
            delta = torch.clamp(adv_image - orig_image, min=-epsilon, max=epsilon)
            adv_image = torch.clamp(orig_image + delta, 0, 1).detach()
            
            # Check current prediction
            if (i+1) % 2 == 0 or i == num_iter-1:
                with torch.no_grad():
                    current_pred = torch.argmax(self.model(adv_image), 1).item()
                print(f"  Iteration {i+1}/{num_iter}: Prediction = {current_pred}")
                
                # Early stop if attack succeeds
                if target_label is not None:
                    if current_pred == target_label:
                        print(f"  ✅ Target achieved at iteration {i+1}")
                        break
                else:
                    if current_pred != orig_pred:
                        print(f"  ✅ Attack succeeded at iteration {i+1}")
                        break
        
        return adv_image, orig_pred
    
    def visualize_attack(self, original_image, original_tensor, adv_tensor, 
                         orig_pred, adv_pred, orig_conf, adv_conf, 
                         attack_type="FGSM", epsilon=0.2, save_path=None):
        """Visualize attack results"""
        
        # Convert to numpy for display
        orig_np = original_tensor.squeeze().cpu().numpy()
        adv_np = adv_tensor.squeeze().cpu().numpy()
        perturbation = np.abs(adv_np - orig_np) * 10
        
        # Get all class probabilities
        with torch.no_grad():
            orig_probs = F.softmax(self.model(original_tensor.to(self.device)), dim=1).cpu().numpy()[0]
            adv_probs = F.softmax(self.model(adv_tensor.to(self.device)), dim=1).cpu().numpy()[0]
        
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        
        # Set English font
        plt.rcParams.update({
            'font.size': 10,
            'font.family': 'DejaVu Sans',
            'axes.unicode_minus': False
        })
        
        # 1. Original Image
        ax1 = plt.subplot(3, 4, 1)
        ax1.imshow(original_image, cmap='gray')
        ax1.set_title(f'Original Image\nPrediction: {orig_pred} (Confidence: {orig_conf:.2%})')
        ax1.axis('off')
        
        # 2. Adversarial Image
        ax2 = plt.subplot(3, 4, 2)
        ax2.imshow(adv_np, cmap='gray')
        ax2.set_title(f'Adversarial Image\nPrediction: {adv_pred} (Confidence: {adv_conf:.2%})')
        ax2.axis('off')
        
        # 3. Perturbation Visualization
        ax3 = plt.subplot(3, 4, 3)
        im3 = ax3.imshow(perturbation, cmap='hot')
        ax3.set_title(f'Perturbation (x10 amplified)\nε={epsilon}')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # 4. Pixel Value Histogram
        ax4 = plt.subplot(3, 4, 4)
        ax4.hist(orig_np.flatten(), bins=20, alpha=0.7, label='Original', color='blue')
        ax4.hist(adv_np.flatten(), bins=20, alpha=0.7, label='Adversarial', color='red')
        ax4.set_title('Pixel Value Distribution')
        ax4.set_xlabel('Pixel Value')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        
        # 5. Confidence Bar Chart (Original)
        ax5 = plt.subplot(3, 4, 5)
        bars1 = ax5.bar(range(10), orig_probs * 100, color='skyblue')
        ax5.set_title('Original Image Confidence')
        ax5.set_xlabel('Digit Class')
        ax5.set_ylabel('Confidence (%)')
        ax5.set_ylim([0, 100])
        ax5.set_xticks(range(10))
        bars1[orig_pred].set_color('red')
        
        # 6. Confidence Bar Chart (Adversarial)
        ax6 = plt.subplot(3, 4, 6)
        bars2 = ax6.bar(range(10), adv_probs * 100, color='lightcoral')
        ax6.set_title('Adversarial Image Confidence')
        ax6.set_xlabel('Digit Class')
        ax6.set_ylabel('Confidence (%)')
        ax6.set_ylim([0, 100])
        ax6.set_xticks(range(10))
        if adv_pred < 10:
            bars2[adv_pred].set_color('red')
        
        # 7. Confidence Change
        ax7 = plt.subplot(3, 4, 7)
        confidence_change = adv_probs - orig_probs
        colors = ['green' if x < 0 else 'red' for x in confidence_change]
        ax7.bar(range(10), confidence_change * 100, color=colors)
        ax7.set_title('Confidence Change (Adversarial - Original)')
        ax7.set_xlabel('Digit Class')
        ax7.set_ylabel('Change (%)')
        ax7.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 8. Pixel Value Scatter
        ax8 = plt.subplot(3, 4, 8)
        ax8.scatter(orig_np.flatten(), adv_np.flatten(), alpha=0.1, s=1, color='purple')
        ax8.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        ax8.set_title('Pixel Value Scatter Plot')
        ax8.set_xlabel('Original Pixel Value')
        ax8.set_ylabel('Adversarial Pixel Value')
        
        # 9. Attack Info Text
        ax9 = plt.subplot(3, 4, (9, 12))
        ax9.axis('off')
        
        # Calculate statistics
        perturbation_actual = adv_np - orig_np
        mean_pert = np.mean(np.abs(perturbation_actual))
        max_pert = np.max(np.abs(perturbation_actual))
        l_inf_norm = np.max(np.abs(perturbation_actual))
        
        info_text = f"""
        ====== Adversarial Attack Analysis Report ======
        
        [Attack Information]
        - Attack Type: {attack_type}
        - Perturbation Strength (ε): {epsilon}
        - Attack Success: {'✅ YES' if orig_pred != adv_pred else '❌ NO'}
        
        [Prediction Results]
        - Original Prediction: Digit {orig_pred} (Confidence: {orig_conf:.2%})
        - Adversarial Prediction: Digit {adv_pred} (Confidence: {adv_conf:.2%})
        - Prediction Change: {abs(orig_pred - adv_pred)} classes
        
        [Perturbation Statistics]
        - Mean Perturbation: {mean_pert:.4f}
        - Max Perturbation: {max_pert:.4f}
        - L∞ Norm: {l_inf_norm:.4f}
        
        [Conclusion]
        {'✅ Model successfully deceived! Digit ' + str(orig_pred) + ' → ' + str(adv_pred) 
         if orig_pred != adv_pred else '✅ Model successfully defended the attack!'}
        """
        
        ax9.text(0, 1, info_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'Adversarial Attack Analysis: {attack_type} (ε={epsilon})', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save or show
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return fig

def main():
    """Main function: Interactive single image attack"""
    print("="*60)
    print("🤖 Single Image Adversarial Attack Demo System")
    print("="*60)
    
    # Configuration
    MODEL_PATH = "./models/MNIST_robust.pth"
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("Please run robust_trainer.py first to train the model:")
        print("  python robust_trainer.py")
        return
    
    # Initialize attacker
    print(f"\n🔧 Initializing attack system...")
    attacker = SingleImageAttacker(MODEL_PATH)
    print(f"  Device: {attacker.device}")
    
    # Interactive input
    print("\n📁 [File Selection]")
    print("1. Use MNIST test image")
    print("2. Use custom image file")
    
    choice = input("Enter choice (1/2): ").strip()
    
    if choice == '1':
        # Use MNIST test image
        image_tensor, original_image = attacker.load_test_image("test")
        
    elif choice == '2':
        # Use custom image
        print("\n📸 [Custom Image Selection]")
        print("Tip: For best results, use:")
        print("  - Clear digit image (black on white or white on black)")
        print("  - Preferably 28x28 pixels or close")
        print("  - Well-centered digit")
        
        image_path = input("Enter image path: ").strip()
        image_tensor, original_image = attacker.load_user_image(image_path)
        
    else:
        print("❌ Invalid choice")
        return
    
    if image_tensor is None:
        print("❌ Failed to load image")
        return
    
    # 确保图像张量在正确的设备上
    image_tensor = image_tensor.to(attacker.device)
    
    # Original prediction
    print("\n🔍 Making original prediction...")
    orig_pred, orig_conf, orig_probs = attacker.predict(image_tensor, verbose=True)
    
    # Select attack type
    print("\n⚔️ [Attack Type Selection]")
    print("1. FGSM (Fast Gradient Sign Method) - Single step")
    print("2. I-FGSM (Iterative FGSM) - Multiple steps")
    print("3. Targeted Attack - Force specific prediction")
    
    attack_choice = input("Enter choice (1/2/3): ").strip()
    
    # Select perturbation strength
    epsilon = float(input("Enter perturbation strength ε (0.1-0.3 recommended): ").strip())
    
    # Execute attack
    print(f"\n🚀 Executing attack with ε={epsilon}...")
    
    if attack_choice == '1':
        attack_type = "FGSM"
        adv_tensor, _ = attacker.fgsm_attack(image_tensor, epsilon=epsilon)
        
    elif attack_choice == '2':
        attack_type = "I-FGSM"
        num_iter = int(input("Enter iteration number (5-20 recommended): ").strip())
        adv_tensor, _ = attacker.i_fgsm_attack(image_tensor, epsilon=epsilon, num_iter=num_iter)
        
    elif attack_choice == '3':
        attack_type = "Targeted FGSM"
        target_label = int(input("Enter target digit class (0-9): ").strip())
        adv_tensor, _ = attacker.fgsm_attack(image_tensor, target_label=target_label, epsilon=epsilon)
    else:
        print("❌ Invalid choice")
        return
    
    # Adversarial prediction
    print("\n🔍 Making adversarial prediction...")
    adv_pred, adv_conf, adv_probs = attacker.predict(adv_tensor, verbose=True)
    
    print(f"\n📊 [Attack Results Summary]")
    print(f"  Original: Digit {orig_pred} (confidence: {orig_conf:.2%})")
    print(f"  Adversarial: Digit {adv_pred} (confidence: {adv_conf:.2%})")
    print(f"  Attack Success: {'✅ YES' if orig_pred != adv_pred else '❌ NO'}")
    
    # Generate visualization
    save_dir = "./single_image_results"
    timestamp = f"attack_{orig_pred}to{adv_pred}_{attack_type}"
    save_path = f"{save_dir}/{timestamp}.png"
    
    print(f"\n🎨 Generating visualization...")
    attacker.visualize_attack(
        original_image=original_image,
        original_tensor=image_tensor,
        adv_tensor=adv_tensor,
        orig_pred=orig_pred,
        adv_pred=adv_pred,
        orig_conf=orig_conf,
        adv_conf=adv_conf,
        attack_type=attack_type,
        epsilon=epsilon,
        save_path=save_path
    )
    
    print(f"\n✅ Attack completed!")
    print(f"📁 Analysis saved to: {save_path}")
    print("="*60)

if __name__ == "__main__":
    main()