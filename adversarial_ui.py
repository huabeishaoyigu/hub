"""
adversarial_ui.py
对抗攻击可视化图形界面
基于PyQt5实现
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# PyQt5 导入
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# 导入原有的攻击套件
from adversarial_suite import AdversarialAttacker

# ==========================================
# 模型定义 (与训练时一致)
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
# Matplotlib画布封装
# ==========================================
class MplCanvas(FigureCanvas):
    """Matplotlib画布控件"""
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.axes = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

# ==========================================
# 工作线程 - 用于执行批量测试避免界面卡顿
# ==========================================
class BatchTestWorker(QThread):
    """批量测试工作线程"""
    progress_update = pyqtSignal(int, str)  # 进度更新信号 (当前批次, 状态信息)
    result_ready = pyqtSignal(dict)         # 测试结果信号
    finished = pyqtSignal()                  # 完成信号
    
    def __init__(self, model, device, test_loader, epsilon, attack_type):
        super().__init__()
        self.model = model
        self.device = device
        self.test_loader = test_loader
        self.epsilon = epsilon
        self.attack_type = attack_type
        self.is_running = True
        
    def run(self):
        """执行批量测试"""
        try:
            config = {'epsilon': self.epsilon, 'alpha': 0.01, 'num_iter': 10}
            attacker = AdversarialAttacker(self.model, self.device, config=config)
            
            success_count = 0
            total_count = 0
            l_inf_dist = []
            
            # 先计算原始准确率
            self.progress_update.emit(0, "计算原始准确率...")
            original_correct = 0
            original_total = 0
            for images, labels in self.test_loader:
                if not self.is_running:
                    return
                images, labels = images.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    outputs = self.model(images)
                    preds = outputs.max(1)[1]
                    original_correct += preds.eq(labels).sum().item()
                    original_total += labels.size(0)
            
            # 执行攻击评估
            batch_count = 0
            for images, labels in self.test_loader:
                if not self.is_running:
                    return
                    
                images, labels = images.to(self.device), labels.to(self.device)
                
                with torch.no_grad():
                    orig_outputs = self.model(images)
                    orig_preds = orig_outputs.max(1)[1]
                    correct_mask = orig_preds.eq(labels)
                
                correct_indices = correct_mask.nonzero(as_tuple=True)[0]
                if len(correct_indices) == 0:
                    continue
                    
                correct_images = images[correct_indices]
                correct_labels = labels[correct_indices]
                
                # 生成攻击样本
                if self.attack_type == 'fgsm':
                    adv_images = attacker.fgsm_attack(correct_images, correct_labels, epsilon=self.epsilon)
                else:
                    adv_images = attacker.i_fgsm_attack(correct_images, correct_labels, epsilon=self.epsilon)

                with torch.no_grad():
                    outputs = self.model(adv_images)
                    preds = outputs.max(1)[1]
                    
                    attack_success_mask = preds != correct_labels
                    success_count += attack_success_mask.sum().item()
                    total_count += correct_labels.size(0)
                    
                    dist = torch.norm((adv_images - correct_images).view(correct_images.shape[0], -1), 
                                     p=float('inf'), dim=1)
                    l_inf_dist.extend(dist.cpu().numpy())
                
                batch_count += 1
                self.progress_update.emit(batch_count, f"已处理 {batch_count} 批次...")
            
            # 准备结果
            attack_success_rate = success_count / total_count if total_count > 0 else 0
            avg_l_inf = np.mean(l_inf_dist) if l_inf_dist else 0
            
            result = {
                'original_accuracy': original_correct / original_total,
                'attack_success_rate': attack_success_rate,
                'avg_l_inf': avg_l_inf,
                'robust_accuracy': 1 - attack_success_rate,
                'total_samples': total_count
            }
            
            self.result_ready.emit(result)
            
        except Exception as e:
            self.progress_update.emit(-1, f"错误: {str(e)}")
        finally:
            self.finished.emit()
    
    def stop(self):
        """停止线程"""
        self.is_running = False

# ==========================================
# 主窗口类
# ==========================================
class AdversarialAttackUI(QMainWindow):
    """对抗攻击可视化主窗口"""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_dataset = None
        self.test_loader = None
        self.current_image = None
        self.current_label = None
        self.current_tensor = None
        self.adv_tensor = None
        self.original_image_pil = None
        self.batch_thread = None
        
        self.init_ui()
        self.load_model()
        self.load_test_data()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("对抗攻击可视化系统 - Adversarial Attack UI")
        self.setGeometry(100, 100, 1400, 900)
        
        # 设置全局样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #d3d3d3;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QTextEdit, QPlainTextEdit {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 4px;
                font-family: Consolas, monospace;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                border-radius: 5px;
            }
        """)
        
        # 创建中央控件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # ===== 顶部信息栏 =====
        info_layout = QHBoxLayout()
        
        self.device_label = QLabel(f"设备: {self.device}")
        self.device_label.setStyleSheet("background-color: #e0e0e0; padding: 5px; border-radius: 3px;")
        
        self.model_status = QLabel("模型状态: 未加载")
        self.model_status.setStyleSheet("background-color: #ffcccb; padding: 5px; border-radius: 3px;")
        
        info_layout.addWidget(self.device_label)
        info_layout.addWidget(self.model_status)
        info_layout.addStretch()
        
        main_layout.addLayout(info_layout)
        
        # ===== 主内容区域 (使用分割器) =====
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧控制面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        # 图像选择组
        image_group = QGroupBox("📁 图像选择")
        image_layout = QVBoxLayout()
        
        self.image_source_combo = QComboBox()
        self.image_source_combo.addItems(["MNIST测试集", "自定义图片"])
        self.image_source_combo.currentIndexChanged.connect(self.on_image_source_changed)
        
        self.digit_spin = QSpinBox()
        self.digit_spin.setRange(0, 9)
        self.digit_spin.setValue(5)
        self.digit_spin.setPrefix("数字: ")
        self.digit_spin.valueChanged.connect(self.load_mnist_image)
        
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setPlaceholderText("点击浏览选择图片...")
        self.image_path_edit.setEnabled(False)
        
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.setEnabled(False)
        self.browse_btn.clicked.connect(self.browse_image)
        
        path_layout = QHBoxLayout()
        path_layout.addWidget(self.image_path_edit)
        path_layout.addWidget(self.browse_btn)
        
        self.load_btn = QPushButton("加载图像")
        self.load_btn.clicked.connect(self.load_selected_image)
        
        image_layout.addWidget(self.image_source_combo)
        image_layout.addWidget(self.digit_spin)
        image_layout.addLayout(path_layout)
        image_layout.addWidget(self.load_btn)
        image_group.setLayout(image_layout)
        
        # 攻击参数组
        attack_group = QGroupBox("⚔️ 攻击参数")
        attack_layout = QGridLayout()
        
        attack_layout.addWidget(QLabel("攻击类型:"), 0, 0)
        self.attack_type_combo = QComboBox()
        self.attack_type_combo.addItems(["FGSM (单步)", "I-FGSM (迭代)", "定向攻击"])
        attack_layout.addWidget(self.attack_type_combo, 0, 1)
        
        attack_layout.addWidget(QLabel("目标类别:"), 1, 0)
        self.target_spin = QSpinBox()
        self.target_spin.setRange(0, 9)
        self.target_spin.setValue(0)
        self.target_spin.setEnabled(False)
        attack_layout.addWidget(self.target_spin, 1, 1)
        
        attack_layout.addWidget(QLabel("迭代次数:"), 2, 0)
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(1, 50)
        self.iter_spin.setValue(10)
        attack_layout.addWidget(self.iter_spin, 2, 1)
        
        attack_layout.addWidget(QLabel("ε (扰动强度):"), 3, 0)
        self.epsilon_slider = QSlider(Qt.Horizontal)
        self.epsilon_slider.setRange(1, 50)
        self.epsilon_slider.setValue(20)
        self.epsilon_slider.setTickPosition(QSlider.TicksBelow)
        self.epsilon_slider.setTickInterval(5)
        self.epsilon_slider.valueChanged.connect(self.on_epsilon_changed)
        
        self.epsilon_spin = QDoubleSpinBox()
        self.epsilon_spin.setRange(0.01, 1.0)
        self.epsilon_spin.setValue(0.2)
        self.epsilon_spin.setSingleStep(0.01)
        self.epsilon_spin.valueChanged.connect(self.on_epsilon_spin_changed)
        
        epsilon_layout = QHBoxLayout()
        epsilon_layout.addWidget(self.epsilon_slider, 3)
        epsilon_layout.addWidget(self.epsilon_spin)
        
        attack_layout.addLayout(epsilon_layout, 3, 1)
        
        self.attack_btn = QPushButton("🚀 执行攻击")
        self.attack_btn.clicked.connect(self.execute_attack)
        attack_layout.addWidget(self.attack_btn, 4, 0, 1, 2)
        
        attack_group.setLayout(attack_layout)
        
        # 批量测试组
        batch_group = QGroupBox("📊 批量测试")
        batch_layout = QGridLayout()
        
        batch_layout.addWidget(QLabel("攻击类型:"), 0, 0)
        self.batch_type_combo = QComboBox()
        self.batch_type_combo.addItems(["FGSM", "I-FGSM"])
        batch_layout.addWidget(self.batch_type_combo, 0, 1)
        
        batch_layout.addWidget(QLabel("ε:"), 1, 0)
        self.batch_epsilon_spin = QDoubleSpinBox()
        self.batch_epsilon_spin.setRange(0.01, 1.0)
        self.batch_epsilon_spin.setValue(0.2)
        self.batch_epsilon_spin.setSingleStep(0.01)
        batch_layout.addWidget(self.batch_epsilon_spin, 1, 1)
        
        self.batch_test_btn = QPushButton("📈 执行批量测试")
        self.batch_test_btn.clicked.connect(self.run_batch_test)
        batch_layout.addWidget(self.batch_test_btn, 2, 0, 1, 2)
        
        self.batch_progress = QProgressBar()
        self.batch_progress.setVisible(False)
        batch_layout.addWidget(self.batch_progress, 3, 0, 1, 2)
        
        batch_group.setLayout(batch_layout)
        
        # 添加所有组到左侧面板
        left_layout.addWidget(image_group)
        left_layout.addWidget(attack_group)
        left_layout.addWidget(batch_group)
        left_layout.addStretch()
        
        # 中间可视化面板
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        
        # 图像显示区域
        viz_group = QGroupBox("🖼️ 图像对比")
        viz_layout = QHBoxLayout()
        
        # 原图显示
        self.original_canvas = MplCanvas(self, width=4, height=4, dpi=80)
        self.original_canvas.axes.set_title("原始图像", fontsize=12, fontweight='bold')
        self.original_canvas.axes.axis('off')
        
        # 对抗图显示
        self.adv_canvas = MplCanvas(self, width=4, height=4, dpi=80)
        self.adv_canvas.axes.set_title("对抗图像", fontsize=12, fontweight='bold')
        self.adv_canvas.axes.axis('off')
        
        # 扰动显示
        self.pert_canvas = MplCanvas(self, width=4, height=4, dpi=80)
        self.pert_canvas.axes.set_title("扰动 (放大10倍)", fontsize=12, fontweight='bold')
        self.pert_canvas.axes.axis('off')
        
        viz_layout.addWidget(self.original_canvas)
        viz_layout.addWidget(self.adv_canvas)
        viz_layout.addWidget(self.pert_canvas)
        viz_group.setLayout(viz_layout)
        
        # 预测结果和置信度
        result_group = QGroupBox("📋 预测结果")
        result_layout = QHBoxLayout()
        
        self.orig_result = QTextEdit()
        self.orig_result.setMaximumHeight(80)
        self.orig_result.setReadOnly(True)
        self.orig_result.setPlaceholderText("原始预测结果")
        
        self.adv_result = QTextEdit()
        self.adv_result.setMaximumHeight(80)
        self.adv_result.setReadOnly(True)
        self.adv_result.setPlaceholderText("对抗样本预测结果")
        
        result_layout.addWidget(self.orig_result)
        result_layout.addWidget(self.adv_result)
        result_group.setLayout(result_layout)
        
        center_layout.addWidget(viz_group)
        center_layout.addWidget(result_group)
        
        # 右侧详细分析面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 攻击详情
        detail_group = QGroupBox("📊 攻击详情")
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        detail_layout = QVBoxLayout()
        detail_layout.addWidget(self.detail_text)
        detail_group.setLayout(detail_layout)
        
        # 批量测试结果
        batch_result_group = QGroupBox("📈 批量测试结果")
        self.batch_result_text = QTextEdit()
        self.batch_result_text.setReadOnly(True)
        batch_result_layout = QVBoxLayout()
        batch_result_layout.addWidget(self.batch_result_text)
        batch_result_group.setLayout(batch_result_layout)
        
        right_layout.addWidget(detail_group)
        right_layout.addWidget(batch_result_group)
        
        # 将所有面板添加到分割器
        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 700, 350])
        
        main_layout.addWidget(splitter)
        
        # 底部状态栏
        self.statusBar().showMessage("就绪")
        
        # 连接信号
        self.attack_type_combo.currentIndexChanged.connect(self.on_attack_type_changed)
        
    def on_attack_type_changed(self, index):
        """攻击类型改变时的处理"""
        self.target_spin.setEnabled(index == 2)  # 定向攻击时启用目标选择
        self.iter_spin.setEnabled(index == 1)    # I-FGSM时启用迭代次数
        
    def on_epsilon_changed(self, value):
        """滑块值改变"""
        eps = value / 100.0
        self.epsilon_spin.setValue(eps)
        
    def on_epsilon_spin_changed(self, value):
        """数字输入框值改变"""
        self.epsilon_slider.setValue(int(value * 100))
        
    def on_image_source_changed(self, index):
        """图像源改变"""
        is_mnist = index == 0
        self.digit_spin.setEnabled(is_mnist)
        self.image_path_edit.setEnabled(not is_mnist)
        self.browse_btn.setEnabled(not is_mnist)
        
    def load_model(self):
        """加载训练好的模型"""
        model_path = "./models/MNIST_robust.pth"
        try:
            self.model = get_mnist_model().to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.model_status.setText("模型状态: ✅ 已加载 (MNIST_robust)")
            self.model_status.setStyleSheet("background-color: #c1e1c1; padding: 5px; border-radius: 3px;")
            self.statusBar().showMessage("模型加载成功")
        except Exception as e:
            self.model_status.setText(f"模型状态: ❌ 加载失败")
            self.model_status.setStyleSheet("background-color: #ffcccb; padding: 5px; border-radius: 3px;")
            QMessageBox.warning(self, "警告", f"模型加载失败: {str(e)}\n请先运行 robust_trainer.py 训练模型。")
            
    def load_test_data(self):
        """加载测试数据"""
        try:
            self.test_dataset = datasets.MNIST(
                './data', 
                train=False, 
                download=True, 
                transform=transforms.ToTensor()
            )
            self.test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=True)
            self.statusBar().showMessage("测试数据加载成功")
        except Exception as e:
            QMessageBox.warning(self, "警告", f"测试数据加载失败: {str(e)}")
            
    def browse_image(self):
        """浏览图片文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.image_path_edit.setText(file_path)
            
    def load_selected_image(self):
        """加载选中的图像"""
        if self.image_source_combo.currentIndex() == 0:
            self.load_mnist_image()
        else:
            self.load_custom_image()
            
    def load_mnist_image(self):
        """加载MNIST测试集图像"""
        if self.test_dataset is None:
            return
            
        digit = self.digit_spin.value()
        indices = [i for i, (_, label) in enumerate(self.test_dataset) if label == digit]
        
        if not indices:
            QMessageBox.warning(self, "警告", f"未找到数字 {digit} 的图片")
            return
            
        idx = indices[0]
        image_tensor, label = self.test_dataset[idx]
        
        # 保存原始数据
        self.current_tensor = image_tensor.unsqueeze(0).to(self.device)
        self.current_label = label
        self.original_image_pil = Image.fromarray(
            (image_tensor.squeeze().numpy() * 255).astype(np.uint8)
        )
        
        # 显示原图
        self.display_image(self.original_canvas, self.current_tensor.cpu(), "原始图像")
        
        # 预测
        self.predict_original()
        
        self.statusBar().showMessage(f"已加载数字 {digit} 的图片")
        
    def load_custom_image(self):
        """加载自定义图片"""
        image_path = self.image_path_edit.text()
        if not image_path or not os.path.exists(image_path):
            QMessageBox.warning(self, "警告", "请选择有效的图片文件")
            return
            
        try:
            # 预处理自定义图片
            image = Image.open(image_path)
            
            # 转换为灰度图并调整大小
            if image.mode != 'L':
                image = image.convert('L')
            image = ImageOps.invert(image)  # 反转颜色 (假设是白底黑字)
            image = image.resize((28, 28), Image.Resampling.LANCZOS)
            
            # 转换为tensor
            transform = transforms.ToTensor()
            image_tensor = transform(image)
            
            # 二值化处理
            image_tensor = (image_tensor > 0.3).float()
            
            self.current_tensor = image_tensor.unsqueeze(0).to(self.device)
            self.original_image_pil = image
            
            # 显示原图
            self.display_image(self.original_canvas, self.current_tensor.cpu(), "原始图像")
            
            # 预测
            self.predict_original()
            
            self.statusBar().showMessage(f"已加载图片: {os.path.basename(image_path)}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"图片加载失败: {str(e)}")
            
    def predict_original(self):
        """预测原始图像"""
        if self.model is None or self.current_tensor is None:
            return
            
        with torch.no_grad():
            outputs = self.model(self.current_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            pred = predicted.item()
            conf = confidence.item()
            
            # 显示结果
            result_text = f"🎯 预测: {pred}\n"
            result_text += f"📊 置信度: {conf:.2%}\n"
            result_text += "-" * 20 + "\n"
            
            # 显示前3个高概率类别
            probs = probabilities.cpu().numpy()[0]
            top3_idx = np.argsort(probs)[-3:][::-1]
            for idx in top3_idx:
                result_text += f"类别 {idx}: {probs[idx]:.2%}\n"
                
            self.orig_result.setText(result_text)
            
    def execute_attack(self):
        """执行攻击"""
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
            
        if self.current_tensor is None:
            QMessageBox.warning(self, "警告", "请先加载图像")
            return
            
        attack_type = self.attack_type_combo.currentIndex()
        epsilon = self.epsilon_spin.value()
        target = self.target_spin.value() if attack_type == 2 else None
        
        self.statusBar().showMessage("正在执行攻击...")
        
        try:
            # 创建攻击器
            config = {'epsilon': epsilon, 'alpha': 0.01, 'num_iter': self.iter_spin.value()}
            attacker = AdversarialAttacker(self.model, self.device, config=config)
            
            # 执行攻击
            if attack_type == 0:  # FGSM
                self.adv_tensor = attacker.fgsm_attack(
                    self.current_tensor, 
                    torch.tensor([self.current_label]).to(self.device) if self.current_label is not None else None,
                    epsilon=epsilon,
                    target_labels=torch.tensor([target]).to(self.device) if target is not None else None
                )
                attack_name = "FGSM"
            elif attack_type == 1:  # I-FGSM
                self.adv_tensor = attacker.i_fgsm_attack(
                    self.current_tensor,
                    torch.tensor([self.current_label]).to(self.device) if self.current_label is not None else None,
                    epsilon=epsilon,
                    alpha=0.01,
                    num_iter=self.iter_spin.value()
                )
                attack_name = "I-FGSM"
            else:  # 定向攻击
                self.adv_tensor = attacker.fgsm_attack(
                    self.current_tensor,
                    torch.tensor([self.current_label]).to(self.device) if self.current_label is not None else None,
                    epsilon=epsilon,
                    target_labels=torch.tensor([target]).to(self.device)
                )
                attack_name = f"定向攻击(目标={target})"
            
            # 预测对抗样本
            with torch.no_grad():
                outputs = self.model(self.adv_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                adv_pred = predicted.item()
                adv_conf = confidence.item()
                
                # 显示对抗样本结果
                adv_result_text = f"🎯 预测: {adv_pred}\n"
                adv_result_text += f"📊 置信度: {adv_conf:.2%}\n"
                adv_result_text += "-" * 20 + "\n"
                
                probs = probabilities.cpu().numpy()[0]
                top3_idx = np.argsort(probs)[-3:][::-1]
                for idx in top3_idx:
                    adv_result_text += f"类别 {idx}: {probs[idx]:.2%}\n"
                    
                self.adv_result.setText(adv_result_text)
            
            # 显示对抗图像和扰动
            self.display_image(self.adv_canvas, self.adv_tensor.cpu(), "对抗图像")
            self.display_perturbation()
            
            # 更新攻击详情
            self.update_attack_details(attack_name, epsilon, adv_pred, adv_conf)
            
            self.statusBar().showMessage("攻击完成")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"攻击执行失败: {str(e)}")
            self.statusBar().showMessage("攻击失败")
            
    def display_image(self, canvas, tensor, title):
        """在画布上显示图像"""
        canvas.axes.clear()
        img_np = tensor.squeeze().cpu().numpy()
        canvas.axes.imshow(img_np, cmap='gray')
        canvas.axes.set_title(title, fontsize=12, fontweight='bold')
        canvas.axes.axis('off')
        canvas.draw()
        
    def display_perturbation(self):
        """显示扰动图像"""
        if self.current_tensor is None or self.adv_tensor is None:
            return
            
        orig_np = self.current_tensor.squeeze().cpu().numpy()
        adv_np = self.adv_tensor.squeeze().cpu().numpy()
        perturbation = np.abs(adv_np - orig_np) * 10  # 放大10倍
        
        self.pert_canvas.axes.clear()
        im = self.pert_canvas.axes.imshow(perturbation, cmap='hot')
        self.pert_canvas.axes.set_title("扰动 (放大10倍)", fontsize=12, fontweight='bold')
        self.pert_canvas.axes.axis('off')
        self.pert_canvas.fig.colorbar(im, ax=self.pert_canvas.axes, fraction=0.046, pad=0.04)
        self.pert_canvas.draw()
        
    def update_attack_details(self, attack_name, epsilon, adv_pred, adv_conf):
        """更新攻击详情"""
        if self.current_label is None:
            return
            
        orig_np = self.current_tensor.squeeze().cpu().numpy()
        adv_np = self.adv_tensor.squeeze().cpu().numpy()
        
        mean_pert = np.mean(np.abs(adv_np - orig_np))
        max_pert = np.max(np.abs(adv_np - orig_np))
        l_inf_norm = max_pert
        
        attack_success = self.current_label != adv_pred
        
        detail_text = f"""
╔════════════════════════════════════╗
║        攻击分析报告                 ║
╚════════════════════════════════════╝

[攻击信息]
• 攻击类型: {attack_name}
• 扰动强度 ε: {epsilon}
• 攻击成功: {'✅ 是' if attack_success else '❌ 否'}

[预测结果]
• 原始图像: 数字 {self.current_label}
• 对抗图像: 数字 {adv_pred} (置信度: {adv_conf:.2%})
• 预测变化: {'改变' if attack_success else '不变'}

[扰动统计]
• 平均扰动: {mean_pert:.4f}
• 最大扰动: {max_pert:.4f}
• L∞ 范数: {l_inf_norm:.4f}

[结论]
{('✅ 模型被成功欺骗！' + str(self.current_label) + ' → ' + str(adv_pred)) 
 if attack_success else '✅ 模型成功防御了此次攻击！'}
"""
        self.detail_text.setText(detail_text)
        
    def run_batch_test(self):
        """运行批量测试"""
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
            
        if self.test_loader is None:
            QMessageBox.warning(self, "警告", "测试数据未加载")
            return
            
        # 禁用按钮
        self.batch_test_btn.setEnabled(False)
        self.batch_progress.setVisible(True)
        self.batch_progress.setRange(0, 0)  # 不确定进度
        
        # 创建并启动工作线程
        attack_type = self.batch_type_combo.currentText().lower()
        self.batch_thread = BatchTestWorker(
            self.model, 
            self.device, 
            self.test_loader,
            self.batch_epsilon_spin.value(),
            attack_type
        )
        
        self.batch_thread.progress_update.connect(self.on_batch_progress)
        self.batch_thread.result_ready.connect(self.on_batch_result)
        self.batch_thread.finished.connect(self.on_batch_finished)
        
        self.batch_thread.start()
        
        self.statusBar().showMessage("正在执行批量测试...")
        
    def on_batch_progress(self, batch, message):
        """批量测试进度更新"""
        if batch > 0:
            self.batch_progress.setRange(0, 100)
            progress = min(100, int(batch * 100 / 10))  # 假设约10个批次
            self.batch_progress.setValue(progress)
        self.statusBar().showMessage(message)
        
    def on_batch_result(self, result):
        """批量测试结果处理"""
        text = f"""
╔════════════════════════════════════╗
║        批量测试结果                 ║
╚════════════════════════════════════╝

[评估指标]
• 原始准确率: {result['original_accuracy']:.2%}
• 攻击成功率: {result['attack_success_rate']:.2%}
• 鲁棒准确率: {result['robust_accuracy']:.2%}

[扰动分析]
• 平均 L∞ 距离: {result['avg_l_inf']:.4f}
• 测试样本数: {result['total_samples']}

[结论]
攻击类型: {self.batch_type_combo.currentText()}
扰动强度 ε: {self.batch_epsilon_spin.value()}
"""
        self.batch_result_text.setText(text)
        
    def on_batch_finished(self):
        """批量测试完成"""
        self.batch_test_btn.setEnabled(True)
        self.batch_progress.setVisible(False)
        self.statusBar().showMessage("批量测试完成")
        
    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.batch_thread and self.batch_thread.isRunning():
            self.batch_thread.stop()
            self.batch_thread.wait()
        event.accept()


# ==========================================
# 应用程序入口
# ==========================================
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # 设置应用程序图标
    app.setWindowIcon(QIcon())
    
    window = AdversarialAttackUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()