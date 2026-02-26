"""
直接运行.py
最简单的启动脚本，直接运行UI
"""

import sys
import os

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


print("="*50)
print("直接启动对抗攻击可视化系统")
print("="*50)

try:
    # 直接导入并运行
    from adversarial_ui import main
    print("✅ 成功导入UI模块")
    print("🚀 启动中...\n")
    main()
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("\n可能的原因：")
    print("1. 缺少PyQt5: pip install PyQt5")
    print("2. 文件命名问题：确保 adversarial_ui.py 存在")
    print("3. Python路径问题")
    
    # 列出当前目录的文件
    print("\n当前目录文件：")
    for f in os.listdir('.'):
        if f.endswith('.py'):
            print(f"  - {f}")
    
    input("\n按Enter键退出...")
except Exception as e:
    print(f"❌ 运行错误: {e}")
    import traceback
    traceback.print_exc()
    input("\n按Enter键退出...")