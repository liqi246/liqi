<img width="189" height="386" alt="9d95329faa4b06f41477f41857922686" src="https://github.com/user-attachments/assets/b27aaf92-94cc-4784-974f-fe422d1e6c9b" /># liqi
# CycleGAN 真实照片转动漫风格

## 效果展示
<img width="189" height="386" alt="9d95329faa4b06f41477f41857922686" src="https://github.com/user-attachments/assets/d1dad45d-d206-46b6-9e4b-cb03b2a2f20d" />

## 项目简介
基于 CycleGAN 实现非配对图像风格迁移，将真实风景照转换为动漫风格。

## 环境要求
- Python 3.8+
- PyTorch 1.9+
- torchvision
- tqdm

## 快速开始
1. 克隆仓库
2. 准备数据集：放在 `./data/trainA` 和 `./data/trainB`
3. 运行训练：
   ```bash
   python CycleGAN--真实图片转为动漫风格.py
