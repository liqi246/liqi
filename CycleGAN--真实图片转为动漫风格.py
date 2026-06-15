import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import os
import itertools
from tqdm import tqdm

# -------------------- 设备配置 --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -------------------- 超参数 --------------------
batch_size = 4
epochs = 200
lr = 2e-4
lambda_cycle = 10.0
lambda_identity = 5.0

# 图片大小与增强
img_size = 256
transforms_train = transforms.Compose([
    transforms.Resize(int(img_size * 1.12)),
    transforms.RandomCrop(img_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 数据集路径（请修改为你的实际路径）
dataroot = "./data"   # 你的 data 文件夹所在位置
trainA_path = os.path.join(dataroot, "trainA")
trainB_path = os.path.join(dataroot, "trainB")

# -------------------- 数据集加载 --------------------
dataset_A = ImageFolder(root=trainA_path, transform=transforms_train)
dataset_B = ImageFolder(root=trainB_path, transform=transforms_train)

loader_A = DataLoader(dataset_A, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
loader_B = DataLoader(dataset_B, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

# -------------------- 网络结构：生成器（ResNet-9块）--------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels)
        )
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_res=9):
        super().__init__()
        # 初始下采样
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # 残差块
        res_blocks = []
        for _ in range(n_res):
            res_blocks.append(ResidualBlock(256))
        self.res_blocks = nn.Sequential(*res_blocks)
        # 上采样
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, 7),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.res_blocks(x)
        x = self.up1(x)
        x = self.up2(x)
        return self.out(x)

# -------------------- 判别器（PatchGAN）--------------------
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        def block(in_c, out_c, stride=2, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride, 1)]
            if norm: layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        self.model = nn.Sequential(
            block(3, 64, stride=2, norm=False),   # 256 -> 128
            block(64, 128, stride=2),              # 128 -> 64
            block(128, 256, stride=2),             # 64 -> 32
            block(256, 512, stride=1),             # 32 -> 31
            nn.Conv2d(512, 1, 4, 1, 1)             # 31 -> 30
        )
    def forward(self, x):
        return self.model(x)

# -------------------- 损失函数 --------------------
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

# -------------------- 初始化网络 --------------------
G_A2B = Generator().to(device)   # 真实 -> 动漫
G_B2A = Generator().to(device)   # 动漫 -> 真实
D_A = Discriminator().to(device)  # 判别真实域 A
D_B = Discriminator().to(device)  # 判别动漫域 B

# 优化器
opt_G = optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()), lr=lr, betas=(0.5, 0.999))
opt_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))

# 学习率调度（逐步衰减）
def lambda_rule(epoch):
    lr_start = 1.0
    lr_end = 0.0
    decay_epoch = 100
    if epoch < decay_epoch:
        return 1.0
    else:
        return max(0.0, lr_end + (lr_start - lr_end) * (1 - (epoch - decay_epoch) / (epochs - decay_epoch)))

scheduler_G = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lambda_rule)
scheduler_D_A = optim.lr_scheduler.LambdaLR(opt_D_A, lr_lambda=lambda_rule)
scheduler_D_B = optim.lr_scheduler.LambdaLR(opt_D_B, lr_lambda=lambda_rule)

# -------------------- 训练循环 --------------------
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("开始训练 CycleGAN...")
for epoch in range(epochs):
    # 使用 tqdm 显示进度
    pbar = tqdm(zip(loader_A, loader_B), total=min(len(loader_A), len(loader_B)), desc=f"Epoch {epoch+1}/{epochs}")
    for (real_A, _), (real_B, _) in pbar:
        real_A, real_B = real_A.to(device), real_B.to(device)
        batch_size_cur = real_A.size(0)

        # 真实标签和假标签（用于对抗损失）
        real_label = torch.ones(batch_size_cur, 1, 30, 30).to(device)   # PatchGAN 输出是 30x30
        fake_label = torch.zeros(batch_size_cur, 1, 30, 30).to(device)

        # ---------- 1. 训练生成器 ----------
        opt_G.zero_grad()

        # Identity loss（有助于保留颜色/内容）
        id_A = G_B2A(real_A)
        loss_identity_A = criterion_identity(id_A, real_A) * lambda_identity
        id_B = G_A2B(real_B)
        loss_identity_B = criterion_identity(id_B, real_B) * lambda_identity

        # GAN loss
        fake_B = G_A2B(real_A)
        pred_fake_B = D_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake_B, real_label)

        fake_A = G_B2A(real_B)
        pred_fake_A = D_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake_A, real_label)

        # Cycle consistency loss
        rec_A = G_B2A(fake_B)
        loss_cycle_A = criterion_cycle(rec_A, real_A) * lambda_cycle
        rec_B = G_A2B(fake_A)
        loss_cycle_B = criterion_cycle(rec_B, real_B) * lambda_cycle

        total_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_A + loss_cycle_B
        total_G.backward()
        opt_G.step()

        # ---------- 2. 训练判别器 D_A (区分真实A和生成A) ----------
        opt_D_A.zero_grad()
        pred_real_A = D_A(real_A)
        loss_D_real_A = criterion_GAN(pred_real_A, real_label)

        fake_A_detach = fake_A.detach()
        pred_fake_A = D_A(fake_A_detach)
        loss_D_fake_A = criterion_GAN(pred_fake_A, fake_label)

        loss_D_A_total = (loss_D_real_A + loss_D_fake_A) * 0.5
        loss_D_A_total.backward()
        opt_D_A.step()

        # ---------- 3. 训练判别器 D_B (区分真实B和生成B) ----------
        opt_D_B.zero_grad()
        pred_real_B = D_B(real_B)
        loss_D_real_B = criterion_GAN(pred_real_B, real_label)

        fake_B_detach = fake_B.detach()
        pred_fake_B = D_B(fake_B_detach)
        loss_D_fake_B = criterion_GAN(pred_fake_B, fake_label)

        loss_D_B_total = (loss_D_real_B + loss_D_fake_B) * 0.5
        loss_D_B_total.backward()
        opt_D_B.step()

        # 更新进度条显示
        pbar.set_postfix({
            "G_loss": f"{total_G.item():.3f}",
            "D_A": f"{loss_D_A_total.item():.3f}",
            "D_B": f"{loss_D_B_total.item():.3f}"
        })

    # 更新学习率
    scheduler_G.step()
    scheduler_D_A.step()
    scheduler_D_B.step()

    # 保存生成器示例图片
    if (epoch+1) % 10 == 0:
        G_A2B.eval()
        with torch.no_grad():
            sample_A = real_A[:4]
            fake_B_sample = G_A2B(sample_A)
            comparison = torch.cat([sample_A, fake_B_sample], dim=0)
            save_image(comparison, f"results/epoch_{epoch+1:03d}.png", nrow=4, normalize=True)
        G_A2B.train()
        # 保存模型
        torch.save(G_A2B.state_dict(), f"checkpoints/G_A2B_epoch{epoch+1}.pth")
        torch.save(G_B2A.state_dict(), f"checkpoints/G_B2A_epoch{epoch+1}.pth")
        print(f"\n💾 Saved model at epoch {epoch+1}")

print("训练完成！")