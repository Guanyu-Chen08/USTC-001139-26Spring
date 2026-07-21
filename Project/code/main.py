import os
import datetime 
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

def seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用的计算设备: {device}")           

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
result_dir = f"results_{timestamp}"
os.makedirs(result_dir, exist_ok=True)
print(f"所有生成的图片和结果将保存在文件夹: {result_dir}\n")

CLASS_NAMES = ["Gaussian Mixture", "Ring", "Two Moons", "Spiral"]

def compute_swd(X, Y, num_projections=100):
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    
    projections = torch.randn(num_projections, 2)
    projections = projections / torch.norm(projections, dim=1, keepdim=True)
    
    proj_X, _ = torch.sort(torch.matmul(X, projections.T), dim=0)
    proj_Y, _ = torch.sort(torch.matmul(Y, projections.T), dim=0)

    swd = torch.mean(torch.abs(proj_X - proj_Y)).item()
    return swd

def compute_mmd(X, Y, sigma=1.0):
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    
    XX = torch.cdist(X, X) ** 2
    YY = torch.cdist(Y, Y) ** 2
    XY = torch.cdist(X, Y) ** 2
    
    K_XX = torch.exp(-XX / (2 * sigma ** 2)).mean()
    K_YY = torch.exp(-YY / (2 * sigma ** 2)).mean()
    K_XY = torch.exp(-XY / (2 * sigma ** 2)).mean()
    
    return (K_XX + K_YY - 2 * K_XY).item()

class UnconditionalMLP(nn.Module):
    def __init__(self, dim=2, hidden_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(32, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(dim + hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )

    def get_time_embed(self, t):
        half_dim = 16
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return self.time_embed(emb)

    def forward(self, x, t, c=None):
        t_emb = self.get_time_embed(t)
        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h)

class ConditionalMLP(nn.Module):
    def __init__(self, dim=2, hidden_dim=128, num_classes=4):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(32, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.class_embed = nn.Embedding(num_classes, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(dim + hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )

    def get_time_embed(self, t):
        half_dim = 16
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return self.time_embed(emb)

    def forward(self, x, t, c):
        t_emb = self.get_time_embed(t)
        c_emb = self.class_embed(c)
        emb = t_emb + c_emb
        h = torch.cat([x, emb], dim=-1)
        return self.net(h)

class DDPM:
    def __init__(self, model, num_timesteps=100, device=device):
        self.num_timesteps = num_timesteps
        self.device = device
        self.model = model.to(device)
        self.beta = torch.linspace(1e-4, 0.02, num_timesteps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def train(self, X_train, Y_train=None, epochs=800, batch_size=1024, show_bar=True):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-3)
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        
        if Y_train is not None:
            Y_tensor = torch.tensor(Y_train, dtype=torch.long)
            dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
        else:
            dataset = torch.utils.data.TensorDataset(X_tensor)
            
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.train()
        pbar_epochs = tqdm(range(epochs), desc="Training DDPM") if show_bar else range(epochs)
        
        for epoch in pbar_epochs:
            for batch in loader:
                x0 = batch[0].to(self.device)
                c = batch[1].to(self.device) if len(batch) > 1 else None
                t = torch.randint(0, self.num_timesteps, (x0.shape[0],), device=self.device)
                noise = torch.randn_like(x0)
                a_bar_t = self.alpha_bar[t].view(-1, 1)
                xt = torch.sqrt(a_bar_t) * x0 + torch.sqrt(1 - a_bar_t) * noise
                pred_noise = self.model(xt, t, c)
                loss = F.mse_loss(pred_noise, noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    @torch.no_grad()
    def sample(self, num_samples, c_label=None):
        self.model.eval()
        x = torch.randn((num_samples, 2), device=self.device)
        if c_label is not None:
            c = torch.full((num_samples,), c_label, dtype=torch.long, device=self.device)
        else:
            c = None
            
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            pred_noise = self.model(x, t_tensor, c)
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            beta_t = self.beta[t]
            noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_noise)
            x = x + torch.sqrt(beta_t) * noise
            
        return x.cpu().numpy()

def plot_scatter(points, title, ax, color='blue'):
    ax.scatter(points[:, 0], points[:, 1], s=2, alpha=0.5, c=color)
    ax.set_title(title)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')


def resolve_data_dir():
    base_dir = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
    candidates = [
        base_dir / 'data',
        base_dir.parent / 'data',
        Path.cwd() / 'data',
        Path.cwd().parent / 'data',
    ]

    required_files = ['train.npy', 'train_label.npy', 'test.npy', 'test_label.npy']
    for data_dir in candidates:
        if all((data_dir / name).exists() for name in required_files):
            return data_dir

    candidate_list = '\n'.join(f'- {path}' for path in candidates)
    raise FileNotFoundError(f'未找到数据目录，请确认以下候选路径中存在数据文件:\n{candidate_list}')

def main():
    print("=" * 70)
    print("正在加载数据集...")
    data_dir = resolve_data_dir()
    print(f"数据目录: {data_dir}")
    train_X = np.load(data_dir / 'train.npy')
    train_Y = np.load(data_dir / 'train_label.npy')
    test_X = np.load(data_dir / 'test.npy')
    test_Y = np.load(data_dir / 'test_label.npy')
    
    print("=" * 70)
    print("【开始任务一】：分别可视化四类真实数据的分布结构")
    for c in range(4):
        c_data = train_X[train_Y == c]
        fig, ax = plt.subplots(figsize=(5, 5))
        plot_scatter(c_data, f"Real Data: {CLASS_NAMES[c]}", ax, color='blue')
        plt.tight_layout()
        safe_name = CLASS_NAMES[c].replace(" ", "")
        plt.savefig(f"{result_dir}/Data_{safe_name}.png", dpi=150)
        plt.close()
    print(f"四张原始数据可视化图已保存至: {result_dir}")
    print("=" * 70)
    print("【开始基础任务】：独立无条件生成 (Unconditional) 并计算评价指标")
    samp_gmm_uncond = {}
    samp_ddpm_uncond = {}
    fig1, axes1 = plt.subplots(4, 3, figsize=(15, 20))
    
    metrics_results = []
    
    for c in range(4):
        print(f"\n正在为 [{CLASS_NAMES[c]}] 训练专属的无条件模型...")
        c_train_X = train_X[train_Y == c]
        c_test_X = test_X[test_Y == c]
        
        gmm = GaussianMixture(n_components=10, random_state=42).fit(c_train_X)
        samp_gmm_uncond[c], _ = gmm.sample(2000)
        
        swd_gmm = compute_swd(c_test_X, samp_gmm_uncond[c])
        mmd_gmm = compute_mmd(c_test_X, samp_gmm_uncond[c])
        metrics_results.append((CLASS_NAMES[c], "GMM", swd_gmm, mmd_gmm))
        
        uncond_mlp = UnconditionalMLP(hidden_dim=128)
        ddpm_uncond = DDPM(model=uncond_mlp, num_timesteps=100)
        ddpm_uncond.train(c_train_X, epochs=500, show_bar=True)
        samp_ddpm_uncond[c] = ddpm_uncond.sample(num_samples=2000)
        
        swd_ddpm = compute_swd(c_test_X, samp_ddpm_uncond[c])
        mmd_ddpm = compute_mmd(c_test_X, samp_ddpm_uncond[c])
        metrics_results.append((CLASS_NAMES[c], "DDPM", swd_ddpm, mmd_ddpm))
        
        plot_scatter(c_test_X, f"Real: {CLASS_NAMES[c]}", axes1[c, 0], 'blue')
        plot_scatter(samp_gmm_uncond[c], f"Uncond GMM: {CLASS_NAMES[c]}", axes1[c, 1], 'red')
        plot_scatter(samp_ddpm_uncond[c], f"Uncond DDPM: {CLASS_NAMES[c]}", axes1[c, 2], 'green')

    plt.tight_layout()
    plt.savefig(f"{result_dir}/Part1_Unconditional_Separated.png", dpi=150)
    plt.close()
    
    print("\n" + "=" * 70)
    print("【基础任务指标评估报告】: 四类图像 x 两种模型的定量打分")
    print(f"| {'数据集 (Dataset)':<20} | {'模型 (Model)':<12} | {'SWD Score ↓':<15} | {'MMD Score ↓':<15} |")
    print("|" + "-"*22 + "|" + "-"*14 + "|" + "-"*17 + "|" + "-"*17 + "|")
    for row in metrics_results:
        print(f"| {row[0]:<20} | {row[1]:<12} | {row[2]:<15.4f} | {row[3]:<15.4f} |")
    print("=" * 70)

    print("\n【开始拓展任务二】：统一架构条件生成 (CDDPM)")
    cond_mlp = ConditionalMLP(hidden_dim=128, num_classes=4)
    cddpm = DDPM(model=cond_mlp, num_timesteps=100)
    cddpm.train(train_X, Y_train=train_Y, epochs=600, show_bar=True)
    
    samp_cddpm_cond = {}
    fig2, axes2 = plt.subplots(4, 2, figsize=(10, 20))
    for c in range(4):
        samp_cddpm_cond[c] = cddpm.sample(num_samples=2000, c_label=c)
        c_test_X = test_X[test_Y == c]
        plot_scatter(c_test_X, f"Real: {CLASS_NAMES[c]}", axes2[c, 0], 'blue')
        plot_scatter(samp_cddpm_cond[c], f"Unified CDDPM: {CLASS_NAMES[c]}", axes2[c, 1], 'green')
        
    plt.tight_layout()
    plt.savefig(f"{result_dir}/Part2_Unified_Conditional_CDDPM.png", dpi=150)
    plt.close()

    print("=" * 70)
    print("【开始拓展任务一】：模型核心超参数剖析测试")
    spiral_train = train_X[train_Y == 3]
    spiral_test = test_X[test_Y == 3]
    
    n_components_list = [1, 2, 5, 8, 10, 15, 20, 30, 40]
    swd_gmm_scores = []
    print(f"\n| {'GMM n_components':<16} | {'SWD Score':<15} |")
    for n in n_components_list:
        gmm = GaussianMixture(n_components=n, random_state=42).fit(spiral_train)
        samp, _ = gmm.sample(2000)
        score = compute_swd(spiral_test, samp)
        swd_gmm_scores.append(score)
        print(f"| {n:<16} | {score:<15.4f} |")

    timesteps_list = [5, 10, 20, 30, 50, 80, 100, 150]
    swd_ddpm_scores = []
    print(f"\n| {'DDPM timesteps':<16} | {'SWD Score':<15} |")
    for ts in timesteps_list:
        net_ts = UnconditionalMLP(hidden_dim=128) 
        ddpm_ts = DDPM(model=net_ts, num_timesteps=ts)
        ddpm_ts.train(spiral_train, epochs=300, batch_size=1024, show_bar=False)
        samp_ts = ddpm_ts.sample(num_samples=2000)
        score = compute_swd(spiral_test, samp_ts)
        swd_ddpm_scores.append(score)
        print(f"| {ts:<16} | {score:<15.4f} |")

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
    axes3[0].plot(n_components_list, swd_gmm_scores, marker='o', c='purple', linewidth=2)
    axes3[0].set_title("GMM: SWD vs n_components")
    axes3[0].set_xlabel("n_components")
    axes3[0].set_ylabel("SWD Score")
    axes3[0].grid(True, linestyle='--', alpha=0.6)
    
    axes3[1].plot(timesteps_list, swd_ddpm_scores, marker='s', c='orange', linewidth=2)
    axes3[1].set_title("DDPM: SWD vs num_timesteps")
    axes3[1].set_xlabel("num_timesteps")
    axes3[1].set_ylabel("SWD Score")
    axes3[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f"{result_dir}/Ext1_Hyperparameters_Multi.png", dpi=150)
    plt.close()

    print("=" * 70)
    print("【开始拓展任务三】：异常点鲁棒性分析 (注入 5% 全局噪点)")
    num_noise_total = int(len(train_X) * 0.05)
    noise_points_total = np.random.uniform(low=-7.0, high=7.0, size=(num_noise_total, 2))
    noise_labels_total = np.random.randint(0, 4, size=(num_noise_total,))
    noisy_train_X = np.concatenate([train_X, noise_points_total], axis=0)
    noisy_train_Y = np.concatenate([train_Y, noise_labels_total], axis=0)

    noisy_gmm_dict = {}
    samp_noisy_gmm = {}
    for c in range(4):
        c_train_noisy = noisy_train_X[noisy_train_Y == c]
        gmm = GaussianMixture(n_components=10, random_state=42).fit(c_train_noisy)
        noisy_gmm_dict[c] = gmm
        samp_noisy_gmm[c], _ = gmm.sample(2000)

    cond_mlp_noisy = ConditionalMLP(hidden_dim=128, num_classes=4)
    cddpm_noisy = DDPM(model=cond_mlp_noisy, num_timesteps=100)
    cddpm_noisy.train(noisy_train_X, Y_train=noisy_train_Y, epochs=600, show_bar=True)
    
    samp_noisy_cddpm = {}
    robustness_metrics = []
    
    fig4, axes4 = plt.subplots(4, 3, figsize=(15, 20))
    for c in range(4):
        c_train_noisy = noisy_train_X[noisy_train_Y == c]
        c_test_X = test_X[test_Y == c]
        
        samp_noisy_cddpm[c] = cddpm_noisy.sample(num_samples=2000, c_label=c)
        
        swd_gmm_noisy = compute_swd(c_test_X, samp_noisy_gmm[c])
        mmd_gmm_noisy = compute_mmd(c_test_X, samp_noisy_gmm[c])
        robustness_metrics.append((CLASS_NAMES[c], "Noisy GMM", swd_gmm_noisy, mmd_gmm_noisy))
        
        swd_ddpm_noisy = compute_swd(c_test_X, samp_noisy_cddpm[c])
        mmd_ddpm_noisy = compute_mmd(c_test_X, samp_noisy_cddpm[c])
        robustness_metrics.append((CLASS_NAMES[c], "Noisy CDDPM", swd_ddpm_noisy, mmd_ddpm_noisy))
        
        plot_scatter(c_train_noisy, f"Noisy Real: {CLASS_NAMES[c]}", axes4[c, 0], 'black')
        plot_scatter(samp_noisy_gmm[c], f"Noisy GMM Gen", axes4[c, 1], 'red')
        plot_scatter(samp_noisy_cddpm[c], f"Noisy CDDPM Gen", axes4[c, 2], 'green')
        
    plt.tight_layout()
    plt.savefig(f"{result_dir}/Ext3_Robustness_All_Classes.png", dpi=150)
    plt.close()
    
    print("\n" + "=" * 70)
    print("【拓展任务三指标评估报告】: 5% 脏数据攻击下的模型鲁棒性定量打分 (对比纯净测试集)")
    print(f"| {'数据集 (Dataset)':<20} | {'模型 (Model)':<15} | {'SWD Score ↓':<15} | {'MMD Score ↓':<15} |")
    print("|" + "-"*22 + "|" + "-"*17 + "|" + "-"*17 + "|" + "-"*17 + "|")
    for row in robustness_metrics:
        print(f"| {row[0]:<20} | {row[1]:<15} | {row[2]:<15.4f} | {row[3]:<15.4f} |")
    print("=" * 70)
    
    print(f"所有实验圆满结束！请查阅结果：{result_dir}")

if __name__ == "__main__":
    main()