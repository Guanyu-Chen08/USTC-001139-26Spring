import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from dataset import get_dataloaders
from models import MLP, PointCNN, PointTransformer

from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import random

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def evaluate_model(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            _, predicted = torch.max(model(X_batch).data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    return (correct / total) if total > 0 else 0

def train_and_eval(model, train_loader, test_loader_seen, test_loader_unseen, val_loader=None, lr=0.01, epochs=50):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        history['train_loss'].append(running_loss / len(train_loader))
        history['train_acc'].append(evaluate_model(model, train_loader))
        
        if val_loader is not None:
            model.eval()
            v_loss = 0.0
            with torch.no_grad():
                for vX, vy in val_loader:
                    vX, vy = vX.to(device), vy.to(device)
                    v_loss += criterion(model(vX), vy).item()
            history['val_loss'].append(v_loss / len(val_loader))
            history['val_acc'].append(evaluate_model(model, val_loader))
            
    return evaluate_model(model, test_loader_seen), evaluate_model(model, test_loader_unseen), history

def plot_decision_boundary(model, dataset, title, save_path):
    model.eval()
    model = model.to(device)
    
    X, y = dataset.X, dataset.y
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    
    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(grid_tensor)
        _, predicted = torch.max(preds, 1)
        
    Z = predicted.cpu().numpy().reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm, s=50)
    plt.title(title)
    plt.xlabel('体长 (Standardized)')
    plt.ylabel('翼长 (Standardized)')
    plt.legend(*scatter.legend_elements(), title="昆虫类别", loc='upper right')
    plt.savefig(save_path, dpi=300)
    plt.close()

def get_all_data(loader):
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs, dim=0).numpy(), torch.cat(ys, dim=0).numpy()

def run_all_experiments(dataset_folder, use_noisy, output_dir, epochs=50):
    prefix = "insects-2" if use_noisy else "insects"
    train_path = os.path.join(dataset_folder, f"{prefix}-training.txt")
    test_path = os.path.join(dataset_folder, f"{prefix}-testing.txt")

    loaders, scaler = get_dataloaders(train_path, test_path)
    
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_seen = loaders['test_seen']
    test_unseen = loaders['test_unseen']
    
    train_loader_full = loaders['full_train_loader'] 
    test_full_dataset = loaders['full_test_ds']

    table_lines = [f"# {prefix} 数据集实验数据汇总\n"]

    models_dict = {
        'MLP (基线)': MLP(hidden_dim=32, num_hidden_layers=2), 
        'CNN': PointCNN(),
        'Transformer': PointTransformer(), 
    }
    results_arch = {}
    table_lines.extend([f"## 1. 宏观模型架构横向对比", "| 模型架构 | 前60个(见过)准确率 | 后150个(未见)准确率 |", "|:---:|:---:|:---:|"])
    
    for name, model in models_dict.items():
        acc_s, acc_u, _ = train_and_eval(model, train_loader, test_seen, test_unseen, val_loader=val_loader, epochs=epochs)
        results_arch[name] = (acc_s, acc_u)
        table_lines.append(f"| {name} | {acc_s:.4f} | {acc_u:.4f} |")

    X_train, y_train = get_all_data(train_loader_full)
    X_seen, y_seen = get_all_data(test_seen)
    X_unseen, y_unseen = get_all_data(test_unseen)
    
    trad_models = {
        'SVM (RBF)': SVC(kernel='rbf'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN (K=3)': KNeighborsClassifier(n_neighbors=3)
    }
    for name, m in trad_models.items():
        m.fit(X_train, y_train)
        acc_s = m.score(X_seen, y_seen)
        acc_u = m.score(X_unseen, y_unseen)
        results_arch[name] = (acc_s, acc_u)
        table_lines.append(f"| {name} | {acc_s:.4f} | {acc_u:.4f} |")

    labels, seen_acc, unseen_acc = list(results_arch.keys()), [res[0] for res in results_arch.values()], [res[1] for res in results_arch.values()]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(14, 6))
    rects1 = ax.bar([i - 0.175 for i in x], seen_acc, 0.35, label='前60个 (训练集内)')
    rects2 = ax.bar([i + 0.175 for i in x], unseen_acc, 0.35, label='后150个 (新数据)')
    ax.bar_label(rects1, padding=3, fmt='%.4f')
    ax.bar_label(rects2, padding=3, fmt='%.4f')
    ax.set_ylim(0, 1.15); ax.set_ylabel('Accuracy'); ax.set_title(f'模型架构对比 ({prefix})')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20); ax.legend(loc='upper right')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, f'arch_comparison_{prefix}.png')); plt.close()

    lrs = [0.001, 0.005, 0.01, 0.05, 0.1]
    table_lines.extend([f"\n## 2. 超参数探究 - 学习率 (模型: MLP 2层 32维)", "| 学习率 | 前60个(见过)准确率 | 后150个(未见)准确率 |", "|:---:|:---:|:---:|"])
    for lr in lrs:
        acc_s, acc_u, _ = train_and_eval(MLP(), train_loader, test_seen, test_unseen, val_loader=val_loader, lr=lr, epochs=epochs)
        table_lines.append(f"| {lr} | {acc_s:.4f} | {acc_u:.4f} |")
    
    acts = ['relu', 'tanh', 'sigmoid']
    table_lines.extend([f"\n## 3. 超参数探究 - 激活函数 (模型: MLP 2层 32维, LR=0.01)", "| 激活函数 | 前60个(见过)准确率 | 后150个(未见)准确率 |", "|:---:|:---:|:---:|"])
    for act in acts:
        acc_s, acc_u, _ = train_and_eval(MLP(activation=act), train_loader, test_seen, test_unseen, val_loader=val_loader, epochs=epochs)
        table_lines.append(f"| {act.upper()} | {acc_s:.4f} | {acc_u:.4f} |")

    dims = [4, 8, 16, 32, 64, 128]
    table_lines.extend([f"\n## 4. 微观架构探究 - 网络宽度 (模型: MLP 2层, LR=0.01)", "| 隐藏层维度 | 前60个(见过)准确率 | 后150个(未见)准确率 |", "|:---:|:---:|:---:|"])
    for dim in dims:
        acc_s, acc_u, _ = train_and_eval(MLP(hidden_dim=dim, num_hidden_layers=2), train_loader, test_seen, test_unseen, val_loader=val_loader, epochs=epochs)
        table_lines.append(f"| {dim} | {acc_s:.4f} | {acc_u:.4f} |")

    layers_list = [1, 2, 3, 5, 8]
    table_lines.extend([f"\n## 5. 微观架构探究 - 网络深度 (模型: MLP 16维, LR=0.01)", "| 隐藏层数 | 前60个(见过)准确率 | 后150个(未见)准确率 |", "|:---:|:---:|:---:|"])
    for num_layers in layers_list:
        acc_s, acc_u, _ = train_and_eval(MLP(hidden_dim=16, num_hidden_layers=num_layers), train_loader, test_seen, test_unseen, val_loader=val_loader, epochs=epochs)
        table_lines.append(f"| {num_layers} 层 | {acc_s:.4f} | {acc_u:.4f} |")

    table_text = "\n".join(table_lines)

    with open(os.path.join(output_dir, f'hyperparams_table_{prefix}.md'), 'w', encoding='utf-8') as f:
        f.write(table_text)

    print(f"\n>>> 正在使用最佳配置训练最终模型并绘制学习曲线和决策边界 ({prefix}) ...")
    
    best_model = MLP(hidden_dim=128, num_hidden_layers=2, activation='relu')
    
    acc_s, acc_u, history = train_and_eval(
        best_model, 
        train_loader, 
        test_seen, 
        test_unseen, 
        val_loader=val_loader, 
        lr=0.005, 
        epochs=epochs 
    ) 
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    eps = range(1, epochs + 1)
    
    ax1.plot(eps, history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    ax1.plot(eps, history['val_loss'], label='Validation Loss', color='orange', linewidth=2, linestyle='--')
    ax1.set_title(f'Loss 随 Epoch 变化曲线 ({prefix})')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(eps, history['train_acc'], label='Train Accuracy', color='blue', linewidth=2)
    ax2.plot(eps, history['val_acc'], label='Validation Accuracy', color='orange', linewidth=2, linestyle='--')
    ax2.set_title(f'Accuracy 随 Epoch 变化曲线 ({prefix})')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'learning_curves_best_{prefix}.png'), dpi=300)
    plt.close()

    plot_decision_boundary(
        best_model, 
        test_full_dataset, 
        title=f"最优配置 MLP 决策边界", 
        save_path=os.path.join(output_dir, f'decision_boundary_best_{prefix}.png')
    )