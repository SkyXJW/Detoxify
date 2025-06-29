from model_train import *
import torch

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from sklearn.manifold import TSNE
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 模型初始设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPAE(
        in_channels=4096,  # 根据 representation 的特征维度调整
        semantic_latent_dim=1024,
        truthful_latent_dim=1024,
        semantic_hidden_dims=[2048],
        truthful_hidden_dims=[2048],
        decoder_hidden_dims=[2048]
    ).to(device)

# 加载训练好的模型权重
checkpoint = torch.load("/home/xjg/myTruthX/llava_truthx_model_100epoch.pt")
model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.to(device)

# 加载数据
val_data_pos = torch.load("/home/xjg/myTruthX/data/dinm/SafeEdit/llava/val_common_representations_pos.pth")
val_data_neg = torch.load("/home/xjg/myTruthX/data/dinm/SafeEdit/llava/val_common_representations_neg.pth")
for i in range(64):
    i = 26
    # 获取latenr representation(这里仅选择其中一层的数据)
    sem_pos_latents = model.get_semantic_latent_rep(val_data_pos[i]).detach().cpu().numpy()
    sem_neg_latents = model.get_semantic_latent_rep(val_data_neg[i]).detach().cpu().numpy()
    tru_pos_latents = model.get_truthful_latent_rep(val_data_pos[i]).detach().cpu().numpy()
    tru_neg_latents = model.get_truthful_latent_rep(val_data_neg[i]).detach().cpu().numpy()

    sem_latents = np.vstack([sem_pos_latents,sem_neg_latents])
    tru_latents = np.vstack([tru_pos_latents,tru_neg_latents])
    labels = np.array([1] * len(sem_pos_latents) + [0] * len(sem_neg_latents))

    # 使用 t-SNE 降维到2维
    tsne = TSNE(n_components=2, random_state=42)
    sem_2d = tsne.fit_transform(sem_latents)
    tru_2d = tsne.fit_transform(tru_latents)

    # 构造DataFrame
    sem_data = pd.DataFrame({
        "Dim1": sem_2d[:, 0],
        "Dim2": sem_2d[:, 1],
        "Label": ["Truthful" if l == 1 else "Untruthful" for l in labels]
    })
    tru_data = pd.DataFrame({
        "Dim1": tru_2d[:, 0],
        "Dim2": tru_2d[:, 1],
        "Label": ["Truthful" if l == 1 else "Untruthful" for l in labels]
    })

    # 绘制核密度估计
    plt.figure(figsize=(12, 6))

    # (a) Semantic Space
    plt.subplot(1, 2, 1)
    sns.kdeplot(
        data=sem_data, x="Dim1", y="Dim2", hue="Label",
        fill=False, common_norm=False, alpha=0.5
    )
    plt.title("Semantic Space")
    # (b) Truthful Space
    plt.subplot(1, 2, 2)
    sns.kdeplot(
        data=tru_data, x="Dim1", y="Dim2", hue="Label",
        fill=False, common_norm=False, alpha=0.5
    )
    plt.title("Truthful Space")

    plt.tight_layout()
    plt.savefig(f"layer{i}.png", dpi=300)
    exit()
