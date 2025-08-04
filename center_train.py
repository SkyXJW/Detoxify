# 虽然该文件命名为train，但实际并没有进行训练
# 只是将训练好的模型用于获取pos_center与neg_center

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

from model_train import *
import torch

import argparse
import gc

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
checkpoint = torch.load("/home/xjg/mistral_truthx_model_100epoch_500sample.pt")
model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.to(device)

# 加载训练数据
train_data_pos = torch.load("/home/xjg/myTruthX/data/dinm/SafeEdit/mistral/500_train_pos.pth")
train_data_neg = torch.load("/home/xjg/myTruthX/data/dinm/SafeEdit/mistral/500_train_neg.pth")

# # 初始化存储结果的张量
# pos_latents = torch.stack([model.get_truthful_latent_rep(layer) for layer in train_data_pos])  # shape: (num_layers, batch_size, latent_dim)
# neg_latents = torch.stack([model.get_truthful_latent_rep(layer) for layer in train_data_neg])

# 分批处理函数
def get_latents_batched(data_list):
    latents = []
    with torch.no_grad():
        for layer in data_list:
            layer = layer.to(device)
            latent = model.get_truthful_latent_rep(layer)
            latents.append(latent.cpu())  # 可选：移到CPU释放显存
            del layer, latent
            gc.collect()
            torch.cuda.empty_cache()
    return torch.stack(latents)  # shape: (num_layers, batch_size, latent_dim)

# 处理并计算中心向量
pos_latents = get_latents_batched(train_data_pos)
neg_latents = get_latents_batched(train_data_neg)

# 计算均值
pos_center = pos_latents.mean(dim=1)  # shape: (num_layers, latent_dim)
neg_center = neg_latents.mean(dim=1)  # shape: (num_layers, latent_dim)

print(pos_center)
print(neg_center)

# 添加pos_center、neg_center
new_checkpoint = {
    'args': checkpoint['args'],
    'state_dict': checkpoint['state_dict'],
    'pos_center': pos_center,
    'neg_center': neg_center
}
torch.save(new_checkpoint,"new_checkpoint.pt")

