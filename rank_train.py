# 与center_train.py文件一样，虽然该文件命名为train，但实际并没有进行训练
# 只是将训练好的模型用于获取rank

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,5,6,7"

from model_train import *
import torch
import gc


import torch.nn.functional as F

def probe(h_truth: Tensor, H_truth_pos_center: Tensor, H_truth_neg_center: Tensor, isPositive: bool):
    num_layers, batch_size, latent_dim = h_truth.shape
    
    # 扩展 H_truth_pos_center 和 H_truth_neg_center 的维度以匹配 batch_size
    H_truth_pos_center = H_truth_pos_center.unsqueeze(1).expand(-1, batch_size, -1)  # [num_layers, batch_size, latent_dim]
    H_truth_neg_center = H_truth_neg_center.unsqueeze(1).expand(-1, batch_size, -1)  # [num_layers, batch_size, latent_dim]
    
    # 计算 h_truth 与两个中心的余弦相似度
    sim_pos = F.cosine_similarity(h_truth, H_truth_pos_center, dim=-1)  # [num_layers, batch_size]
    sim_neg = F.cosine_similarity(h_truth, H_truth_neg_center, dim=-1)  # [num_layers, batch_size]
    
    # 判断 Probe(x) 的结果：sim_pos >= sim_neg 为正样本
    probe_result = sim_pos >= sim_neg  # [num_layers, batch_size]
    
    # 比较 Probe(x) 的结果与实际标签
    ground_truth = torch.ones_like(probe_result) if isPositive else torch.zeros_like(probe_result)
    correct = (probe_result == ground_truth).float()  # [num_layers, batch_size]

    return correct
    
    # # 计算每一层的准确率
    # acc = correct.mean(dim=-1)  # [num_layers]
    
    # return acc

if __name__ == "__main__":
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
    checkpoint = torch.load("/home/xjg/myTruthX/truthx_models/llava-v1.5-7b/new_checkpoint.pt")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(device)

    # 加载训练数据
    validation_data_pos = torch.load("/home/xjg/myTruthX/data/dinm/SafeEdit/llava/200_val_common_representations_pos.pth")
    validation_data_neg = torch.load("/home/xjg/myTruthX/data/dinm/SafeEdit/llava/200_val_common_representations_neg.pth")

    # # 初始化存储结果的张量
    # pos_latents = torch.stack([model.get_truthful_latent_rep(layer) for layer in validation_data_pos])  # shape: (num_layers, batch_size, latent_dim)
    # neg_latents = torch.stack([model.get_truthful_latent_rep(layer) for layer in validation_data_neg])

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
    pos_latents = get_latents_batched(validation_data_pos)
    neg_latents = get_latents_batched(validation_data_neg)

    # 在每一层进行probe
    pos_probe = probe(pos_latents, checkpoint['pos_center'], checkpoint['neg_center'], True)
    neg_probe = probe(neg_latents, checkpoint['pos_center'], checkpoint['neg_center'], False)

    probe_acc = torch.cat((pos_probe,neg_probe), dim=1).mean(dim=-1)

    # 1. 获取降序排序的索引
    _, sorted_indices = torch.sort(probe_acc, descending=True)

    # 2. 生成排名 (从 1 开始)
    ranks = torch.empty_like(sorted_indices, dtype=torch.long)
    ranks[sorted_indices] = torch.arange(1, len(probe_acc) + 1, device=probe_acc.device)

    # 3. 将排名保存到 list 中
    rank_list = ranks.tolist()

    # 添加rank
    new_checkpoint = {
        'args': checkpoint['args'],
        'state_dict': checkpoint['state_dict'],
        'pos_center': checkpoint['pos_center'],
        'neg_center': checkpoint['neg_center'],
        'rank': rank_list
    }
    torch.save(new_checkpoint,"llava_truthx_model_100epoch_300sample.pt")






