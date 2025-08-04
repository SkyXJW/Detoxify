import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
from torch import nn
import torch.nn.functional as F
from abc import abstractmethod
from torch import tensor as Tensor
from typing import List, Any, Dict

import pandas as pd
from torch.utils.data import Dataset, DataLoader
import ast

from tqdm import tqdm

import matplotlib.pyplot as plt
import argparse

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]  # 返回的是 feature 表示

class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass


class MLPAE(BaseVAE):
    def __init__(
        self,
        in_channels: int,
        semantic_latent_dim: int,
        truthful_latent_dim: int,
        semantic_hidden_dims: List = None,
        truthful_hidden_dims: List = None,
        decoder_hidden_dims: List = None,
        **kwargs
    ) -> None:
        super(MLPAE, self).__init__()

        self.semantic_latent_dim = semantic_latent_dim

        if semantic_hidden_dims is None:
            semantic_hidden_dims = []

        # Build Semantic Encoder
        semantic_encoder_modules = []
        flat_size = in_channels
        for h_dim in semantic_hidden_dims:
            semantic_encoder_modules.append(
                nn.Sequential(
                    nn.Linear(flat_size, h_dim), nn.LayerNorm(h_dim), nn.LeakyReLU()
                )
            )
            flat_size = h_dim
        semantic_encoder_modules.append(
            nn.Sequential(
                nn.Linear(flat_size, semantic_latent_dim),
                nn.LayerNorm(semantic_latent_dim),
                nn.LeakyReLU(),
            )
        )

        self.semantic_encoder = nn.Sequential(*semantic_encoder_modules)

        if truthful_hidden_dims is None:
            truthful_hidden_dims = []

        # Build Truthful Encoder
        truthful_encoder_modules = []
        flat_size = in_channels
        for h_dim in truthful_hidden_dims:
            truthful_encoder_modules.append(
                nn.Sequential(
                    (
                        nn.Linear(flat_size, h_dim)
                        if flat_size != h_dim
                        else nn.Identity()
                    ),
                    nn.LayerNorm(h_dim),
                    nn.LeakyReLU(),
                )
            )
            flat_size = h_dim
        truthful_encoder_modules.append(
            nn.Sequential(
                (
                    nn.Linear(flat_size, truthful_latent_dim)
                    if flat_size != truthful_latent_dim
                    else nn.Identity()
                ),
                nn.LayerNorm(truthful_latent_dim),
                nn.LeakyReLU(),
            )
        )

        self.truthful_encoder = nn.Sequential(*truthful_encoder_modules)

        # Cross-Attention Module
        self.num_heads = 1
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=semantic_latent_dim, num_heads=self.num_heads
        )

        self.proj = None
        if semantic_latent_dim != truthful_latent_dim:
            self.proj = nn.Linear(truthful_latent_dim, semantic_latent_dim, bias=False)

        # Build Decoder
        decoder_modules = []
        if len(decoder_hidden_dims) > 0:
            flat_size = semantic_latent_dim
            for h_dim in decoder_hidden_dims:
                decoder_modules.append(
                    nn.Sequential(
                        nn.Linear(flat_size, h_dim), nn.LayerNorm(h_dim), nn.LeakyReLU()
                    )
                )
                flat_size = h_dim

            flat_size = decoder_hidden_dims[-1]
            self.decoder = nn.Sequential(*decoder_modules)
        else:
            self.decoder_input = None

            self.decoder = None
            flat_size = semantic_latent_dim
        self.final_layer = nn.Sequential(nn.Linear(flat_size, in_channels))

    def encode_semantic(self, input: Tensor) -> List[Tensor]:
        semantic_latent_rep = self.semantic_encoder(input)
        return semantic_latent_rep

    def encode_truthful(self, input: Tensor) -> List[Tensor]:
        truthful_latent_rep = self.truthful_encoder(input)
        truthful_latent_rep = F.normalize(truthful_latent_rep, p=2, dim=-1)

        return truthful_latent_rep

    def attention(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        if self.proj is not None and query.size(-1) != key.size(-1):
            key = self.proj(key)
            value = self.proj(value)
        query = query.unsqueeze(0)
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)

        output, attention_weights = self.cross_attention(query, key, value)

        return output[0]

    def decode(self, z: Tensor) -> Tensor:
        result = z
        if self.decoder is not None:
            result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(
        self, input: Tensor, truthful_latent_rep=None, **kwargs
    ) -> List[Tensor]:
        semantic_latent_rep = self.encode_semantic(input)
        if truthful_latent_rep is None:
            truthful_latent_rep = self.encode_truthful(input)
        truthful_latent_rep = truthful_latent_rep.reshape(
            -1, truthful_latent_rep.size(-1)
        )
        z = semantic_latent_rep + self.attention(
            semantic_latent_rep,
            truthful_latent_rep.contiguous(),
            truthful_latent_rep.contiguous(),
        )
        output = self.decode(z)

        return [output, input, semantic_latent_rep, truthful_latent_rep]

    def forward_decoder(self, input, semantic_latent_rep, truthful_latent_rep):
        z = semantic_latent_rep + self.attention(
            semantic_latent_rep, truthful_latent_rep, truthful_latent_rep
        )
        output = self.decode(z)
        return [output, input, semantic_latent_rep, truthful_latent_rep]

    def get_semantic_latent_rep(self, input: Tensor, **kwargs) -> List[Tensor]:
        semantic_latent_rep = self.encode_semantic(input)
        return semantic_latent_rep

    def get_truthful_latent_rep(self, input: Tensor, **kwargs) -> List[Tensor]:
        truthful_latent_rep = self.encode_truthful(input)
        return truthful_latent_rep

    # def recon_loss_function(self, *args, **kwargs) -> dict:
    #     recons = args[0]
    #     input = args[1]
    #     recons_loss = F.mse_loss(recons, input)

    #     loss = recons_loss
    #     return {"loss": loss, "Reconstruction_Loss": recons_loss.detach()}
    
    # custom
    def recon_loss_function(self, input: Tensor, recons: Tensor, **kwargs):
        recons_loss = F.mse_loss(recons, input, reduction='none').mean(dim=1).mean()
        return recons_loss
    
    def ctr_truth_loss_function(self, h_truth: Tensor, H_truth_pos: Tensor, H_truth_neg: Tensor, isPositive: bool, temperature: float, **kwargs):
        # 先把正负样本取均值
        H_truth_pos = torch.mean(H_truth_pos, dim=0)
        H_truth_neg = torch.mean(H_truth_neg, dim=0)

        h_truth = h_truth.unsqueeze(1)

        # 计算与正样本 H_truth_pos 的cosine similarity
        H_pos_representations = H_truth_pos.unsqueeze(0)
        cs = F.cosine_similarity(h_truth, H_pos_representations, dim = 2)
        cs_pos = torch.exp(cs / temperature).sum(dim = 1)

        # 计算与负样本 H_truth_neg 的cosine similarity
        H_neg_representations = H_truth_neg.unsqueeze(0)
        cs = F.cosine_similarity(h_truth, H_neg_representations, dim = 2)
        cs_neg = torch.exp(cs / temperature).sum(dim = 1)

        # 如果h_truth是正样本
        if isPositive:
            return torch.mean(-torch.log(cs_pos/(cs_pos+cs_neg)))
        return torch.mean(-torch.log(cs_neg/(cs_pos+cs_neg)))

    def ctr_semantic_loss_function(self, h_sem_pos: Tensor, h_sem_neg: Tensor, H_sem: Tensor, isPositive: bool, temperature: float, **kwargs):
        # 计算与正样本 S+ 的cosine similarity
        cs = F.cosine_similarity(h_sem_pos, h_sem_neg, dim=-1)
        cs_pos = torch.exp(cs / temperature)

        # trick：由于H_sem要么等于h_sem_pos，要么等于h_sem_neg，再加上H_sem与h_sem_pos或h_sem_neg中的元素同序，
        # 所以在剔除h_sem_pos或h_sem_neg时，对应的mask掩码一定是对角线上的元素为False，其余都为True
        mask = ~torch.eye(H_sem.size(0), dtype=torch.bool)

        # 计算与负样本 S- 的cosine similarity
        if isPositive:
            # 剔除h_sem_pos
            H_sem_representations = torch.stack([H_sem[mask[i]] for i in range(mask.size(0))])
            cs = F.cosine_similarity(h_sem_pos.unsqueeze(1), H_sem_representations, dim=2)
        else:
            # 剔除h_sem_neg
            H_sem_representations = torch.stack([H_sem[mask[i]] for i in range(mask.size(0))])
            cs = F.cosine_similarity(h_sem_neg.unsqueeze(1), H_sem_representations, dim=2)
        cs_neg = torch.exp(cs / temperature).sum(dim = 1)

        return torch.mean(-torch.log(cs_pos/(cs_pos+cs_neg)))
    
    def edit_loss_function(self, x_pos: Tensor, x_neg: Tensor, x_pos2neg: Tensor, x_neg2pos: Tensor, **kwargs):
        edit_loss_pos = F.mse_loss(x_pos, x_neg2pos, reduction='none').mean(dim=1).mean()
        edit_loss_neg = F.mse_loss(x_neg, x_pos2neg, reduction='none').mean(dim=1).mean()
        edit_loss = edit_loss_pos + edit_loss_neg
        return edit_loss

if __name__ == "__main__":
    # load训练数据
    train_data_pos = torch.load("/home/xjg/myTruthX/data/dinm/SafeEdit/mistral/500_train_pos.pth")
    train_data_neg = torch.load("/home/xjg/myTruthX/data/dinm/SafeEdit/mistral/500_train_neg.pth")
    train_data_pos = train_data_pos.reshape(-1,train_data_pos.shape[-1])
    train_data_neg = train_data_neg.reshape(-1,train_data_neg.shape[-1])

    # 创建 DataLoader
    batch_size = 512
    train_dataset_pos = CustomDataset(train_data_pos)
    train_loader_pos = DataLoader(train_data_pos, batch_size=batch_size, shuffle=False)
    train_dataset_neg = CustomDataset(train_data_neg)
    train_loader_neg = DataLoader(train_data_neg, batch_size=batch_size, shuffle=False)

    # 初始化设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPAE(
        in_channels=4096,  # 根据 representation 的特征维度调整
        semantic_latent_dim=1024,
        truthful_latent_dim=1024,
        semantic_hidden_dims=[2048],
        truthful_hidden_dims=[2048],
        decoder_hidden_dims=[2048]
    ).to(device)

    # 设置优化器和学习率
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    temperature = 0.1  # 对比学习的温度参数

    # 训练循环
    epochs = 100
    train_losses = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0


        # 使用 tqdm 包裹 DataLoader
        train_loader_iter = tqdm(
            zip(train_loader_pos, train_loader_neg), 
            total=min(len(train_loader_pos), len(train_loader_neg)), 
            desc=f"Epoch {epoch+1}/{epochs}"
        )

        for batch_pos, batch_neg in train_loader_iter:
            batch_pos = batch_pos.to(device)
            batch_neg = batch_neg.to(device)

            # 前向传播
            outputs_pos, inputs_pos, semantic_latent_pos, truthful_latent_pos = model(batch_pos)
            outputs_neg, inputs_neg, semantic_latent_neg, truthful_latent_neg = model(batch_neg)

            # 1. Reconstruction Loss
            recon_loss_pos = model.recon_loss_function(inputs_pos, outputs_pos)
            recon_loss_neg = model.recon_loss_function(inputs_neg, outputs_neg)
            recon_loss = recon_loss_pos + recon_loss_neg

            # 2. CTR Loss
            # 2.1 计算在truthful空间中的ctr loss
            ctr_truth_pos_loss = model.ctr_truth_loss_function(truthful_latent_pos, truthful_latent_pos, truthful_latent_neg, isPositive=True, temperature=temperature)
            ctr_truth_neg_loss = model.ctr_truth_loss_function(truthful_latent_neg, truthful_latent_pos, truthful_latent_neg, isPositive=False, temperature=temperature)
            ctr_truth_loss = ctr_truth_pos_loss + ctr_truth_neg_loss

            # 2.2 计算在semantic空间中的ctr loss
            ctr_sem_pos_loss = model.ctr_semantic_loss_function(semantic_latent_pos, semantic_latent_neg, semantic_latent_pos, isPositive=True, temperature=temperature)
            ctr_sem_neg_loss = model.ctr_semantic_loss_function(semantic_latent_pos, semantic_latent_neg, semantic_latent_neg, isPositive=False, temperature=temperature)
            ctr_sem_loss = ctr_sem_pos_loss + ctr_sem_neg_loss
            # 2.3 计算总的ctr_loss
            ctr_loss = ctr_truth_loss + ctr_sem_loss

            # 3.Edit Loss
            outputs_pos2neg, inputs_pos, semantic_latent_pos, truthful_latent_neg = model(batch_pos, truthful_latent_neg)
            outputs_neg2pos, inputs_neg, semantic_latent_neg, truthful_latent_pos = model(batch_neg, truthful_latent_pos)
            edit_loss = model.edit_loss_function(inputs_pos, inputs_neg, outputs_pos2neg, outputs_neg2pos)

            # 4.总损失
            total_loss = recon_loss + ctr_loss + edit_loss

            # 反向传播和优化
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            num_batches += 1

            train_loader_iter.set_postfix(loss=total_loss.item())

        train_losses.append(train_loss/num_batches)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss/num_batches:.4f}")

    # 定义预定参数
    args = argparse.Namespace(
        semantic_latent_dim=1024,
        truthful_latent_dim=1024,
        semantic_hidden_dims="2048",
        truthful_hidden_dims="2048",
        decoder_hidden_dims="2048"
    )
    # 保存预定参数与模型权重
    to_save = {
        'args': args,
        'state_dict': model.state_dict()
    }
    torch.save(to_save, "mistral_truthx_model_100epoch_500sample.pt")

    # 绘制train_loss曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", marker='o')
    plt.title("Train Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("mistral_truthx_model_100epoch_500sample.png", dpi=300)

