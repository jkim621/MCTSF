import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import os

from modules2 import TSEncoder, EmbeddingDatabase # TSEncoder로 변경
from dataset_forecasting import get_dataloader

# --- TS2Vec의 유틸리티 및 손실 함수를 여기에 직접 정의합니다 ---
# from zhihanyue/ts2vec/ts2vec-main/utils.py
def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]

# from zhihanyue/ts2vec/ts2vec-main/models/losses.py
def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d

def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)
    z = z.transpose(0, 1)
    sim = torch.matmul(z, z.transpose(1, 2))
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss

def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)
    sim = torch.matmul(z, z.transpose(1, 2))
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss

def pretrain_encoder(args, config, foldername):
    """TS2Vec 방식으로 인코더를 사전 학습하고 임베딩 DB를 생성"""
    print("Starting TS2Vec Pre-training process...")
    
    train_loader, _, _, _, _ = get_dataloader(
        datatype=args.datatype, device=args.device,
        batch_size=config["train"]["batch_size"], args=args
    )

    # 모델, 손실 함수, 옵티마이저 초기화
    encoder = TSEncoder(
        input_dims=1, # 단변량 데이터 기준
        output_dims=config["model"]["featureemb"], # featureemb와 동일하게 설정
        hidden_dims=128,
        depth=10
    ).to(args.device)
    
    optimizer = Adam(encoder.parameters(), lr=config.get("pretrain_lr", 1e-4))

    # 사전 학습 진행
    encoder.train()
    for epoch in range(config.get("pretrain_epochs", 10)):
        avg_loss = 0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.get('pretrain_epochs', 10)}") as it:
            for batch in it:
                optimizer.zero_grad()
                
                x = batch["observed_data"][:, :args.seq_len, :].to(args.device).float()
                
                # TS2Vec의 데이터 증강(크롭) 방식
                ts_l = x.size(1)
                crop_l = np.random.randint(low=2, high=ts_l + 1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
                
                # 두 개의 뷰 생성 및 인코딩
                out1 = encoder(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
                out1 = out1[:, -crop_l:]
                
                out2 = encoder(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
                out2 = out2[:, :crop_l]
                
                loss = hierarchical_contrastive_loss(out1, out2)
                loss.backward()
                optimizer.step()
                
                avg_loss += loss.item()
                it.set_postfix(ordered_dict={"loss": f"{avg_loss / len(it):.4f}"})

    pretrain_path = os.path.join(foldername, 'pretrained')
    os.makedirs(pretrain_path, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(pretrain_path, 'ts_encoder.pth'))
    print(f"Pre-trained encoder saved to: {os.path.join(pretrain_path, 'ts_encoder.pth')}")

    # 임베딩 데이터베이스 구축
    print("Building embedding database...")
    encoder.eval()
    db = EmbeddingDatabase(embedding_dim=config["model"]["featureemb"])
    all_embeddings = []

    with torch.no_grad():
        for batch in train_loader:
            x = batch["observed_data"][:, :args.seq_len, :].to(args.device).float()
            # 전체 시계열을 인코딩하고, 시간 축으로 Max pooling하여 대표 벡터 추출
            out = encoder(x)
            repr = F.max_pool1d(out.transpose(1, 2), kernel_size=out.size(1)).transpose(1, 2).squeeze(1)
            all_embeddings.append(repr.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    db.add(all_embeddings)
    db.save(os.path.join(pretrain_path, 'embedding_database.faiss'))
    print(f"Embedding database saved to: {os.path.join(pretrain_path, 'embedding_database.faiss')}")