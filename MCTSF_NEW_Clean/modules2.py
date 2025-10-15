import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss

# --- TS2Vec의 인코더 아키텍처를 여기에 직접 정의합니다 ---
# from zhihanyue/ts2vec/ts2vec-main/models/dilated_conv.py
class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual

class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):
        return self.net(x)

# from zhihanyue/ts2vec/ts2vec-main/models/encoder.py
class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        
    def forward(self, x, mask=None):
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)
        
        if mask is None:
            mask = 'all_true'
        
        if mask == 'binomial':
            mask = torch.from_numpy(np.random.binomial(1, 0.5, size=(x.size(0), x.size(1)))).to(torch.bool)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)

        mask &= nan_mask.to(x.device)
        x[~mask] = 0
        
        x = x.transpose(1, 2)
        x = self.repr_dropout(self.feature_extractor(x))
        x = x.transpose(1, 2)
        
        return x

# --- 기존 모듈 ---
class EmbeddingDatabase:
    """FAISS를 사용하여 시계열 임베딩을 저장하고 검색하는 데이터베이스"""
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)

    def add(self, embeddings: np.ndarray):
        self.index.add(embeddings)

    def search(self, query_embeddings: np.ndarray, k: int):
        distances, indices = self.index.search(query_embeddings, k)
        return distances, indices
        
    def save(self, file_path):
        faiss.write_index(self.index, file_path)

    def load(self, file_path):
        self.index = faiss.read_index(file_path)

#class AlignmentModule(nn.Module):
#    """시계열 임베딩과 텍스트 임베딩을 정렬하는 모듈"""
#    def __init__(self, ts_embedding_dim, text_embedding_dim, output_dim):
#        super(AlignmentModule, self).__init__()
#        self.fc1 = nn.Linear(ts_embedding_dim + text_embedding_dim, 512)
#        self.fc2 = nn.Linear(512, output_dim)
#        self.relu = nn.ReLU()
#        self.dropout = nn.Dropout(0.2)

#    def forward(self, ts_embeddings, text_embedding):
#        mean_ts_embedding = ts_embeddings.mean(dim=1)
#        combined_embedding = torch.cat([mean_ts_embedding, text_embedding], dim=1)
#        x = self.dropout(self.relu(self.fc1(combined_embedding)))
#        aligned_embedding = self.fc2(x)
#        return aligned_embedding

class AlignmentModule(nn.Module):
    """
    Multi-head Attention을 사용하여 텍스트와 검색된 시계열 임베딩을 정렬하는 모듈.
    - Query: 텍스트 임베딩
    - Key/Value: Top-k 시계열 임베딩
    """
    def __init__(self, ts_embedding_dim, text_embedding_dim, n_heads=8, dropout=0.1):
        super(AlignmentModule, self).__init__()
        
        # Multi-head Attention은 Q, K, V의 임베딩 차원이 동일해야 합니다.
        # 따라서 텍스트 임베딩 차원을 기준으로 통일합니다.
        self.embed_dim = text_embedding_dim
        
        # 시계열 임베딩을 텍스트 임베딩 차원으로 변환하는 프로젝션 레이어
        self.ts_proj = nn.Linear(ts_embedding_dim, self.embed_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True  # (batch, seq, feature) 형태로 입력을 받도록 설정
        )
        
        # Transformer 블록처럼 Add & Norm, FeedForward 레이어를 추가하여 표현력을 높입니다.
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim * 4, self.embed_dim)
        )
        self.norm2 = nn.LayerNorm(self.embed_dim)

    def forward(self, ts_embeddings, text_embedding):
        # 입력 텐서 shape:
        # - ts_embeddings (Key, Value): (batch_size, k, ts_embedding_dim)
        # - text_embedding (Query): (batch_size, text_embedding_dim)
        
        # 1. Key, Value 준비: 시계열 임베딩을 어텐션 차원으로 프로젝션
        # (B, k, ts_emb_dim) -> (B, k, embed_dim)
        key_value = self.ts_proj(ts_embeddings)
        
        # 2. Query 준비: 텍스트 임베딩에 시퀀스 차원 추가
        # (B, text_emb_dim) -> (B, 1, embed_dim)
        query = text_embedding.unsqueeze(1)
        
        # 3. Multi-head Attention 수행
        # attn_output shape: (B, 1, embed_dim)
        attn_output, _ = self.attention(query=query, key=key_value, value=key_value)
        
        # 4. Add & Norm (Residual Connection)
        x = self.norm1(query + attn_output)
        
        # 5. FeedForward Network
        ff_output = self.ffn(x)
        x = self.norm2(x + ff_output)
        
        # 최종 출력에서 불필요한 시퀀스 차원 제거
        # (B, 1, embed_dim) -> (B, embed_dim)
        aligned_embedding = x.squeeze(1)
        
        return aligned_embedding