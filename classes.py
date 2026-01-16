import os
import time
import math 
import torch.utils.data
from torch import nn
import torch.nn.functional as F

from CLIP_modules.modeling import CLIP4IDC
from CLIP_modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE


class BatchNormalize(torch.nn.Module):
    def __init__(self, mean, std, device):
        super().__init__()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, 3, 1, 1).to(device)

    def forward(self, tensor):
        # Tensor shape: [Batch, 3, H, W]
        return (tensor - self.mean) / self.std

class CLIPVisualEncoder(nn.Module):
    def __init__(self, args, clip_model_path):
        super().__init__()
        # Prepare model
        cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),"distributed")

        # 2. Checkpoint'i yükleyin
        model_state_dict = torch.load(clip_model_path, map_location="cpu")
        self.clip_model = CLIP4IDC.from_pretrained(
            args.cross_model,
            args.decoder_model,
            cache_dir=cache_dir,
            state_dict=model_state_dict,
            task_config=args,
        )
        
        self.visual_encoder = self.clip_model.clip.visual 

        # Modeli Float32 (Tam Hassasiyet) moduna zorla
        self.visual_encoder.float()
        self.visual_encoder.cuda()

        # 4. CLIP encoder (freeze) 
        
class CustomLayerNorm(nn.Module):
    def __init__(self, clip_dim=768, resnet_dim=1024):
        super().__init__()
        
        # --- ADIM 4 (Önceki adımın hazırlığı) ---
        # (Burada Conv2d + GELU tanımları var varsayıyoruz)
        
        # --- ADIM 5: Layer Norm Tanımları ---
        # LayerNorm'a sadece kanal sayısını veriyoruz (örneğin 512).
        # Bu, her pikseldeki (14x14) 512'lik vektörü kendi içinde normalize eder.
        self.ln_resnet = nn.LayerNorm(resnet_dim)
        self.ln_clip = nn.LayerNorm(clip_dim)

    def forward(self, g_feat, c_feat):
        # Girdi Boyutları (4. Adımdan gelen): 
        # g_feat -> [Batch, 1024, 14, 14]
        # c_feat -> [Batch, 49, 768] (Burada 49, 7x7 gridten gelmektedir)

        # --- ADIM 5 UYGULAMA ---

        # 1. ResNet Özellikleri için LayerNorm
        # [B, C, H, W] -> [B, H, W, C] (Kanalı en sona atıyoruz)
        g_feat = g_feat.permute(0, 2, 3, 1) 
        g_feat = self.ln_resnet(g_feat)      # Normalizasyon
        g_feat = g_feat.permute(0, 3, 1, 2)  # Tekrar [B, C, H, W] yapıyoruz

        # 2. CLIP Özellikleri için LayerNorm
        # Aynı işlem CLIP kolu için
        c_feat = self.ln_clip(c_feat)

        # Çıktılar şu an 6. Adım (Concat) için hazır.
        return g_feat, c_feat

class AdaptLayer(nn.Module):
    def __init__(self, clip_dim=768, resnet_dim=1024, target_size=14):
        super().__init__()
        
        self.target_size = target_size
        self.projection_dim = nn.Conv2d(1792, 1024, kernel_size=1)
        
        # Eğer CLIP çıktısı 7x7 ise (ViT-B/32), bunu 14x14'e büyütmek gerekebilir
        self.upsample = nn.Upsample(size=(target_size, target_size), mode='bilinear', align_corners=False)

        self.gate_fc = nn.Sequential(
            nn.Linear(resnet_dim * 3, 100), # Before + After feature concat
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid() # 0 ile 1 arası çıktı verir
        )

    def forward(self, resnet_feat_before, resnet_feat_after, clip_feat_before, clip_feat_after, soft=True):
        """
        resnet_feat: [Batch, 2048, 14, 14]
        clip_vec:    [Batch, 768] (Henüz 1x1 veya 14x14 değil)
        """

        # [Batch, 49, 768]
        # 2. Kare olup olmadığını kontrol edelim
        b, seq, dim = clip_feat_after.shape
        grid_size = int(math.sqrt(seq)) # 196 ise 14, 49 ise 7
        
        # 3. Sequence'ı Grid'e çevirme (Reshape & Permute)
        # Önce: [Batch, Dim, Seq] -> [Batch, Dim, H, W]
        clip_feat_after = clip_feat_after.permute(0, 2, 1)  # [Batch, 768, 49]
        clip_feat_after = clip_feat_after.view(b, dim, grid_size, grid_size) # [Batch, 768, 7, 7] (veya 7x7)
        clip_feat_before = clip_feat_before.permute(0, 2, 1)  # [Batch, 768, 49]
        clip_feat_before = clip_feat_before.view(b, dim, grid_size, grid_size) # [Batch, 768, 7, 7] (veya 7x7)

        # [Batch, 768, 14, 14]
        # 4. Eğer boyut 7x7 ise 14x14'e büyüt (ViT-B/32 kullanıyorsanız)
        if grid_size != self.target_size:
            clip_feat_before = self.upsample(clip_feat_before)
            clip_feat_after = self.upsample(clip_feat_after)

        if soft:
            res_pool_before = resnet_feat_before.mean([2, 3]) # [Batch, 1024]
            res_pool_after = resnet_feat_after.mean([2, 3])   # [Batch, 1024]
            
            # Fark vektörü (Pooling sonrası hesaplanmalı)
            res_diff_vec = torch.abs(res_pool_before - res_pool_after)

            # Gate'e düzleşmiş vektörleri veriyoruz
            gate_input = torch.cat([res_pool_before, res_pool_after, res_diff_vec], dim=1) # [Batch, 3072]

            # 3. GATE Mekanizması: Ne kadar değişim var?
            # ResNet fark vektörüne bakarak bir "alpha" katsayısı üret
            alpha = self.gate_fc(gate_input)
            # alpha output: [Batch, 1] -> Her görüntü için 0 (değişim yok) ile 1 (değişim var) arası.

            alpha = alpha.view(b, 1, 1, 1)
            
            resnet_feat_before = (1-alpha) * resnet_feat_before
            resnet_feat_after = (1-alpha) * resnet_feat_after
            clip_feat_before = alpha * clip_feat_before
            clip_feat_after = alpha * clip_feat_after

        final_before = torch.cat([resnet_feat_before, clip_feat_before], dim=1)
        final_after = torch.cat([resnet_feat_after, clip_feat_after], dim=1)

        final_before = F.normalize(final_before, p=2, dim=1)
        final_after = F.normalize(final_after, p=2, dim=1)

        final_before = self.projection_dim(final_before)
        final_after = self.projection_dim(final_after)

        return final_before, final_after

class AdaptLayerClip(nn.Module):
    def __init__(self, target_dim=1024):
        super().__init__()
        
        # 2. CLIP için Dönüşüm (768 -> 512)
        self.clip_adapt = nn.Sequential(
            nn.Conv2d(768, target_dim, kernel_size=1),
            nn.GELU()
        )

    def forward(self, clip_vec):
        """
        resnet_feat: [Batch, 2048, 14, 14]
        clip_vec:    [Batch, 768] (Henüz 1x1 veya 14x14 değil)
        """
        # --- CLIP KOLU (DİKKAT) ---
        # CLIP vektörü düz ([B, 768]) olduğu için Conv2d'ye girmeden önce
        # onu 4 boyutlu hale getirip genişletmeliyiz (3. Madde burada uygulanır).
        
        # 1. Boyut ekle: [Batch, 768, 1, 1]
        clip_grid = clip_vec.view(clip_vec.size(0), clip_vec.size(1), 1, 1)
        
        # 2. Kopyala (Broadcasting): [Batch, 768, 14, 14]
        clip_grid = clip_grid.expand(-1, -1, 14, 14)
        
        # 3. Adaptasyon katmanına sok
        # Girdi: [B, 768, 14, 14] -> Çıktı: [B, 512, 14, 14]
        c_feat = self.clip_adapt(clip_grid)
        
        return c_feat

class GatedSelfAttention(nn.Module):
    def __init__(self, d_model, n_head=8, dropout=0.1):
        super(GatedSelfAttention, self).__init__()
        
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        # Standart Q, K, V projeksiyonları
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

        # --- LINEAR + GLU KISMI ---
        # GLU giriş boyutunu yarıya indirdiği için, önce boyutu 2 katına çıkarıyoruz.
        # Bu kısım attention çıktısını "gate"lemek için kullanılır.
        self.linear_glu = nn.Linear(d_model, d_model * 2) 
        self.glu = nn.GLU(dim=-1) # Çıktı boyutu: d_model olur
        # --------------------------

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, clip_grid, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 1. Self-Attention Hesaplamaları
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        
        # Head'lere ayırma: (Batch, Seq, Head, Dim)
        Q = Q.view(batch_size, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        
        # Attention Score
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(x.device)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        
        # Weighted Sum
        out = torch.matmul(attention, V)
        
        # Boyutları geri düzeltme
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 2. LINEAR + GLU Uygulaması
        # Attention çıktısını bir "kapıdan" geçiriyoruz.
        # Residual connection (x + out) öncesi veya sonrası kullanılabilir, 
        # burada doğrudan attention çıktısına uygulanmış hali:
        gate_input = self.linear_glu(out) 
        gated_out = self.glu(gate_input) # Boyut tekrar d_model'e düşer
        
        # Residual + Norm
        output = self.layer_norm(x + gated_out)
        
        return output, attention


import logging
def get_logger(filename=None):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
    return logger
