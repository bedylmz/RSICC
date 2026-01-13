import os
import time
import json
import math 
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import argparse
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

from models import MCCFormers_diff_as_Q, DecoderTransformer, CNN_Encoder
from datasets import CaptionDataset
from utils import AverageMeter, adjust_learning_rate, bridge_embeddings_and_transfer, accuracy
from eval import evaluate_transformer

from CLIP_modules.modeling import CLIP4IDC
from CLIP_modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

# Sabit seed yerine o anki zamanı kullanarak rastgelelik sağlayın
seed = int(time.time())
torch.manual_seed(seed)

# Eğer GPU kullanıyorsanız oradaki rastgeleliği de serbest bırakın
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

import torch

class BatchNormalize(torch.nn.Module):
    def __init__(self, mean, std, device):
        super().__init__()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, 3, 1, 1).to(device)

    def forward(self, tensor):
        # Tensor shape: [Batch, 3, H, W]
        return (tensor - self.mean) / self.std

class CLIPVisualEncoder(nn.Module):
    def __init__(self, clip_model_path):
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
        for param in self.visual_encoder.parameters():
            param.requires_grad = False # veya True

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


def train(
    args= None,
    train_loader= None,
    clip_encoder_image= None,
    encoder_feat= None,
    decoder= None,
    criterion= None,
    encoder_image_optimizer= None,
    encoder_feat_optimizer= None,
    encoder_feat_lr_scheduler= None,
    decoder_optimizer= None,
    decoder_lr_scheduler= None,
    epoch= None,
    
    encoder_image = None,
    clip_encoder_optimizer = None,
    layerNormalizeLayer = None,
    adaptLayer = None,
    adaptLayerClip = None,
    encoder_image_lr_scheduler = None,
    gateSelf = None,

):
    global logger 
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    if(args.dual_branch ==True):
        encoder_image.train()
        adaptLayer.train()
        layerNormalizeLayer.train()
        if(args.gate ==True):
            gateSelf.train()
    else:
        adaptLayerClip.train()


    clip_encoder_image.eval()
    encoder_feat.train()
    decoder.train()  # train mode (dropout and batchnorm is used)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ResNet İstatistikleri
    norm_resnet = BatchNormalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225], 
                                 device=device)
    
    # CLIP İstatistikleri
    norm_clip = BatchNormalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                               std=[0.26862954, 0.26130258, 0.27577711], 
                               device=device)
    # -----------------------------------------------

    for i, (img_pairs, caps, caplens) in enumerate(train_loader):

        data_time.update(time.time() - start)

        # Back prop.
        decoder_optimizer.zero_grad()
        encoder_feat_optimizer.zero_grad()

        if(args.dual_branch == True):
            encoder_image_optimizer.zero_grad()

        # Move to GPU, if available
        img_pairs = img_pairs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs_A = img_pairs[:, 0, :, :, :]
        imgs_B = img_pairs[:, 1, :, :, :]

        
        if(args.dual_branch == True ):
            b, t, c, h, w = img_pairs.shape
            imgs_full = img_pairs.view(-1, c, h, w) 
            imgs_full_clip = norm_clip(imgs_full) # CLIP için normalize et

            # 2. Pass the flattened pairs and set frames to 2
            # Note: Remove parentheses from .shape (it is a property, not a function)
            clip_out = clip_encoder_image(imgs_full_clip, 2) # 768 100 b
            size = clip_out.size(1)//2
            clip_out_A = clip_out[:,1:size,:] # 768 1 b
            clip_out_B = clip_out[:,size+1:,:]

            imgs_A_resnet = norm_resnet(imgs_A) # ResNet için normalize et
            imgs_B_resnet = norm_resnet(imgs_B)
            
            resnet_A = encoder_image(imgs_A_resnet)
            resnet_B = encoder_image(imgs_B_resnet)

            resnet_A_normed, clip_A_normed = layerNormalizeLayer(resnet_A, clip_out_A)
            resnet_B_normed, clip_B_normed = layerNormalizeLayer(resnet_B, clip_out_B)          

            final_A, final_B = adaptLayer(resnet_A_normed, resnet_B_normed, clip_A_normed, clip_B_normed, soft=args.soft)

            # train fonksiyonu içinde (satır 194 civarı)
            # Girdi: [Batch, 512, 14, 14]
            if(args.gate ==True and 0 == 1):
                # 1. Kanalı sona alıp düzleştirin: [Batch, 196, 512]
                b, c, h, w = resnet_A_adapt.shape
                resnet_A_flat = resnet_A_adapt.permute(0, 2, 3, 1).view(b, h*w, c) 

                # 2. Attention uygulayın (Çıktı yine [Batch, 196, 512] olacak)
                resnet_A_att, _ = gateSelf(resnet_A_flat)

                # 3. Tekrar [Batch, 512, 14, 14] formatına dönün (Concat için gerekli)
                resnet_A_adapt = resnet_A_att.view(b, h, w, c).permute(0, 3, 1, 2)

                b, c, h, w = resnet_B_adapt.shape
                resnet_B_flat = resnet_B_adapt.permute(0, 2, 3, 1).view(b, h*w, c) 

                # 2. Attention uygulayın (Çıktı yine [Batch, 196, 512] olacak)
                resnet_B_att, _ = gateSelf(resnet_B_flat)

                # 3. Tekrar [Batch, 512, 14, 14] formatına dönün (Concat için gerekli)
                resnet_B_adapt = resnet_B_att.view(b, h, w, c).permute(0, 3, 1, 2)

            fused_feat = encoder_feat(
                final_A,
                final_B,
            ) # encoder_out: (S, batch, feature_dim) # fused_feat: (S, batch, feature_dim) # buyuk tensor atama yavaslatior (#batch time = 0.5)
        elif(args.fusedclip):
            b, t, c, h, w = img_pairs.shape
            imgs_full = img_pairs.view(-1, c, h, w) 
            imgs_full_clip = norm_clip(imgs_full) # CLIP için normalize et

            # 2. Pass the flattened pairs and set frames to 2
            # Note: Remove parentheses from .shape (it is a property, not a function)
            clip_out = clip_encoder_image(imgs_full_clip, 2)
            size = clip_out.size(1)//2
            clip_out_A = clip_out[:,1:size,:] # 768 1 b
            clip_out_B = clip_out[:,size+1:,:]

            clip_out_A = adaptLayerClip(clip_out_A)
            clip_out_B = adaptLayerClip(clip_out_B)


            imgs_A_resnet = norm_resnet(imgs_A) # ResNet için normalize et
            imgs_B_resnet = norm_resnet(imgs_B)
            
            resnet_A = encoder_image(imgs_A_resnet)
            resnet_B = encoder_image(imgs_B_resnet)

            final_A = torch.cat([resnet_A_normed, clip_A_normed], dim=1)
            final_B = torch.cat([resnet_B_normed, clip_B_normed], dim=1)

            fused_feat = encoder_feat(
                resnet_A,
                resnet_B,
            ) # encoder_out: (S, batch, feature_dim) # fused_feat: (S, batch, feature_dim) # buyuk tensor atama yavaslatior (#batch time = 0.5)

            clip_out =  torch.cat([clip_out_A, clip_out_B], dim=1)
            clip_out =  clip_out.permute(1,2,0)

            fused_feat = torch.cat([fused_feat, clip_out], dim=1)

            
        else:
            b, t, c, h, w = img_pairs.shape
            imgs_full = img_pairs.view(-1, c, h, w) 
            imgs_full_clip = norm_clip(imgs_full)
            # 2. Pass the flattened pairs and set frames to 2
            # Note: Remove parentheses from .shape (it is a property, not a function)
            clip_out = clip_encoder_image(imgs_full_clip, 2)
            clip_out_A = clip_out[:,0,:] # 768 100 b
            clip_out_B = clip_out[:,50,:]
            clip_out_A = adaptLayerClip(clip_out_A)
            clip_out_B = adaptLayerClip(clip_out_B)

            fused_feat = encoder_feat(
                clip_out_A,
                clip_out_B,
            ) # encoder_out: (S, batch, feature_dim) # fused_feat: (S, batch, feature_dim) # buyuk tensor atama yavaslatior (#batch time = 0.5)

        scores, caps_sorted, decode_lengths, sort_ind = decoder(fused_feat, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss
        loss = criterion(scores, targets)

        loss.backward()

        # Update weights
        decoder_optimizer.step()
        decoder_lr_scheduler.step()
        
        encoder_feat_optimizer.step()
        encoder_feat_lr_scheduler.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 1)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()
        if i % args.print_freq == 0:
            # logger.info('TIME: ', time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
            logger.info(
                "Epoch: {}/{} step: {}/{} Loss: {} AVG_Loss: {} Top-5 Accuracy: {} Batch_time: {}s".format(
                    epoch + 0,
                    args.epochs,
                    i + 0,
                    len(train_loader),
                    losses.val,
                    losses.avg,
                    top5accs.val,
                    batch_time.val,
                )
            )

def key_transformation(old_key):
    global logger 
    if old_key == "layer.0.weight":
        return "layer.1.weight"

    return old_key

def prep_optimizer(args, model, device, num_train_optimization_steps, coef_lr=1.0):
    global logger 
    if hasattr(model, "module"):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [
        (n, p) for n, p in decay_param_tp
        if "clip." in n
        and not any(exclude in n for exclude in [
            "clip.visual.ln_mid",
            "clip.visual.joint_positional_embedding",
            "clip.visual.bef_embedding",
            "clip.visual.aft_embedding"
        ])
    ]
    decay_noclip_param_tp = [
        (n, p) for n, p in decay_param_tp
        if any(include in n for include in [
            "clip.visual.ln_mid",
            "clip.visual.joint_positional_embedding",
            "clip.visual.bef_embedding",
            "clip.visual.aft_embedding"
        ])
    ]

    no_decay_clip_param_tp = [
        (n, p) for n, p in no_decay_param_tp
        if "clip." in n
        and not any(exclude in n for exclude in [
            "clip.visual.ln_mid",
            "clip.visual.joint_positional_embedding",
            "clip.visual.bef_embedding",
            "clip.visual.aft_embedding"
        ])
    ]
    no_decay_noclip_param_tp = [
        (n, p) for n, p in no_decay_param_tp
        if any(include in n for include in [
            "clip.visual.ln_mid",
            "clip.visual.joint_positional_embedding",
            "clip.visual.bef_embedding",
            "clip.visual.aft_embedding"
        ])
    ]

    weight_decay = 0.2
    optimizer_grouped_parameters = [
        {
            "params": [p for _, p in decay_clip_param_tp],
            "weight_decay": weight_decay,
            "lr": args.clip_encoder_lr * coef_lr,
        },
        {
            "params": [p for _, p in decay_noclip_param_tp],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for _, p in no_decay_clip_param_tp],
            "weight_decay": 0.0,
            "lr": args.clip_encoder_lr * coef_lr,
        },
        {
            "params": [p for _, p in no_decay_noclip_param_tp],
            "weight_decay": 0.0,
        }
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, 
                         lr=args.clip_encoder_lr, 
                         warmup=args.warmup_proportion,
                         schedule='warmup_cosine',
                         b1=0.9,
                         b2=0.98,
                         e=1e-6,
                         t_total=num_train_optimization_steps,
                         weight_decay=weight_decay,
                         max_grad_norm=1.0)
    model.to(device)
    return optimizer, scheduler, model

def save_checkpoint(args= None, 
                    data_name= None, 
                    epoch= None, 
                    epochs_since_improvement= None,

                    clip_encoder_image= None,
                    encoder_feat= None,
                    decoder= None,
                    encoder_image_optimizer= None,
                    encoder_feat_optimizer= None,
                    decoder_optimizer= None,
                    
                    encoder_image = None,
                    clip_encoder_optimizer = None,
                    layerNormalizeLayer = None,
                    adaptLayer = None,
                    adaptLayerClip = None,
                    encoder_image_lr_scheduler = None,
                    gateSelf= None,
                    ):
        
    global logger 
    """
    Model checkpoint'ini kaydeder.
    
    Önemli: clip_encoder_image (CustomCLIPVisualEncoder) içindeki 
    hem visual_encoder hem de projection katmanlarını tek seferde kaydeder.
    """
    state = {
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        
        # --- Modeller ---
        # MCCFormers Feature Encoder
        'encoder_feat': encoder_feat.state_dict(),
        # Transformer Decoder
        'decoder': decoder.state_dict(),
        # BİZİM İÇİN EN ÖNEMLİ KISIM:
        # Wrapper sınıfının state_dict'i hem CLIP ağırlıklarını hem Projection katmanını içerir.
        'clip_encoder_image': clip_encoder_image.state_dict(),
        
        # --- Optimizerlar ---
        'encoder_feat_optimizer': encoder_feat_optimizer.state_dict(),
        'decoder_optimizer': decoder_optimizer.state_dict(),
    }
    # Eski ResNet encoder varsa (Opsiyonel)
    if encoder_image is not None:
        state['encoder_image'] = encoder_image.state_dict()
    if encoder_image_optimizer is not None:
        state['encoder_image_optimizer'] = encoder_image_optimizer.state_dict()

    if layerNormalizeLayer is not None:
        state['layerNormalizeLayer'] = layerNormalizeLayer.state_dict()
    if adaptLayer is not None:
        state['adaptLayer'] = adaptLayer.state_dict()
    if adaptLayerClip is not None:
        state['adaptLayerClip'] = adaptLayerClip.state_dict()
    if clip_encoder_optimizer is not None:
        state['clip_encoder_optimizer'] = clip_encoder_optimizer.state_dict()
    if encoder_image_lr_scheduler is not None:
        state['encoder_image_lr_scheduler'] = encoder_image_lr_scheduler.state_dict()
    if gateSelf is not None:
        state['gateSelf'] = gateSelf.state_dict()

    # Kayıt Dizini Kontrolü
    directory = args.save_model_path
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 1. En Son Checkpoint'i Kaydet (Her epoch'ta üzerine yazar)
    filename = os.path.join(directory, 'checkpoint_' + data_name + '.pth.tar')
    torch.save(state, filename)

#suan kullanilmiyor
def validate_loss(val_loader, encoder_image, clip_encoder_image, encoder_feat, decoder, criterion):
    global logger 
    
    """
    Validation seti üzerinde sadece Loss hesabı yapar.
    """
    # Modelleri eval moduna al (Dropout ve BatchNorm'u kapatır)
    # encoder_image.eval() # Eğer kullanılıyorsa
    clip_encoder_image.eval()
    encoder_feat.eval()
    decoder.eval()

    losses = AverageMeter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad(): # Gradient hesabı yapma
        for i, (img_pairs, caps, caplens, *_) in enumerate(val_loader):
            # Veriyi GPU'ya taşı
            img_pairs = img_pairs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            imgs_A = img_pairs[:, 0, :, :, :]
            imgs_B = img_pairs[:, 1, :, :, :]

            clip_imgs_A = clip_encoder_image(imgs_A)
            clip_imgs_B = clip_encoder_image(imgs_B)

            fused_feat = encoder_feat(clip_imgs_A, clip_imgs_B)

            scores, caps_sorted, decode_lengths, sort_ind = decoder(fused_feat, caps, caplens)

            # Hedefleri ayarla
            targets = caps_sorted[:, 1:]
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Loss hesapla
            loss = criterion(scores, targets)
            losses.update(loss.item(), sum(decode_lengths))
            
    return losses.avg

def print_trainable_parameters(models_dict):
    global logger 

    """
    Verilen model sözlüğündeki eğitilen (gradient hesaplanan) parametreleri ve
    toplam eğitilebilir parametre sayısını yazdırır.
    
    Kullanım:
    models = {
        "ResNet": encoder_image,
        "CLIP": clip_encoder_image,
        "MCCFormers": encoder_feat,
        "Decoder": decoder,
        "AdaptLayer": adaptLayer,  # Eğer tanımlıysa
        "LayerNorm": layerNormalizeLayer, # Eğer tanımlıysa
        "Gate": gateSelf # Eğer tanımlıysa
    }
    print_trainable_parameters(models)
    """
    logger.info("\n" + "="*50)
    logger.info("EĞİTİLEN KATMANLAR VE PARAMETRE SAYILARI")
    logger.info("="*50)
    
    total_params = 0
    
    for model_name, model in models_dict.items():
        if model is None:
            continue
            
        logger.info(f"Model: {model_name}")
        logger.info("-" * 20)
        
        model_params = 0
        trainable_layers = set()
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Parametre sayısını topla
                num_params = param.numel()
                model_params += num_params
                
                # Katman ismini sadeleştirip listeye ekle (örn: encoder.layer1.weight -> encoder.layer1)
                layer_name = ".".join(name.split(".")[:-1])
                # Detaylı yazdırmak isterseniz alttaki satırı açın:
                # logger.info(f"  - {name} ({num_params:,} parametre)")
                trainable_layers.add(layer_name)
        
        logger.info(f"  > Eğitilen Toplam Parametre: {model_params:,}")
        if model_params > 0:
            logger.info("  > Örnek Eğitilen Katmanlar:")
            # İlk 5 eğitilen katmanı örnek olarak yazdır
            for i, layer in enumerate(list(trainable_layers)[:5]):
                logger.info(f"    * {layer}")
            if len(trainable_layers) > 5:
                logger.info(f"    ... ve {len(trainable_layers) - 5} katman daha.")
        else:
            logger.info("  > DİKKAT: Bu modelde eğitilen parametre yok (Tamamen dondurulmuş).")
            
        total_params += model_params

    logger.info("="*50)
    logger.info(f"TOPLAM EĞİTİLEBİLİR PARAMETRE SAYISI: {total_params:,}")
    logger.info("="*50 + "\n")


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


def main(args):
    global logger 
    logger = get_logger(os.path.join(args.save_model_path, "log.txt"))
    logger.info(f"Kullanılan Random Seed: {seed}") # İleride gerekirse tekrar üretmek için
    if args.eval_mode:
        logging.disable(logging.CRITICAL)

    logger.info(args)
    global metrics_list

    start_epoch = 0
    best_bleu4 = 0.0  # BLEU-4 score right now
    epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(device)

    cudnn.benchmark = (
        True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
    )

    # Read word map
    word_map_file = os.path.join(args.data_folder, "WORDMAP_" + args.data_name + ".json")
    with open(word_map_file, "r") as j:
        word_map = json.load(j)
    
    # ------------------------------ CLIP ENTEGRASYONU ------------------------------

    if(args.eval_just_RSICC == False):

        clip = CLIPVisualEncoder(args.clip_path)

        clip_encoder_image = clip.visual_encoder.float()
        clip_encoder_image = clip_encoder_image.cuda()

        clip_encoder_image.eval()

    if(args.dual_branch == True):
        adaptLayer = AdaptLayer()
        adaptLayer = adaptLayer.cuda()
        layerNormalizeLayer = CustomLayerNorm()
        layerNormalizeLayer = layerNormalizeLayer.cuda()
        if(args.gate ==True):
            gateSelf = GatedSelfAttention(512)
            gateSelf = gateSelf.cuda()
    else:
        adaptLayerClip = AdaptLayerClip() 
        adaptLayerClip = adaptLayerClip.cuda()
    
    # ------------------------------ CLIP ENTEGRASYONU ------------------------------

    # Initialize
    # Encoder
    if(args.dual_branch == True or args.eval_just_RSICC == True):
        encoder_image = CNN_Encoder(NetType=args.encoder_image, method=args.decoder)
        # set the encoder_dim
    encoder_image_dim = 1024 # resnet101

    if args.encoder_feat == "MCCFormers_diff_as_Q":
        encoder_feat = MCCFormers_diff_as_Q(
            feature_dim=encoder_image_dim,
            dropout=0.5,
            h=14,
            w=14,
            d_model=512,
            n_head=args.n_heads,
            n_layers=args.n_layers,
        )


    # Decoder
    args.feature_dim_de = 1024 # 当有concat是1024,否则为512
    if args.decoder == "trans":
        decoder = DecoderTransformer(
            feature_dim=args.feature_dim_de,
            vocab_size=len(word_map),
            n_head=args.n_heads,
            n_layers=args.decoder_n_layers,
            dropout=args.dropout,
        )

    if(args.dual_branch == True):
        #! we will not train encoder image
        encoder_image_optimizer = (
            torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder_image.parameters()), lr=args.encoder_lr)
            if args.fine_tune_encoder
            else None
        )

    if args.checkpoint is not "None":
        filename = os.listdir(args.checkpoint)
        checkpoint_path = os.path.join(args.checkpoint, filename[0])
        checkpoint = torch.load(checkpoint_path, map_location=str(device))

    if(args.dual_branch == True):
        encoder_image_lr_scheduler = (
            StepLR(encoder_image_optimizer, step_size=900, gamma=1) if args.fine_tune_encoder else None
        )
    
    params_to_optimize = list(filter(lambda p: p.requires_grad, encoder_feat.parameters()))

    if args.dual_branch:
        # Dual branch ise bu katmanları ekle
        params_to_optimize += list(adaptLayer.parameters())
        params_to_optimize += list(layerNormalizeLayer.parameters())
        if(args.gate ==True):
            params_to_optimize += list(gateSelf.parameters())
    else:
        # Tek branch ise sadece bunu ekle
        params_to_optimize += list(adaptLayerClip.parameters())

    # Optimizer'ı bu birleşik liste ile oluşturun
    encoder_feat_optimizer = torch.optim.Adam(
        params=params_to_optimize, 
        lr=args.encoder_lr
    )

    encoder_feat_lr_scheduler = StepLR(encoder_feat_optimizer, step_size=900, gamma=1)

    decoder_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=args.decoder_lr
    )
    decoder_lr_scheduler = StepLR(decoder_optimizer, step_size=900, gamma=1)

    # Move to GPU, if available
    if(args.dual_branch == True or args.eval_just_RSICC == True):
        encoder_image = encoder_image.to(device)
    encoder_feat = encoder_feat.to(device)
    decoder = decoder.to(device)
    
    logger.info("Checkpoint_savepath:{}".format(args.savepath))
    logger.info(
        "Encoder_image_mode:{}   Encoder_feat_mode:{}   Decoder_mode:{}".format(
            args.encoder_image_model, args.encoder_feat, args.decoder
        )
    )
    logger.info(
        "encoder_layers {} decoder_layers {} n_heads {} dropout {} encoder_lr {} "
        "decoder_lr {}".format(
            args.n_layers, args.decoder_n_layers, args.n_heads, args.dropout, args.encoder_lr, args.decoder_lr
        )
    )

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)

    # Custom dataloaders
    # Burada SADECE boyutlandırma yapıyoruz. Normalizasyonu döngü içine taşıdık.
    # CLIP genelde Bicubic sever, ResNet de buna uyum sağlar.
    train_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC)
    ])

    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(args.data_folder, args.data_name, "TRAIN", 
                    transform=train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    #------------------------ TEXT ENCODER ENTEGRASYONU ----------------

    if(args.clip_text_encoder):
        from CLIP_modules.tokenization_clip import SimpleTokenizer

        clip_tokenizer = SimpleTokenizer()
        clip_model_ref = clip.clip_model.clip
        
        # Köprü fonksiyonunu çalıştır
        bridge_embeddings_and_transfer(
            rsicc_decoder=decoder, 
            clip_model=clip_model_ref, 
            clip_tokenizer=clip_tokenizer, 
            rsicc_word_map=word_map
        )
        
    #------------------------ TEXT ENCODER ENTEGRASYONU ----------------

    # Eğitilenleri kontrol etmek için sözlük oluştur
    check_models = {
        "ResNet (Encoder)": encoder_image if args.fine_tune_encoder else None,
        "CLIP (Visual)": clip_encoder_image,
        "MCCFormers (Feat)": encoder_feat,
        "Decoder": decoder,
    }
    
    # Dual branch modülleri varsa ekle
    if args.dual_branch:
        check_models["AdaptLayer"] = adaptLayer
        check_models["LayerNorm"] = layerNormalizeLayer
        if args.gate:
            check_models["GatedSelfAttention"] = gateSelf
    else:
        check_models["AdaptLayerClip"] = adaptLayerClip

    # Fonksiyonu çağır
    print_trainable_parameters(check_models)


    if(args.eval_mode == False):
        # Epochs
        for epoch in range(start_epoch, args.epochs):
            
            if (args.dual_branch == True):
                if(args.gate ==True):
                    train(
                        args,
                        train_loader=train_loader,
                        encoder_image=encoder_image,
                        clip_encoder_image=clip_encoder_image,
                        encoder_feat=encoder_feat,
                        decoder=decoder,
                        criterion=criterion,
                        encoder_image_optimizer=encoder_image_optimizer,
                        encoder_image_lr_scheduler=encoder_image_lr_scheduler,
                        encoder_feat_optimizer=encoder_feat_optimizer,
                        encoder_feat_lr_scheduler=encoder_feat_lr_scheduler,
                        decoder_optimizer=decoder_optimizer,
                        decoder_lr_scheduler=decoder_lr_scheduler,
                        epoch=epoch,
                        adaptLayer= adaptLayer,
                        layerNormalizeLayer=layerNormalizeLayer,
                        gateSelf=gateSelf,
                    )
                else:
                    train(
                        args,
                        train_loader=train_loader,
                        encoder_image=encoder_image,
                        clip_encoder_image=clip_encoder_image,
                        encoder_feat=encoder_feat,
                        decoder=decoder,
                        criterion=criterion,
                        encoder_image_optimizer=encoder_image_optimizer,
                        encoder_image_lr_scheduler=encoder_image_lr_scheduler,
                        encoder_feat_optimizer=encoder_feat_optimizer,
                        encoder_feat_lr_scheduler=encoder_feat_lr_scheduler,
                        decoder_optimizer=decoder_optimizer,
                        decoder_lr_scheduler=decoder_lr_scheduler,
                        epoch=epoch,
                        adaptLayer= adaptLayer,
                        layerNormalizeLayer=layerNormalizeLayer,
                    )
            else:
                train(
                    args,
                    train_loader=train_loader,
                    clip_encoder_image=clip_encoder_image,
                    encoder_feat=encoder_feat,
                    decoder=decoder,
                    criterion=criterion,
                    encoder_feat_optimizer=encoder_feat_optimizer,
                    encoder_feat_lr_scheduler=encoder_feat_lr_scheduler,
                    decoder_optimizer=decoder_optimizer,
                    decoder_lr_scheduler=decoder_lr_scheduler,
                    epoch=epoch,
                    adaptLayerClip= adaptLayerClip,
                )
            if(args.dual_branch == True):
              if(args.gate == True):
                metrics = evaluate_transformer(
                        args, 
                        encoder_image=encoder_image, 
                        clip_encoder_image=clip_encoder_image, 
                        encoder_feat=encoder_feat,
                        decoder=decoder,
                        layerNormalizeLayer=layerNormalizeLayer,
                        adaptLayer=adaptLayer,
                        gateSelf=gateSelf,
                        logger=logger,
                        )
              else:
                  metrics = evaluate_transformer(
                        args, 
                        encoder_image=encoder_image, 
                        clip_encoder_image=clip_encoder_image, 
                        encoder_feat=encoder_feat,
                        decoder=decoder,
                        layerNormalizeLayer=layerNormalizeLayer,
                        adaptLayer=adaptLayer,
                        logger=logger,
                        )
            else:
              metrics = evaluate_transformer(
                    args, 
                    clip_encoder_image=clip_encoder_image, 
                    encoder_feat=encoder_feat,
                    decoder=decoder,
                    adaptLayerClip=adaptLayerClip,
                    logger=logger,
                    )

            # -----------------------------------------------------------------------------------------------------
            # One epoch's validation
            logger.info("-------------------------epoch passed-------------------------")

            recent_bleu4 = metrics["Bleu_4"]
            
            # Check if there was an improvement
            is_best = recent_bleu4 > best_bleu4
            best_bleu4 = max(recent_bleu4, best_bleu4)
            if not is_best:
                epochs_since_improvement += 1
                logger.info(" Epochs since last improvement: %d\n" % (epochs_since_improvement,))
            else:
                epochs_since_improvement = 0
            if is_best:
                logger.info("-------------------------checkpoint Saved-------------------------")
                # Save checkpoint
                if(args.dual_branch == True):
                    if(args.gate == True):
                        save_checkpoint(args,
                                        "SecondCC",
                                        epoch = epoch,
                                        epochs_since_improvement = epochs_since_improvement, 
                                        encoder_image = encoder_image,
                                        encoder_feat=encoder_feat,
                                        decoder=decoder, 
                                        encoder_image_optimizer=encoder_image_optimizer,
                                        encoder_feat_optimizer=encoder_feat_optimizer,
                                        decoder_optimizer=decoder_optimizer, 
                                        clip_encoder_image=clip_encoder_image,
                                        adaptLayer=adaptLayer,
                                        layerNormalizeLayer=layerNormalizeLayer,
                                        gateSelf=gateSelf,
                                        )
                    else:
                        save_checkpoint(args,
                                        "SecondCC",
                                        epoch = epoch,
                                        epochs_since_improvement = epochs_since_improvement, 
                                        encoder_image = encoder_image,
                                        encoder_feat=encoder_feat,
                                        decoder=decoder, 
                                        encoder_image_optimizer=encoder_image_optimizer,
                                        encoder_feat_optimizer=encoder_feat_optimizer,
                                        decoder_optimizer=decoder_optimizer, 
                                        clip_encoder_image=clip_encoder_image,
                                        adaptLayer=adaptLayer,
                                        layerNormalizeLayer=layerNormalizeLayer,
                                        )
                else:
                    save_checkpoint(args,
                                    "SecondCC",
                                    epoch = epoch,
                                    epochs_since_improvement = epochs_since_improvement, 
                                    encoder_feat=encoder_feat,
                                    decoder=decoder, 
                                    encoder_feat_optimizer=encoder_feat_optimizer,
                                    decoder_optimizer=decoder_optimizer, 
                                    clip_encoder_image=clip_encoder_image,
                                    adaptLayerClip=adaptLayerClip,
                                    )
                
            # Early Stopping
            if epochs_since_improvement == args.stop_criteria:
                logger.info(f"Early stopping triggered! Validation metrics hasn't increased for {args.stop_criteria} epochs.")
                break
            if epochs_since_improvement > 0 and epochs_since_improvement % 3 == 0:
                adjust_learning_rate(decoder_optimizer, 0.7)

    # ---------------------------- EVAL SECTION ----------------------------
    else:
        logging.disable(logging.NOTSET)
        logger.info(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=str(device))
        
        if 'encoder_image' in checkpoint:
            encoder_image.load_state_dict(checkpoint['encoder_image'])
            logger.info("Loaded 'encoder_image' weights from checkpoint.")
        
        if 'encoder_feat' in checkpoint:
            encoder_feat.load_state_dict(checkpoint['encoder_feat'])
            logger.info("Loaded 'encoder_feat' weights from checkpoint.")
            
        if 'decoder' in checkpoint:
            decoder.load_state_dict(checkpoint['decoder'])
            logger.info("Loaded 'decoder' weights from checkpoint.")
        
        if 'layerNormalizeLayer' in checkpoint:
            layerNormalizeLayer.load_state_dict(checkpoint['layerNormalizeLayer'])
            logger.info("Loaded 'layerNormalizeLayer' weights from checkpoint.")

        if 'adaptLayer' in checkpoint:
            adaptLayer.load_state_dict(checkpoint['adaptLayer'])
            logger.info("Loaded 'adaptLayer' weights from checkpoint.")
        
        if 'adaptLayerClip' in checkpoint:
            adaptLayerClip.load_state_dict(checkpoint['adaptLayerClip'])
            logger.info("Loaded 'adaptLayerClip' weights from checkpoint.")

        if 'gateSelf' in checkpoint:
            gateSelf.load_state_dict(checkpoint['gateSelf'])
            logger.info("Loaded 'gateSelf' weights from checkpoint.")

        # Check for CLIP specifically
        if 'clip_encoder_image' in checkpoint:
            clip_encoder_image.load_state_dict(checkpoint['clip_encoder_image'])
            logger.info("Loaded 'clip_encoder_image' weights from checkpoint.")
        else:
            logger.warning("!!!!!!!!!!!!!!!!!     WARNING        !!!!!!!!!!!!!!!!!")
            logger.warning("No 'clip_encoder_image' weights found in checkpoint.")

        #------------------------ TEXT ENCODER ENTEGRASYONU ----------------
        if(args.clip_text_encoder):
            from CLIP_modules.tokenization_clip import SimpleTokenizer

            clip_tokenizer = SimpleTokenizer()
            clip_model_ref = clip_encoder_image.clip_model.clip
            
            # Köprü fonksiyonunu çalıştır
            bridge_embeddings_and_transfer(
                rsicc_decoder=decoder, 
                clip_model=clip_model_ref, 
                clip_tokenizer=clip_tokenizer, 
                rsicc_word_map=word_map
            )
        #------------------------ TEXT ENCODER ENTEGRASYONU ----------------

        if(args.dual_branch):
              if(args.gate):
                metrics = evaluate_transformer(
                        args, 
                        encoder_image=encoder_image, 
                        clip_encoder_image=clip_encoder_image, 
                        encoder_feat=encoder_feat,
                        decoder=decoder,
                        layerNormalizeLayer=layerNormalizeLayer,
                        adaptLayer=adaptLayer,
                        gateSelf= gateSelf,
                        logger=logger,
                        )
              else:
                   metrics = evaluate_transformer(
                        args, 
                        encoder_image=encoder_image, 
                        clip_encoder_image=clip_encoder_image, 
                        encoder_feat=encoder_feat,
                        decoder=decoder,
                        layerNormalizeLayer=layerNormalizeLayer,
                        adaptLayer=adaptLayer,
                        logger=logger,
                        )
        elif(args.eval_just_RSICC):
            metrics = evaluate_transformer(
                        args, 
                        encoder_image=encoder_image, 
                        encoder_feat=encoder_feat,
                        decoder=decoder,
                        logger=logger,
                        )
        else:
              metrics = evaluate_transformer(
                    args, 
                    clip_encoder_image=clip_encoder_image, 
                    encoder_feat=encoder_feat,
                    decoder=decoder,
                    adaptLayerClip=adaptLayerClip,
                    logger=logger)

if __name__ == "__main__":
    folder_path = ""
   
    parser = argparse.ArgumentParser(description="Image_Change_Captioning")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Data parameters
    parser.add_argument("--data_folder", default="")
    parser.add_argument("--data_name", default="LEVIR_CC_5_cap_per_img_10_min_word_freq", help="base name shared by data files.")
    
    # Model parameters
    parser.add_argument('--encoder_image', default="resnet101", help='which model does encoder use?')
    parser.add_argument("--encoder_image_model", default="clip4IDC", help="which model does encoder use?")
    parser.add_argument("--encoder_feat", default="MCCFormers_diff_as_Q")
    parser.add_argument("--decoder", default="trans")
    parser.add_argument("--n_heads", type=int, default=8, help="Multi-head attention in Transformer.")
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--decoder_n_layers", type=int, default=1)
    parser.add_argument("--feature_dim_de", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout")
    parser.add_argument("--eval_mode", action='store_true')
    parser.add_argument("--checkpoint_path", type=str, default="")

    #params for CLIP4IDC implementation
    parser.add_argument("--do_pretrain", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--dataloader_type", type=str, default="test")

    parser.add_argument("--soft", action="store_true")


    parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate")
    parser.add_argument("--lr_decay", type=float, default=0.9, help="Learning rate exp epoch decay")
    parser.add_argument("--n_display", type=int, default=100, help="Information display frequence")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--max_words", type=int, default=20, help="")
    parser.add_argument("--feature_framerate", type=int, default=1, help="")
    parser.add_argument("--margin", type=float, default=0.1, help="margin for loss")
    parser.add_argument("--hard_negative_rate", type=float, default=0.5, help="rate of intra negative sample")
    parser.add_argument("--negative_weighting", type=int, default=1, help="Weight the loss for intra negative")
    parser.add_argument("--n_pair", type=int, default=1, help="Num of pair to output from data loader")

    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--decoder_model", default="decoder-base", type=str, required=False, help="Decoder module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")

    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,help="Number of updates steps to accumulate before performing a " "backward/update pass.",)

    parser.add_argument("--cache_dir",default="",type=str,help="Where do you want to store the pre-trained models downloaded " "from s3",)

    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")
    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument("--coef_lr", type=float, default=1.0, help="coefficient for bert branch.")
    parser.add_argument("--use_mil", action="store_true", help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument("--sampled_use_mil", action="store_true", help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument("--text_num_hidden_layers", type=int, default=12, help="Layer NO. of text.")
    parser.add_argument("--visual_num_hidden_layers", type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument("--intra_num_hidden_layers", type=int, default=9, help="Layer NO. of intra module")
    parser.add_argument("--cross_num_hidden_layers", type=int, default=2, help="Layer NO. of cross.")

    parser.add_argument("--freeze_layer_num", type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument("--linear_patch", type=str, default="3d", choices=["2d", "3d"], help="linear projection of flattened patches.")

    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")

    parser.add_argument("--clip_path", type=str, default="/content/RSICC/ckpts/pytorch_model.bin.0", help="Layer NO. of intra module")
    parser.add_argument("--save_model_path", type=str, default="/content/RSICC/ckpts", help="Layer NO. of intra module")
    
    #params for dual branch
    parser.add_argument("--dual_branch", action='store_true', help="Enable dual branch")
    parser.add_argument("--gate", action='store_true', help="Enable dual branch")
    parser.add_argument("--eval_just_RSICC", action='store_true', help="Enable dual branch")

    #params for text encoder
    parser.add_argument("--clip_text_encoder", action='store_true')

    # Training parameters
    parser.add_argument("--epochs", type=int, default=40, help="number of epochs to train for (if early stopping is not triggered).")
    parser.add_argument("--stop_criteria", type=int, default=10, help="training stop if epochs_since_improvement == stop_criteria")
    parser.add_argument("--batch_size", type=int, default=28, help="batch_size")
    parser.add_argument("--print_freq", type=int, default=100, help="printing training/validation stats every __ batches.")
    parser.add_argument("--workers", type=int, default=0, help="for data-loading; right now, only 0 works with h5pys in windows.")
    parser.add_argument("--encoder_lr", type=float, default=5e-5, help="learning rate for encoder if fine-tuning.")
    parser.add_argument("--decoder_lr", type=float, default=5e-5, help="learning rate for decoder.")
    parser.add_argument("--clip_encoder_lr", type=float, default=0.0001, help="learning rate for CLIP fine-tuning.")    
    parser.add_argument("--grad_clip", type=float, default=5.0, help="clip gradients at an absolute value of.")
    parser.add_argument("--fine_tune_encoder", type=bool, default=True, help="whether fine-tune encoder or not")
    parser.add_argument("--checkpoint", default="None", help="path to checkpoint, None if none.")
    
    # Validation
    parser.add_argument("--Split", default="VAL", help="which")
    parser.add_argument("--beam_size", type=int, default=1, help="beam_size.")
    parser.add_argument("--savepath", default=folder_path)
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup " "for. E.g., 0.1 = 10%% of training.",
    )

    args = parser.parse_args()
    main(args)
    