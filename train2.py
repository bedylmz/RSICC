import os
import time
import json
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import argparse
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

from tqdm import tqdm
import math

import random
from itertools import islice

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from models import MCCFormers_diff_as_Q, DecoderTransformer, CNN_Encoder
from datasets import CaptionDataset
from utils1 import AverageMeter, adjust_learning_rate, bridge_embeddings_and_transfer, accuracy
from eval1 import evaluate_transformer

from CLIP_modules.modeling import CLIP4IDC
from CLIP_modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

seed = int(time.time())
torch.manual_seed(seed)

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms.functional as F


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

    def forward(self, resnet_feat_before, resnet_feat_after, clip_feat_before, clip_feat_after):
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

        resnet_diff = torch.abs(resnet_feat_before - resnet_feat_after)

        # 3. GATE Mekanizması: Ne kadar değişim var?
        # ResNet fark vektörüne bakarak bir "alpha" katsayısı üret
        alpha = self.gate_fc(torch.cat([resnet_feat_before, resnet_feat_after, resnet_diff], dim=1))
        # alpha output: [Batch, 1] -> Her görüntü için 0 (değişim yok) ile 1 (değişim var) arası.
        
        resnet_feat_before = (1-alpha) * resnet_feat_before
        resnet_feat_after = (1-alpha) * resnet_feat_after
        clip_feat_before = alpha * clip_feat_before
        clip_feat_after = alpha * clip_feat_after

        final_before = torch.cat([resnet_feat_before, clip_feat_before], dim=1)
        final_after = torch.cat([resnet_feat_after, clip_feat_after], dim=1)

        final_before = F.normalize(final_before, p=2, dim=1)
        final_after = F.normalize(final_after, p=2, dim=1)

        final_before = self.projection_dim(final_before)
        final_before = self.projection_dim(final_after)

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
            clip_out = clip_encoder_image(imgs_full_clip, 2)
            clip_out_A = clip_out[:,0,:] # 768 100 b
            clip_out_B = clip_out[:,50,:]

            imgs_A_resnet = norm_resnet(imgs_A) # ResNet için normalize et
            imgs_B_resnet = norm_resnet(imgs_B)
            
            resnet_A = encoder_image(imgs_A_resnet)
            resnet_B = encoder_image(imgs_B_resnet)
            

            resnet_A_adapt, clip_A_adapt = adaptLayer(resnet_A, clip_out_A)
            resnet_B_adapt, clip_B_adapt = adaptLayer(resnet_B, clip_out_B)


            resnet_A_normed, clip_A_normed = layerNormalizeLayer(resnet_A_adapt, clip_A_adapt)
            resnet_B_normed, clip_B_normed = layerNormalizeLayer(resnet_B_adapt, clip_B_adapt)

            # train fonksiyonu içinde (satır 194 civarı)
            # Girdi: [Batch, 512, 14, 14]
            if(args.gate ==True):
                # 1. Kanalı sona alıp düzleştirin: [Batch, 196, 512]
                b, c, h, w = resnet_A_normed.shape
                resnet_A_flat = resnet_A_normed.permute(0, 2, 3, 1).view(b, h*w, c) 

                # 2. Attention uygulayın (Çıktı yine [Batch, 196, 512] olacak)
                resnet_A_att, _ = gateSelf(resnet_A_flat)

                # 3. Tekrar [Batch, 512, 14, 14] formatına dönün (Concat için gerekli)
                resnet_A_normed = resnet_A_att.view(b, h, w, c).permute(0, 3, 1, 2)

                b, c, h, w = resnet_B_normed.shape
                resnet_B_flat = resnet_B_normed.permute(0, 2, 3, 1).view(b, h*w, c) 

                # 2. Attention uygulayın (Çıktı yine [Batch, 196, 512] olacak)
                resnet_B_att, _ = gateSelf(resnet_B_flat)

                # 3. Tekrar [Batch, 512, 14, 14] formatına dönün (Concat için gerekli)
                resnet_B_normed = resnet_B_att.view(b, h, w, c).permute(0, 3, 1, 2)

            final_A = torch.cat([resnet_A_normed, clip_A_normed], dim=1)
            final_B = torch.cat([resnet_B_normed, clip_B_normed], dim=1)

            fused_feat = encoder_feat(
                final_A,
                final_B,
            ) # encoder_out: (S, batch, feature_dim) # fused_feat: (S, batch, feature_dim) # buyuk tensor atama yavaslatior (#batch time = 0.5)
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
            # print('TIME: ', time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
            print(
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
    if old_key == "layer.0.weight":
        return "layer.1.weight"

    return old_key

def prep_optimizer(args, model, device, num_train_optimization_steps, coef_lr=1.0):
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
    import torch
    
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


def main(args):
    print(args)
    global metrics_list
    print(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))

    start_epoch = 0
    best_bleu4 = 0.0  # BLEU-4 score right now
    epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

    print(f"CUDA available: {torch.cuda.is_available()}")

    cudnn.benchmark = (
        True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
    )

    print("*"*20 ,device,"*"*20)
    # Read word map
    word_map_file = os.path.join(args.data_folder, "WORDMAP_" + args.data_name + ".json")
    with open(word_map_file, "r") as j:
        word_map = json.load(j)

    
    # ------------------------------ CLIP ENTEGRASYONU ------------------------------
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
    if(args.dual_branch == True):
        encoder_image = CNN_Encoder(NetType=args.encoder_image, method=args.decoder)
        # set the encoder_dim
    encoder_image_dim = 1024 # resnet101

    if args.encoder_feat == "MCCFormers_diff_as_Q":
        encoder_feat = MCCFormers_diff_as_Q(
            feature_dim=1024,
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
    if(args.dual_branch == True):
        encoder_image = encoder_image.to(device)
    encoder_feat = encoder_feat.to(device)
    decoder = decoder.to(device)

    print("Checkpoint_savepath:{}".format(args.savepath))
    print(
        "Encoder_image_mode:{}   Encoder_feat_mode:{}   Decoder_mode:{}".format(
            args.encoder_image_model, args.encoder_feat, args.decoder
        )
    )
    print(
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

    # ---------------------------- TRAINING SECTION ----------------------------
    if args.eval_mode == False and args.eval_caption == False:
        # Epochs
        for epoch in range(start_epoch, args.epochs):

            print(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
            
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
                        )
            else:
              metrics = evaluate_transformer(
                    args, 
                    clip_encoder_image=clip_encoder_image, 
                    encoder_feat=encoder_feat,
                    decoder=decoder,
                    adaptLayerClip=adaptLayerClip)

            # -----------------------------------------------------------------------------------------------------
            # One epoch's validation
            print("-------------------------epoch passed-------------------------")

            recent_bleu4 = metrics["Bleu_4"]
            
            # Check if there was an improvement
            is_best = recent_bleu4 > best_bleu4
            best_bleu4 = max(recent_bleu4, best_bleu4)
            if not is_best:
                epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            else:
                epochs_since_improvement = 0
            if is_best:
                print("-------------------------checkpoint Saved-------------------------")
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
                print(f"Early stopping triggered! Validation metrics hasn't increased for {args.stop_criteria} epochs.")
                break
            if epochs_since_improvement > 0 and epochs_since_improvement % 3 == 0:
                adjust_learning_rate(decoder_optimizer, 0.7)

    # ---------------------------- EVAL SECTION ----------------------------
    elif args.eval_mode == True and args.eval_caption == False:
        print(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=str(device))
        
        if 'encoder_image' in checkpoint:
            encoder_image.load_state_dict(checkpoint['encoder_image'])
        
        if 'encoder_feat' in checkpoint:
            encoder_feat.load_state_dict(checkpoint['encoder_feat'])
            
        if 'decoder' in checkpoint:
            decoder.load_state_dict(checkpoint['decoder'])
        
        if 'layerNormalizeLayer' in checkpoint:
            layerNormalizeLayer.load_state_dict(checkpoint['layerNormalizeLayer'])

        if 'adaptLayer' in checkpoint:
            adaptLayer.load_state_dict(checkpoint['adaptLayer'])
        
        if 'adaptLayerClip' in checkpoint:
            adaptLayerClip.load_state_dict(checkpoint['adaptLayerClip'])

        if 'gateSelf' in checkpoint:
            gateSelf.load_state_dict(checkpoint['gateSelf'])

        # Check for CLIP specifically
        if 'clip_encoder_image' in checkpoint:
            clip_encoder_image.load_state_dict(checkpoint['clip_encoder_image'])
        else:
            print("WARNING: 'clip_encoder_image' weights not found in checkpoint. Using initialized weights.")
        if(args.dual_branch == True):
              if(args.gate ==  True):
                metrics = evaluate_transformer(
                        args, 
                        encoder_image=encoder_image, 
                        clip_encoder_image=clip_encoder_image, 
                        encoder_feat=encoder_feat,
                        decoder=decoder,
                        layerNormalizeLayer=layerNormalizeLayer,
                        adaptLayer=adaptLayer,
                        gateSelf= gateSelf,
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
                        )
        else:
              metrics = evaluate_transformer(
                    args, 
                    clip_encoder_image=clip_encoder_image, 
                    encoder_feat=encoder_feat,
                    decoder=decoder,
                    adaptLayerClip=adaptLayerClip)

    #---------------------------- CAPTION EVAL SECTION ----------------------------
    else:
        print(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=str(device))

        raw_state_dict = checkpoint['clip_encoder_image']
        clip_state_dict = {}
        for k, v in raw_state_dict.items():
            if "conv2.weight" in k:
                continue

            clip_state_dict[k] = v

        
        if 'encoder_image' in checkpoint:
            encoder_image.load_state_dict(checkpoint['encoder_image'])
        
        if 'encoder_feat' in checkpoint:
            encoder_feat.load_state_dict(checkpoint['encoder_feat'])
            
        if 'decoder' in checkpoint:
            decoder.load_state_dict(checkpoint['decoder'])
        
        if 'layerNormalizeLayer' in checkpoint:
            layerNormalizeLayer.load_state_dict(checkpoint['layerNormalizeLayer'])

        if 'adaptLayer' in checkpoint:
            adaptLayer.load_state_dict(checkpoint['adaptLayer'])
        
        if 'adaptLayerClip' in checkpoint:
            adaptLayerClip.load_state_dict(checkpoint['adaptLayerClip'])

        if 'gateSelf' in checkpoint:
            gateSelf.load_state_dict(checkpoint['gateSelf'])

        # Check for CLIP specifically
        if 'clip_encoder_image' in checkpoint:
            clip_encoder_image.load_state_dict(clip_state_dict)
        else:
            print("WARNING: 'clip_encoder_image' weights not found in checkpoint. Using initialized weights.")

        if(args.dual_branch):
            if(args.gate):
                hypotheses, references, img_A_tensor, img_B_tensor, random_index = evaluate_transformer_caption(
                        args, 
                        encoder_image=encoder_image, 
                        clip_encoder_image=clip_encoder_image, 
                        encoder_feat=encoder_feat,
                        decoder=decoder,
                        layerNormalizeLayer=layerNormalizeLayer,
                        adaptLayer=adaptLayer,
                        gateSelf= gateSelf,
                        )
            else:
                hypotheses, references, img_A_tensor, img_B_tensor, random_index = evaluate_transformer_caption(
                        args, 
                        encoder_image=encoder_image, 
                        clip_encoder_image=clip_encoder_image, 
                        encoder_feat=encoder_feat,
                        decoder=decoder,
                        layerNormalizeLayer=layerNormalizeLayer,
                        adaptLayer=adaptLayer,
                        )
        elif(args.eval_just_RSICC):
                hypotheses, references, img_A_tensor, img_B_tensor, random_index = evaluate_transformer_caption(
                        args, 
                        encoder_image=encoder_image, 
                        encoder_feat=encoder_feat,
                        decoder=decoder,
                        )
        else:
            hypotheses, references, img_A_tensor, img_B_tensor, random_index = evaluate_transformer_caption(
                args, 
                clip_encoder_image=clip_encoder_image, 
                encoder_feat=encoder_feat,
                decoder=decoder,
                adaptLayerClip=adaptLayerClip,
                )
            
        result_json, references_json = save_captions(args, word_map, hypotheses, references)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Resimleri hazırla ve çiz
        axes[0].imshow(img_A_tensor)
        axes[0].set_title("Image A (Before)")
        axes[0].axis('off')

        axes[1].imshow(img_B_tensor)
        axes[1].set_title("Image B (After)")
        axes[1].axis('off')



        # Başlığı yaz ve dosyayı kaydet
        plt.suptitle(f"GT: {references_json["0"][0]} \nGuess: {result_json["0"][0]}", fontsize=12)
        
        filename = f"batch_sample_{random_index}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig) # Figürü kapatarak belleği temizle
        
        print(f"✅ Görüntü başarıyla kaydedildi: {filename}")
        # --- GÖRÜNTÜ KAYDETME KODU BİTİŞİ ---


def save_captions(args, word_map, hypotheses, references):
    result_json_file = {}
    reference_json_file = {}
    kkk = -1
    for item in hypotheses:
        kkk += 1
        line_hypo = ""

        for word_idx in item:
            word = get_key(word_map, word_idx)
            # logger.info(word)
            line_hypo += word[0] + " "

        result_json_file[str(kkk)] = []
        result_json_file[str(kkk)].append(line_hypo)

        line_hypo += "\r\n"

    kkk = -1
    for item in tqdm(references):
        kkk += 1

        reference_json_file[str(kkk)] = []

        for sentence in item:
            line_repo = ""
            for word_idx in sentence:
                word = get_key(word_map, word_idx)
                line_repo += word[0] + " "
            reference_json_file[str(kkk)].append(line_repo)
            line_repo += "\r\n"
    
    print(result_json_file)
    print(reference_json_file)

    return result_json_file, reference_json_file

def get_key(dict_, value):
  return [k for k, v in dict_.items() if v == value]

def evaluate_transformer_caption(
        args: argparse.Namespace = None,
        encoder_feat=None,
        decoder=None,
        clip_encoder_image=None,
        encoder_image=None,
        layerNormalizeLayer=None,
        adaptLayer=None,        # Burası
        adaptLayerClip=None,
        gateSelf=None,
        ):
    import torch.nn.functional as F
    # Load model
    encoder_feat = encoder_feat.to(device)
    encoder_feat.eval()
    decoder = decoder.to(device)
    decoder.eval()

    if (encoder_image != None):
        encoder_image.to(device)
        encoder_image.eval()

    if (clip_encoder_image != None):
        clip_encoder_image.to(device)
        clip_encoder_image.eval()

    if (layerNormalizeLayer != None):
        layerNormalizeLayer.to(device)
        layerNormalizeLayer.eval()

    if (adaptLayer != None):
        adaptLayer.to(device)
        adaptLayer.eval()

    if (adaptLayerClip != None):
        adaptLayerClip.to(device)
        adaptLayerClip.eval()

    if (gateSelf != None):
        gateSelf.to(device)
        gateSelf.eval()

    # Load word map (word2ix)
    word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + args.data_name + '.json')
    with open(word_map_file, 'r') as f:
        word_map = json.load(f)

    rev_word_map = {v: k for k, v in word_map.items()}
    vocab_size = len(word_map)

    """
    Evaluation for decoder: transformer
    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    beam_size = args.beam_size
    Caption_End = False

    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC)
    ])

    loader = torch.utils.data.DataLoader(
        CaptionDataset(args.data_folder, args.data_name, args.Split, 
                    transform=transform),
        batch_size=1,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()
    change_references = list()
    change_hypotheses = list()
    nochange_references = list()
    nochange_hypotheses = list()
    change_acc=0
    nochange_acc=0

    with torch.no_grad():

        # ResNet İstatistikleri
        norm_resnet = BatchNormalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225], 
                                    device=device)
        
        # CLIP İstatistikleri
        norm_clip = BatchNormalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                std=[0.26862954, 0.26130258, 0.27577711], 
                                device=device)
        # -----------------------------------------------
        num_batches = len(loader)

        # 2. 0 ile toplam_sayı arasında rastgele bir index seç
        random_index = random.randint(0, num_batches - 1)

        print(f"{random_index}. sıradaki batch seçiliyor...")

        # 3. O index'e gidip batch'i çek (islice o noktaya kadar iterate eder)
        random_batch = next(islice(loader, random_index, None))

        # 4. Senin veri yapına göre unpack et
        img_pairs, caps, caplens, allcaps = random_batch

        from PIL import Image

        # 1. Load the image using Pillow
        if args.img_a and args.img_b:
            img0 = Image.open(args.img_a).convert('RGB')
            img1 = Image.open(args.img_b).convert('RGB')
        else:
            # Eğer parametre gelmezse hata vermemesi için eski usul veya bir hata mesajı
            print("Hata: Resim yolları belirtilmedi!")

        # 2. Define the transform to Tensor
        transform = transforms.Compose([
            transforms.Resize(256),            # Kısa kenarı 256 yapar (oranı korur)
            transforms.CenterCrop(224),        # Merkezin tam ortasından 224x224 keser
            transforms.ToTensor(),             # Tensor'a çevirir [0, 1]
            transforms.Normalize(              # İsteğe bağlı: ImageNet standart normalizasyonu
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

        # 3. Apply and check shape
        tensor_img0 = transform(img0)
        tensor_img1 = transform(img1)
        
        img_pairs = torch.stack([tensor_img0, tensor_img1])

        img_pairs = img_pairs.view(1, *img_pairs.shape)
        
        print(img_pairs.shape)

        # Kontrol
        print("Seçilen Caption:", caps[0])
        print("Batch Index:", random_index)
        # 5 image is the same when "shuffle=False" of the dataloader
        # if i>10:
        #     break
        k = beam_size

        # Move to GPU device, if available
        img_pairs = img_pairs.to(device)  # [1, 2, 3, 256, 256]
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

            final_A, final_B = adaptLayer(resnet_A_normed, clip_A_normed,resnet_B_normed, clip_B_normed)

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

        tgt = torch.zeros(52, k).to(device).to(torch.int64)
        tgt_length = tgt.size(0)
        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)

        tgt[0, :] = torch.LongTensor([word_map['<start>']]*k).to(device) # k_prev_words:[52,k]
        # Tensor to store top k sequences; now they're just <start>
        seqs = torch.LongTensor([[word_map['<start>']]*1] * k).to(device)  # [1,k]
        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)
        # Lists to store completed sequences and scores
        complete_seqs = []
        complete_seqs_scores = []
        step = 1

        k_prev_words = tgt.permute(1,0)
        S = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)

        # # We'll treat the problem as having a batch size of k, where k is beam_size
        encoder_out = encoder_out.expand(S,k, encoder_dim)  # [S,k, encoder_dim]
        encoder_out = encoder_out.permute(1,0,2)

        # Start decoding
        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:
            tgt = k_prev_words.permute(1,0)
            tgt_embedding = decoder.vocab_embedding(tgt)
            tgt_embedding = decoder.position_encoding(tgt_embedding)  # (length, batch, feature_dim)

            encoder_out = encoder_out.permute(1, 0, 2)
            pred = decoder.transformer(tgt_embedding, encoder_out, tgt_mask=mask)  # (length, batch, feature_dim)
            encoder_out = encoder_out.permute(1, 0, 2)
            pred = decoder.wdc(pred)  # (length, batch, vocab_size)
            scores = pred.permute(1,0,2)  # (batch,length,  vocab_size)
            scores = scores[:, step - 1, :].squeeze(1)  # [s, 1, vocab_size] -> [s, vocab_size]
            scores = F.log_softmax(scores, dim=1)
            # top_k_scores: [s, 1]
            scores = top_k_scores.expand_as(scores) + scores  # [s, vocab_size]
            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)


            # Convert unrolled indices to actual indices of scores
            # prev_word_inds = top_k_words // vocab_size  # (s)
            # if max(top_k_words)>vocab_size:
            #     logger.info(">>>>>>>>>>>>>>>>>>")
            prev_word_inds = torch.div(top_k_words, vocab_size, rounding_mode='floor')
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
            # Set aside complete sequences
            if len(complete_inds) > 0:
                Caption_End = True
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly
            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            # Important: this will not work, since decoder has self-attention
            # k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1).repeat(k, 52)
            k_prev_words = k_prev_words[incomplete_inds]
            k_prev_words[:, :step + 1] = seqs  # [s, 52]
            # k_prev_words[:, step] = next_word_inds[incomplete_inds]  # [s, 52]
            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        # choose the caption which has the best_score.
        if (len(complete_seqs_scores) == 0):
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        if (len(complete_seqs_scores) > 0):
            assert Caption_End
            indices = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[indices]
            # References
            img_caps = allcaps[0].tolist()
            img_captions = list(
                map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                    img_caps))  # remove <start> and pads

            references.append(img_captions)
            # Hypotheses
            new_sent = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
            hypotheses.append(new_sent)
            assert len(references) == len(hypotheses)

            # # 判断有没有变化
            nochange_list = ["the scene is the same as before ", "there is no difference ",
                                "the two scenes seem identical ", "no change has occurred ",
                                "almost nothing has changed "]
            ref_sentence = img_captions[1]
            ref_line_repo = ""
            for ref_word_idx in ref_sentence:
                ref_word = get_key(word_map, ref_word_idx)
                ref_line_repo += ref_word[0] + " "

            hyp_sentence = new_sent
            hyp_line_repo = ""
            for hyp_word_idx in hyp_sentence:
                hyp_word = get_key(word_map, hyp_word_idx)
                hyp_line_repo += hyp_word[0] + " "
            # 对于变化图像对
            if ref_line_repo not in nochange_list:
                change_references.append(img_captions)
                change_hypotheses.append(new_sent)
                if hyp_line_repo not in nochange_list:
                    change_acc = change_acc+1
            else:
                nochange_references.append(img_captions)
                nochange_hypotheses.append(new_sent)
                if hyp_line_repo in nochange_list:
                    nochange_acc = nochange_acc+1

            # --- GÖRÜNTÜ KAYDETME KODU BAŞLANGICI ---
        
        
        # Tensörleri al (Batch size 1 olduğu için 0. index)
        img_A_tensor = img_pairs[0, 0].cpu()
        img_B_tensor = img_pairs[0, 1].cpu()


        # Figür oluştur

        img_A_tensor = img_A_tensor.permute(1, 2, 0).numpy()
        # Görüntü normalize edilmişse veya değerleri garipe, 0-1 arasına çek
        img_A_tensor = (img_A_tensor - img_A_tensor.min()) / (img_A_tensor.max() - img_A_tensor.min())


        img_B_tensor = img_B_tensor.permute(1, 2, 0).numpy()
        # Görüntü normalize edilmişse veya değerleri garipe, 0-1 arasına çek
        img_B_tensor = (img_B_tensor - img_B_tensor.min()) / (img_B_tensor.max() - img_B_tensor.min())
        

    return hypotheses, references, img_A_tensor, img_B_tensor, random_index

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
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--decoder_model", default="decoder-base", type=str, required=False, help="Decoder module")
    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--intra_num_hidden_layers", type=int, default=9, help="Layer NO. of intra module")
    parser.add_argument("--clip_path", type=str, default="/content/RSICC/ckpts/pytorch_model.bin.0", help="Layer NO. of intra module")
    parser.add_argument("--save_model_path", type=str, default="/content/RSICC/ckpts", help="Layer NO. of intra module")
    
    #params for dual branch
    parser.add_argument("--dual_branch", action='store_true', help="Enable dual branch")
    parser.add_argument("--gate", action='store_true', help="Enable dual branch")
    parser.add_argument("--eval_caption", action='store_true', help="Enable dual branch")

    #params for text encoder
    parser.add_argument("--clip_text_encoder", action='store_true')

    # Training parameters
    parser.add_argument("--epochs", type=int, default=40, help="number of epochs to train for (if early stopping is not triggered).")
    parser.add_argument("--stop_criteria", type=int, default=10, help="training stop if epochs_since_improvement == stop_criteria")
    parser.add_argument("--batch_size", type=int, default=28, help="batch_size")
    parser.add_argument("--print_freq", type=int, default=100, help="print training/validation stats every __ batches.")
    parser.add_argument("--workers", type=int, default=0, help="for data-loading; right now, only 0 works with h5pys in windows.")
    parser.add_argument("--encoder_lr", type=float, default=5e-5, help="learning rate for encoder if fine-tuning.")
    parser.add_argument("--decoder_lr", type=float, default=5e-5, help="learning rate for decoder.")
    parser.add_argument("--clip_encoder_lr", type=float, default=0.0001, help="learning rate for CLIP fine-tuning.")    
    parser.add_argument("--grad_clip", type=float, default=5.0, help="clip gradients at an absolute value of.")
    parser.add_argument("--fine_tune_encoder", type=bool, default=True, help="whether fine-tune encoder or not")
    parser.add_argument("--checkpoint", default="None", help="path to checkpoint, None if none.")

    parser.add_argument('--img_a', type=str, default=None, help='İlk resmin yolu')
    parser.add_argument('--img_b', type=str, default=None, help='İkinci resmin yolu')
    
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
   