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

from mg_models import MCCFormers_diff_as_Q, DecoderTransformer, CNN_Encoder
from datasets import CaptionDataset
from utils import AverageMeter, adjust_learning_rate, bridge_embeddings_and_transfer, accuracy
from eval import evaluate_transformer

from CLIP_modules.modeling import CLIP4IDC
from CLIP_modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from mg_models.transformer import MemoryAugmentedEncoder, GlobalGroupingAttention_with_DC

seed = 1
torch.manual_seed(seed)

class CLIPVisualEncoder(nn.Module):
    def __init__(self, clip_model_path, target_dim):
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

        # 3. Boyut Eşitleyici (Projection Layer)
        self.projection = nn.Linear(768, target_dim) # 768 olmasının sebebi custom forward yapmamız
        self.projection.float()

        # 4. CLIP encoder (freeze) 
        for param in self.visual_encoder.parameters():
            param.requires_grad = False # veya True
    
    def forward(self, x):
        # x shape: [Batch, 3, 224, 224]
        
        # --- CLIP'in içindeki forward akışını MANUEL yapıyoruz ---
        with torch.no_grad():
            # 1. Conv1 (Patch Embedding)
            # Çıktı: [Batch, 768, 7, 7] (ViT-B/32 için)
            x = self.visual_encoder.conv1(x) 
            
            # 2. Flatten ve Transpose
            # Çıktı: [Batch, 49, 768]
            x = x.reshape(x.shape[0], x.shape[1], -1) 
            x = x.permute(0, 2, 1) 
            
            # 3. Class Embedding ve Positional Embedding Ekleme
            # class_embedding shape: [768] -> [1, 1, 768] -> [Batch, 1, 768]
            class_embedding = self.visual_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
            
            # x shape: [Batch, 50, 768] (49 patch + 1 class)
            x = torch.cat([class_embedding, x], dim=1) 
            x = x + self.visual_encoder.positional_embedding.to(x.dtype)
            
            # 4. Layer Norm (Pre-Transformer)
            x = self.visual_encoder.ln_pre(x)
            
            # 5. Transformer Katmanları
            x = x.permute(1, 0, 2)  # [50, Batch, 768] 
            x = self.visual_encoder.transformer(x)
            x = x.permute(1, 0, 2)  # [Batch, 50, 768] (Geri çevir)
            
            # 6. Layer Norm (Post-Transformer)
            x = self.visual_encoder.ln_post(x)

        # --- RSICCformer İçin Şekillendirme ---
        
        # 7. CLS Tokeni at (İlk tokeni çıkar)
        # Geriye [Batch, 49, 768] kalır
        features = x[:, 1:, :] 
        
        # 8. Kare formata geri döndür
        # Shape: [Batch, 768, 7, 7]
        batch_size = features.shape[0]
        side = int(features.shape[1] ** 0.5) # 7
        features = features.permute(0, 2, 1).view(batch_size, -1, side, side)
        
        # 9. Interpolasyon (14x14'e büyüt) 
        features = torch.nn.functional.interpolate(features, size=(14, 14), mode='bilinear', align_corners=False)
        # Shape: [Batch, 768, 14, 14]

        # 10. Projection (Boyut Eşitleme)
        # Linear katman son boyutta çalışır, bu yüzden kanalları sona al: [Batch, 14, 14, 768]
        features = features.permute(0, 2, 3, 1)
        out = self.projection(features)
        
        # 11. Son Çıktı Formatı: [Batch, 1024, 14, 14]
        out = out.permute(0, 3, 1, 2)
        
        return out

def train(
    args,
    train_loader,
    encoder_image,
    clip_encoder_image,
    encoder_feat,
    decoder,
    criterion,
    encoder_image_optimizer,
    clip_encoder_optimizer,
    encoder_image_lr_scheduler,
    #clip_encoder_scheduler,
    encoder_feat_optimizer,
    encoder_feat_lr_scheduler,
    decoder_optimizer,
    decoder_lr_scheduler,
    epoch,
    projection_layer,
    projection_optimizer,
    mg_encoder,
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
    #encoder_image2.train()
    clip_encoder_image.train()
    encoder_feat.train()
    decoder.train()  # train mode (dropout and batchnorm is used)
    projection_layer.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, (img_pairs, caps, caplens) in enumerate(train_loader):

        data_time.update(time.time() - start)

        # Back prop.
        decoder_optimizer.zero_grad()
        encoder_feat_optimizer.zero_grad()
        clip_encoder_optimizer.zero_grad()
        if(args.dual_branch == True):
            encoder_image_optimizer.zero_grad()
            projection_optimizer.zero_grad()

        # Move to GPU, if available
        img_pairs = img_pairs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs_A = img_pairs[:, 0, :, :, :]
        imgs_B = img_pairs[:, 1, :, :, :]

        clip_imgs_A = clip_encoder_image(imgs_A)
        clip_imgs_B = clip_encoder_image(imgs_B)

        if(args.dual_branch == True ):
            res_imgs_A = encoder_image(imgs_A)
            res_imgs_B = encoder_image(imgs_B)

        final_imgs_A = clip_encoder_image(imgs_A)
        final_imgs_B = clip_encoder_image(imgs_B)

        if(args.dual_branch == True and args.feature_fusion == "addition"):
            final_imgs_A = (clip_imgs_A + res_imgs_A) / 2
            final_imgs_B = (clip_imgs_B + res_imgs_B) / 2
        
        elif(args.dual_branch == True and args.feature_fusion == "concat"):
            final_imgs_A = torch.cat([clip_imgs_A, res_imgs_B], dim=1)
            final_imgs_B = torch.cat([clip_imgs_B, res_imgs_B], dim=1)
            final_imgs_A = projection_layer(final_imgs_A)
            final_imgs_B = projection_layer(final_imgs_B)

        elif(args.dual_branch == True and args.feature_fusion == "mg"):
            final_imgs_A = torch.cat([clip_imgs_A, res_imgs_B], dim=1)
            final_imgs_B = torch.cat([clip_imgs_B, res_imgs_B], dim=1)
            final_imgs_A, mask_enc = mg_encoder(res_imgs_A, clip_imgs_A, None, isencoder=True)
            final_imgs_B, mask_enc = mg_encoder(res_imgs_B, clip_imgs_B, None, isencoder=True)
            

        fused_feat = encoder_feat(
            final_imgs_A,
            final_imgs_B,
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

        clip_encoder_optimizer.step()

        projection_optimizer.step()

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

def save_checkpoint(args, data_name, epoch, epochs_since_improvement, 
                             encoder_image, encoder_feat, decoder, 
                             encoder_image_optimizer, encoder_feat_optimizer, decoder_optimizer,
                             clip_encoder_image, clip_encoder_optimizer=None, projection_layer=None, projection_optimizer=None):
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

    # CLIP Optimizer (Eğer gönderildiyse - Ki gönderilmeli!)
    if clip_encoder_optimizer is not None:
        state['clip_encoder_optimizer'] = clip_encoder_optimizer.state_dict()
    # CLIP Optimizer (Eğer gönderildiyse - Ki gönderilmeli!)
    if projection_layer is not None:
        state['projection_layer'] = projection_layer.state_dict()

    if projection_optimizer is not None:
        state['projection_optimizer'] = projection_optimizer.state_dict()

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

def main(args, meteor_output=None):
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

    # Initialize
    # Encoder
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

    encoder_image_lr_scheduler = (
        StepLR(encoder_image_optimizer, step_size=900, gamma=1) if args.fine_tune_encoder else None
    )

    encoder_feat_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, encoder_feat.parameters()), lr=args.encoder_lr
    )
    encoder_feat_lr_scheduler = StepLR(encoder_feat_optimizer, step_size=900, gamma=1)

    decoder_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=args.decoder_lr
    )
    decoder_lr_scheduler = StepLR(decoder_optimizer, step_size=900, gamma=1)

    # Move to GPU, if available
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
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Önce Resize ekliyoruz, sonra Normalize yapıyoruz
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(args.data_folder, args.data_name, "TRAIN", 
                    transform=transforms.Compose([
                        transforms.Resize((224, 224)),
                        normalize
                    ])),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    # ------------------------------ CLIP ENTEGRASYONU ------------------------------
    Clip_visual_encoder_module = CLIPVisualEncoder(args.clip_path,1024)

    clip_encoder_image = Clip_visual_encoder_module

    clip_encoder_image_optimizer = torch.optim.Adam([
        {'params': clip_encoder_image.visual_encoder.parameters(), 'lr': 1e-6},
        {'params': clip_encoder_image.projection.parameters(), 'lr': 1e-4},])
    clip_encoder_image = clip_encoder_image.cuda()

    clip_encoder_image.train()
    # ------------------------------ CLIP ENTEGRASYONU ------------------------------
    
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

    #------------------------ DUAL BRANCH ENTEGRASYONU (CONCAT)----------------

    projection_layer = nn.Conv2d(2048, 1024, kernel_size=1).cuda()

    projection_optimizer = torch.optim.Adam(projection_layer.parameters(), lr=args.encoder_lr)

    #------------------------ DUAL BRANCH ENTEGRASYONU (CONCAT)----------------

    #------------------------ MG TRANSFORMER ----------------

    mg_encoder = MemoryAugmentedEncoder(3, 0, attention_module=GlobalGroupingAttention_with_DC)

    #------------------------ MG TRANSFORMER ----------------




    if(args.eval_mode == False):
        # Epochs
        for epoch in range(start_epoch, args.epochs):

            print(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
            
            train(
                args,
                train_loader=train_loader,
                encoder_image=encoder_image,
                clip_encoder_image=clip_encoder_image,
                encoder_feat=encoder_feat,
                decoder=decoder,
                criterion=criterion,
                encoder_image_optimizer=encoder_image_optimizer,
                clip_encoder_optimizer=clip_encoder_image_optimizer,
                encoder_image_lr_scheduler=encoder_image_lr_scheduler,
                #clip_encoder_scheduler=clip_encoder_scheduler,
                encoder_feat_optimizer=encoder_feat_optimizer,
                encoder_feat_lr_scheduler=encoder_feat_lr_scheduler,
                decoder_optimizer=decoder_optimizer,
                decoder_lr_scheduler=decoder_lr_scheduler,
                epoch=epoch,
                projection_optimizer = projection_optimizer,
                projection_layer = projection_layer,
                mg_encoder = mg_encoder,
            )

            metrics = evaluate_transformer(
                args, encoder_image=encoder_image,clip_encoder_image=clip_encoder_image, encoder_feat=encoder_feat, decoder=decoder
            )

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
                save_checkpoint(args, "SecondCC", epoch, epochs_since_improvement, 
                                encoder_image, encoder_feat, decoder, 
                                encoder_image_optimizer, encoder_feat_optimizer, decoder_optimizer,
                                clip_encoder_image, clip_encoder_image_optimizer)
                
            # Early Stopping
            if epochs_since_improvement == args.stop_criteria:
                print(f"Early stopping triggered! Validation metrics hasn't increased for {args.stop_criteria} epochs.")
                break
            if epochs_since_improvement > 0 and epochs_since_improvement % 3 == 0:
                adjust_learning_rate(decoder_optimizer, 0.7)

    # ---------------------------- EVAL SECTION ----------------------------
    else:
        print(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=str(device))
        
        if 'encoder_image' in checkpoint:
            encoder_image.load_state_dict(checkpoint['encoder_image'])
        
        if 'encoder_feat' in checkpoint:
            encoder_feat.load_state_dict(checkpoint['encoder_feat'])
            
        if 'decoder' in checkpoint:
            decoder.load_state_dict(checkpoint['decoder'])

        # Check for CLIP specifically
        if 'clip_encoder_image' in checkpoint:
            clip_encoder_image.load_state_dict(checkpoint['clip_encoder_image'])
        else:
            print("WARNING: 'clip_encoder_image' weights not found in checkpoint. Using initialized weights.")

        # Now run evaluation
        metrics = evaluate_transformer(
            args, 
            encoder_image=encoder_image,
            clip_encoder_image=clip_encoder_image, 
            encoder_feat=encoder_feat, 
            decoder=decoder
        )

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
    parser.add_argument("--eval_mode", type=bool, default=False)
    parser.add_argument("--checkpoint_path", type=str, default="")

    #params for CLIP4IDC implementation
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--decoder_model", default="decoder-base", type=str, required=False, help="Decoder module")
    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--intra_num_hidden_layers", type=int, default=9, help="Layer NO. of intra module")
    parser.add_argument("--clip_path", type=str, default="/content/RSICC/ckpts/pytorch_model.bin.0", help="Layer NO. of intra module")
    parser.add_argument("--save_model_path", type=str, default="/content/RSICC/ckpts", help="Layer NO. of intra module")
    
    #params for dual branch
    parser.add_argument("--dual_branch", type=bool, default=False)
    parser.add_argument("--feature_fusion", type=str, default="concat") #concat, addition, mg

    #params for text encoder
    parser.add_argument("--clip_text_encoder", type=bool, default=False)

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
    