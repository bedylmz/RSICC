import datetime
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import argparse
from torch.optim.lr_scheduler import StepLR

from models import CNN_Encoder
from models_RSICCformerDfusion import *
from datasets import *
from utils import *
from eval import evaluate_transformer



from exploringDebugging import write_debug

seed = 1
torch.manual_seed(seed)

metrics_list = []
losses_output = []
AVG_losses_output = []
top5_accuracy_output = []
batch_time_output = []

train_model_sonuc_map = {}
text_terminal = " "

rogue_l_output = []
cider_output = []
bleu_4_output = []
rogue_l_nochange_output = []
cider_nochange_output = []
bleu_4_nochange_output = []
#meteor1_nochange_output = []
#meteor1_change_output = []
#meteor1_output = []
rogue_l_change_output = []
cider_change_output = []
bleu_4_change_output = []

val_model_sonuc_map = {}


def print_with_json(text):
    global text_terminal
    print(text)
    text_terminal += str(text) + "\n"


from CLIP_modules.modeling import CLIP4IDC

from CLIP_modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

class CLIPVisualEncoder(nn.Module):
    def __init__(self, clip_model_path, target_dim):
        super().__init__()
        # 1. Sizin kütüphanenizden CLIP modelini başlatın
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
        
        # 2. Checkpoint'i yükleyin
        #model_state_dict = torch.load(clip_model_path, map_location="cpu")
        #self.clip_model.load_state_dict(model_state_dict['model_state_dict']) # Key'lere dikkat

        # Sadece görsel kısmı al (örneğin visual transformer)
        self.visual_encoder = self.clip_model.clip.visual 

        # --- ÇÖZÜM BURADA ---
        # Modeli Float32 (Tam Hassasiyet) moduna zorla
        self.visual_encoder.float()

        # --- YENİ DÜZELTME (BUNU EKLEYİN) ---
        self.visual_encoder.cuda()

        # Modelin orijinal çıktı boyutunu al (örn: 768)
        clip_out_dim = self.visual_encoder.output_dim 

        # 3. Boyut Eşitleyici (Projection Layer)
        self.projection = nn.Linear(clip_out_dim, target_dim)
        self.projection.float()

        # 4. İsteğe bağlı: CLIP encoder'ı dondur (freeze) 
        # Eğer sadece feature extractor olacaksa dondurun, eğitilecekse açık bırakın.
        for param in self.visual_encoder.parameters():
            param.requires_grad = False # veya True

    """Old basic forward
    def forward(self, images):
        # CLIP'ten özellikleri çıkar
        # Dikkat: RSICCformer 'sequence' (yama dizisi) bekliyorsa, 
        # CLIP'in son katmanındaki pooling öncesi çıktıya ihtiyacınız var.

        with torch.no_grad(): # Eğer freeze ise
            features = self.visual_encoder(images) 
            # features shape örneği: [Batch, 197, 768] (ViT için)


        features = features[:, 1:, :]

        print("Feature Boyutu: "+str(features.shape()))

        # Boyut dönüşümü yap
        out = self.projection(features) 
        # out shape: [Batch, 197, target_dim]

        return out"""
    
    def forward(self, x):
        # x shape: [Batch, 3, 224, 224]
        
        # --- CLIP'in içindeki forward akışını MANUEL yapıyoruz ---
        # Böylece module_clip.py line 508'deki hatadan kaçınıyoruz.
        
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
            x = x.permute(1, 0, 2)  # [50, Batch, 768] (Transformer NLD formatı ister)
            x = self.visual_encoder.transformer(x)
            x = x.permute(1, 0, 2)  # [Batch, 50, 768] (Geri çevir)
            
            # 6. Layer Norm (Post-Transformer)
            x = self.visual_encoder.ln_post(x)
            
            # --- Buraya kadar özellikler çıkarıldı ---

        # --- RSICCformer İçin Şekillendirme ---
        
        # 7. CLS Tokeni at (İlk tokeni çıkar)
        # Geriye [Batch, 49, 768] kalır
        features = x[:, 1:, :] 
        
        # 8. Kare formata geri döndür
        # Shape: [Batch, 768, 7, 7]
        batch_size = features.shape[0]
        side = int(features.shape[1] ** 0.5) # 7
        features = features.permute(0, 2, 1).view(batch_size, -1, side, side)
        
        # 9. Interpolasyon (14x14'e büyüt) -> RSICCformer ResNet boyutu sever
        features = torch.nn.functional.interpolate(features, size=(14, 14), mode='bilinear', align_corners=False)
        # Shape: [Batch, 768, 14, 14]

        # 10. Projection (Boyut Eşitleme)
        # Linear katman son boyutta çalışır, bu yüzden kanalları sona al: [Batch, 14, 14, 768]
        features = features.permute(0, 2, 3, 1)
        out = self.projection(features)
        
        # 11. Son Çıktı Formatı: [Batch, 1024, 14, 14]
        # Kanalları tekrar başa al
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
    #clip_encoder_optimizer,
    encoder_image_lr_scheduler,
    #clip_encoder_scheduler,
    encoder_feat_optimizer,
    encoder_feat_lr_scheduler,
    decoder_optimizer,
    decoder_lr_scheduler,
    epoch,
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

    #encoder_image.train()
    #encoder_image2.train()
    clip_encoder_image.eval()
    encoder_feat.train()
    decoder.train()  # train mode (dropout and batchnorm is used)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs_our = AverageMeter()
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    best_bleu4 = 0.0  # BLEU-4 score right now

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, (img_pairs, caps, caplens) in enumerate(train_loader):
        #         if i == 20:
        #             break
        data_time.update(time.time() - start)

        # Back prop.
        decoder_optimizer.zero_grad()
        encoder_feat_optimizer.zero_grad()
        #clip_encoder_optimizer.zero_grad()
        #encoder_image_optimizer.zero_grad()

        # Move to GPU, if available
        img_pairs = img_pairs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        # Eklemek ve carparak eklemeyi de dene
        imgs_A = img_pairs[:, 0, :, :, :]
        imgs_B = img_pairs[:, 1, :, :, :]

        clip_imgs_A = clip_encoder_image(imgs_A)
        clip_imgs_B = clip_encoder_image(imgs_B)
        clip_encoded = []

        # imgs_C = img_pairs[:, 2, :, :, :]
        # sem_A = img_pairs[:, 2, :, :, :]
        # sem_B = img_pairs[:, 3, :, :, :]

        # rsformer image encoder 
        #imgs_A = encoder_image(imgs_A)  # imgs_A: [batch_size,1024, 14, 14]
        #imgs_B = encoder_image(imgs_B)  # batch time = 0.35

        #imgs_A = encoder_image(imgs_A)  # imgs_A: [batch_size,1024, 14, 14]
        #imgs_B = encoder_image(imgs_B)  # batch time = 0.35

        """En son birlikte yaptığımız deneme
        # Convert a single tensor image (C,H,W) in range [0,1] or [0,255] to PIL
        
        # --new--
        # Tekil görüntüleri al: [Batch, 3, 224, 224]
        img_A_single = img_pairs[:, 0, :, :, :]
        img_B_single = img_pairs[:, 1, :, :, :]

        # YÖNTEM: Concatenate + View (Düzleştirme)
        
        # 1. A Resmini Hazırla
        # [Batch, 2, 3, 224, 224] oluşturuyoruz (Sizin yaptığınız kısım)
        img_A_input_5d = torch.cat([img_A_single.unsqueeze(1), img_A_single.unsqueeze(1)], dim=1)
        
        # HATA ÇÖZÜMÜ: 5D -> 4D'ye çeviriyoruz ([Batch * 2, 3, 224, 224])
        # Model bunu "52 tane resim" olarak görecek, içeride "26 tane 2 karelik video" olduğunu video_frame=2 ile anlayacak.
        b, t, c, h, w = img_A_input_5d.shape
        img_A_input = img_A_input_5d.view(b * t, c, h, w) 
        
        imgs_A_features = clip_encoder_image(img_A_input, video_frame=2)
        imgs_A = imgs_A_features[:, 0, :] # İlk kareyi al

        # 2. B Resmini Hazırla
        img_B_input_5d = torch.cat([img_B_single.unsqueeze(1), img_B_single.unsqueeze(1)], dim=1)
        
        # Aynı şekilde düzleştiriyoruz
        img_B_input = img_B_input_5d.view(b * t, c, h, w)
        
        imgs_B_features = clip_encoder_image(img_B_input, video_frame=2)
        imgs_B = imgs_B_features[:, 0, :] # İlk kareyi al"""

        """berkayın yapıtığı implement        
        to_pil = transforms.ToPILImage()

        # clip image encoder 
        for imgA,imgB in zip(clip_imgs_A,clip_imgs_B):
            # Clamp and convert to PIL
            imgA_pil = to_pil(imgA.cpu().clamp(0, 1))
            imgB_pil = to_pil(imgB.cpu().clamp(0, 1))

            # Now pass PIL images to your CLIP encode function
            encoded = model_arrange.encode_image(clip_encoder_image, imgA_pil, imgB_pil, device)
            clip_encoded.append(encoded)

        # stack along batch dimension
        clip_encoded = torch.stack(clip_encoded).to(device)  # shape [B, 2, 7, 7, 768]
        print("DEBUG: type(clip_encoded) =", type(clip_encoded))
        if isinstance(clip_encoded, torch.Tensor):
            print("DEBUG: clip_encoded.shape =", clip_encoded.shape)
        elif isinstance(clip_encoded, list):
            print("DEBUG: len(clip_encoded) =", len(clip_encoded))
            if len(clip_encoded) > 0 and isinstance(clip_encoded[0], torch.Tensor):
                print("DEBUG: clip_encoded[0].shape =", clip_encoded[0].shape)


        #clip_encoded = torch.stack(clip_encoded)

        write_debug("clip_encoded", clip_encoded)

        NewimgA = clip_encoded[:, 0, :, :, :]
        NewimgB = clip_encoded[:, 1, :, :, :]"""

        fused_feat = encoder_feat(
            clip_imgs_A,
            clip_imgs_B,
            #clip_encoded
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

        #if encoder_image_optimizer is not None:
        #    encoder_image_optimizer.zero_grad()
        #    clip_encoder_optimizer.zero_grad()  
  
        loss.backward()

        # Clip gradients
        #if args.grad_clip is not None:
        #    clip_gradient(decoder_optimizer, args.grad_clip)
        #    if encoder_image_optimizer is not None:
        #        clip_gradient(encoder_image_optimizer, args.grad_clip)

        # Update weights
        decoder_optimizer.step()
        decoder_lr_scheduler.step()
        
        encoder_feat_optimizer.step()
        encoder_feat_lr_scheduler.step()

        #encoder_image_optimizer.step()
        #encoder_image_lr_scheduler.step()
   
        #clip_encoder_optimizer.step()

        #if clip_encoder_scheduler is not None:
        #    clip_encoder_scheduler.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 1)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()
        if i % args.print_freq == 0:
            # print('TIME: ', time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
            print_with_json(
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
            losses_output.append(losses.val)
            AVG_losses_output.append(losses.avg)
            top5_accuracy_output.append(top5accs.val)
            batch_time_output.append(batch_time.val)


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


def main(args, meteor_output=None):
    print_with_json("bu toplayan modeldir.")
    print_with_json(args)
    global metrics_list
    print_with_json(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))

    start_epoch = 0
    best_bleu4 = 0.0  # BLEU-4 score right now
    epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
    # device = torch.device("cpu")  # sets device for model and PyTorch tensors

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
    #clip_encoder_image = model_arrange.load_model("C:/Users/AliCan/Desktop/clip4idc/trained_model_4090/pytorch_model.bin.7")
    #clip_encoder_image = model_arrange.load_model("C:/Users/AliCan/Desktop/clip4idc/ckpts/caption/pytorch_model.bin.9")

    # Retrieval Trained clip 
    #clip_encoder_image = model_arrange.load_model("/content/RSICC/ckpts/pytorch_model.bin.0")
    # encoder_image2 = CNN_Encoder(NetType=args.encoder_image, method=args.decoder)

    # encoder_image.fine_tune(args.fine_tune_encoder)
    # Weightleri yazdir

    # set the encoder_dim
    encoder_image_dim = 1024 # resnet101
    # filename = os.listdir(args.checkpoint)
    # checkpoint_path = os.path.join(args.checkpoint, filename[0])
    # print_with_json(args.checkpoint + filename[0])
    # checkpoint = torch.load(checkpoint_path, map_location=str(device))
    # encoder_image2 = checkpoint['encoder_image']
    # encoder_feat2 = checkpoint['encoder_feat']
    # decoder2 = checkpoint['decoder']

    if args.encoder_feat == "MCCFormers_diff_as_Q":
        encoder_feat = MCCFormers_diff_as_Q(
            feature_dim=encoder_image_dim,
            dropout=0.5,
            h=300, # 14 ten 300 çıkardım hadi bakalım demet akalın
            w=300, # yukardakinin aynısı
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
        # print_with_json(args.checkpoint + filename[0])
        checkpoint = torch.load(checkpoint_path, map_location=str(device))

    # encoder_image2 = checkpoint['encoder_image']
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

    print_with_json("Checkpoint_savepath:{}".format(args.savepath))
    print_with_json(
        "Encoder_image_mode:{}   Encoder_feat_mode:{}   Decoder_mode:{}".format(
            args.encoder_image_model, args.encoder_feat, args.decoder
        )
    )
    print_with_json(
        "encoder_layers {} decoder_layers {} n_heads {} dropout {} encoder_lr {} "
        "decoder_lr {}".format(
            args.n_layers, args.decoder_n_layers, args.n_heads, args.dropout, args.encoder_lr, args.decoder_lr
        )
    )

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)

    # Custom dataloaders
    # normalize seyleri degismeli
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
    # If your data elements are a custom type, or your collate_fn returns a batch that is a custom type.
    # Önce Resize ekliyoruz, sonra Normalize yapıyoruz
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(args.data_folder, args.data_name, "TRAIN", 
                    transform=transforms.Compose([
                        transforms.Resize((224, 224)),  # <--- BURAYI EKLEYİN
                        normalize
                    ])),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    num_train_optimization_steps = len(train_loader) * args.epochs 

    # 1. Encoder'ı Hazırla
    # --- KULLANIM ---
    # Eğitimde kullandığınız intra layer sayısını (args.intra_num_hidden_layers) buraya yazın.
    # Kodlarınızda varsayılan değer 9 görünüyordu.

    Clip_visual_encoder_module = CLIPVisualEncoder("/content/RSICC/ckpts/pytorch_model.bin.0",1024)

    # Wrapper'ın kendisini değişkene ata!
    clip_encoder_image = Clip_visual_encoder_module 

    # GPU'ya taşıdığınızdan emin olun (Sınıf içinde yaptıysanız bile garanti olsun)
    clip_encoder_image = clip_encoder_image.cuda()
    #clip_encoder_image = load_trained_visual_encoder("/content/RSICC/pytorch_model.bin.0", device)
    #print("Visual Encoder başarıyla ayıklandı.")

    """clip_encoder_optimizer, clip_encoder_scheduler, clip_encoder_image = prep_optimizer(
            args,
            clip_encoder_image,
            device,
            num_train_optimization_steps
        )"""

    # Epochs
    for epoch in range(start_epoch, args.epochs):

        # Decay learning rate if there is no improvement for x consecutive epochs, and terminate training after x
        if epochs_since_improvement == args.stop_criteria:
            print_with_json("the model has not improved in the last {} epochs".format(args.stop_criteria))
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 3 == 0:
            adjust_learning_rate(decoder_optimizer, 0.7)
            if args.fine_tune_encoder and encoder_image_optimizer is not None:
                print_with_json(encoder_image_optimizer)
                # adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        print_with_json(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
        train(
            args,
            train_loader=train_loader,
            encoder_image=encoder_image,
            clip_encoder_image=clip_encoder_image,
            encoder_feat=encoder_feat,
            decoder=decoder,
            criterion=criterion,
            encoder_image_optimizer=encoder_image_optimizer,
            #clip_encoder_optimizer=clip_encoder_optimizer,
            encoder_image_lr_scheduler=encoder_image_lr_scheduler,
            #clip_encoder_scheduler=clip_encoder_scheduler,
            encoder_feat_optimizer=encoder_feat_optimizer,
            encoder_feat_lr_scheduler=encoder_feat_lr_scheduler,
            decoder_optimizer=decoder_optimizer,
            decoder_lr_scheduler=decoder_lr_scheduler,
            epoch=epoch,
        )

        # -----------------------------------------------------------------------------------------------------
        # One epoch's validation
        print("-------------------------epoch passed-------------------------")
        metrics, nochange_metrics, change_metrics = evaluate_transformer(
            args, encoder_image=encoder_image,clip_encoder_image=clip_encoder_image, encoder_feat=encoder_feat, decoder=decoder
        )

        metrics_list.append(metrics)
        recent_bleu4 = metrics["Bleu_4"]
        bleu_4_output.append([metrics["Bleu_1"], metrics["Bleu_2"], metrics["Bleu_3"], metrics["Bleu_4"]])
        rogue_l_output.append(metrics["ROUGE_L"])
        #meteor1_output.append(metrics["METEOR"])
        cider_output.append(metrics["CIDEr"])
        bleu_4_nochange_output.append(
            [
                nochange_metrics["Bleu_1"],
                nochange_metrics["Bleu_2"],
                nochange_metrics["Bleu_3"],
                nochange_metrics["Bleu_4"],
            ]
        )
        rogue_l_nochange_output.append(nochange_metrics["ROUGE_L"])
        cider_nochange_output.append(nochange_metrics["CIDEr"])
        #meteor1_nochange_output.append(nochange_metrics["METEOR"])
        bleu_4_change_output.append(
            [change_metrics["Bleu_1"], change_metrics["Bleu_2"], change_metrics["Bleu_3"], change_metrics["Bleu_4"]]
        )
        rogue_l_change_output.append(change_metrics["ROUGE_L"])
        cider_change_output.append(change_metrics["CIDEr"])
        #meteor1_change_output.append(change_metrics["METEOR"])
        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print_with_json("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        checkpoint_name = (
            args.encoder_image_model + "_" + args.encoder_feat + "_" + args.decoder
        )  # _tengxun_aggregation
        """save_checkpoint_toplayan(
            args,
            checkpoint_name,
            epoch,
            epochs_since_improvement,
            encoder_image,
            encoder_feat,
            decoder,
            encoder_image_optimizer,
            encoder_feat_optimizer,
            decoder_optimizer,
            metrics,
            is_best,
            clip_encoder_image
        )"""
    train_model_sonuc_map["losses"] = losses_output
    train_model_sonuc_map["avg_losses"] = AVG_losses_output
    train_model_sonuc_map["top5_acc"] = top5_accuracy_output
    val_model_sonuc_map["rogue_l"] = rogue_l_output
    val_model_sonuc_map["cider"] = cider_output
    val_model_sonuc_map["bleu_4"] = bleu_4_output
    #val_model_sonuc_map["meteor"] = meteor1_output
    val_model_sonuc_map["rogue_l_nochange"] = rogue_l_nochange_output
    val_model_sonuc_map["cider_nochange"] = cider_nochange_output
    #val_model_sonuc_map["meteor_nochange"] = meteor1_nochange_output
    val_model_sonuc_map["bleu_4_nochange"] = bleu_4_nochange_output
    val_model_sonuc_map["rogue_l_change"] = rogue_l_change_output
    val_model_sonuc_map["cider_change"] = cider_change_output
    val_model_sonuc_map["bleu_4_change"] = bleu_4_change_output
    #val_model_sonuc_map["meteor_change"] = meteor1_change_output

    train_model_sonuc_json = json.dumps(train_model_sonuc_map, indent=4)
    val_model_sonuc_json = json.dumps(val_model_sonuc_map, indent=4)
    # Get the current date in the format YYYY-MM-DD
    current_date = datetime.date.today().strftime("%Y%m%d")

    # Define your save path
    output_save_path = args.savepath.replace("/model_dir", "")

    # Construct the filename with the current date
    file_name = f"{output_save_path}/train_{current_date}.json"
    file_name2 = f"{output_save_path}/val_{current_date}.json"
    file_name3 = f"{output_save_path}/terminal_text_{current_date}.txt"

    # Assuming you already have train_model_sonuc_json
    # Write the JSON data to the file
    with open(file_name3, "w") as dosya:
        dosya.write(text_terminal)
    with open(file_name, "w") as dosya:
        dosya.write(train_model_sonuc_json)
    with open(file_name2, "w") as dosya:
        dosya.write(val_model_sonuc_json)


current_date = datetime.date.today().strftime("%Y%m%d")


if __name__ == "__main__":
    print_with_json("bu toplayan modeldir.")
    dosya_index = 0
    folder_path = f"./model_sonucları/{current_date}_RSICCformerRGB_{dosya_index}"
    while os.path.exists(folder_path):
        # If it doesn't exist, create it
        print(f"Folder '{folder_path}' already exists.")
        dosya_index += 1
        folder_path = f"./model_sonucları/{current_date}_RSICCformerRGB_{dosya_index}"
    folder_path += "/model_dir"
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created successfully.")

    parser = argparse.ArgumentParser(description="Image_Change_Captioning")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Data parameters
    parser.add_argument(
        "--data_folder",
        default=r"Z:\createdFileBlackAUG",
        help="folder with data files saved by create_input_files.py.",
    )
    # parser.add_argument('--data_folder', default=r"C:\Users\TUBITAK\Desktop\RSICC_v2\SECONDCCpap\createdFileBlackAUG",
    #                     help='folder with data files saved by create_input_files.py.')
    parser.add_argument(
        "--data_name", default="LEVIR_CC_5_cap_per_img_10_min_word_freq", help="base name shared by data files."
    )
    # Model parameters
    parser.add_argument('--encoder_image', default="resnet101", help='which model does encoder use?')
    parser.add_argument("--encoder_image_model", default="clip4IDC", help="which model does encoder use?")
    parser.add_argument("--encoder_feat", default="MCCFormers_diff_as_Q")
    parser.add_argument("--decoder", default="trans")
    parser.add_argument("--n_heads", type=int, default=8, help="Multi-head attention in Transformer.")
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--decoder_n_layers", type=int, default=1)
    parser.add_argument("--feature_dim_de", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout")

    #params for CLIP4IDC implementation
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--decoder_model", default="decoder-base", type=str, required=False, help="Decoder module")
    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--intra_num_hidden_layers", type=int, default=9, help="Layer NO. of intra module")

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=40, help="number of epochs to train for (if early stopping is not triggered)."
    )
    parser.add_argument(
        "--stop_criteria", type=int, default=10, help="training stop if epochs_since_improvement == stop_criteria"
    )
    parser.add_argument("--batch_size", type=int, default=26, help="batch_size")
    parser.add_argument("--print_freq", type=int, default=100, help="print training/validation stats every __ batches.")
    parser.add_argument(
        "--workers", type=int, default=0, help="for data-loading; right now, only 0 works with h5pys in windows."
    )
    parser.add_argument(
        "--encoder_lr", type=float, default=5e-5, help="learning rate for encoder if fine-tuning."
    )  # en son 5e-5 yap
    parser.add_argument("--decoder_lr", type=float, default=5e-5, help="learning rate for decoder.")  # en son 5e-5 yap
    parser.add_argument("--clip_encoder_lr", type=float, default=0.0001, help="learning rate for CLIP fine-tuning.")    
    parser.add_argument("--grad_clip", type=float, default=5.0, help="clip gradients at an absolute value of.")
    parser.add_argument("--fine_tune_encoder", type=bool, default=True, help="whether fine-tune encoder or not")

    # parser.add_argument('--checkpoint', default="C:/Users\TUBITAK\Desktop\Turabi\model_sonucları/20231103_5/model_dir/", help='path to checkpoint, None if none.')
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
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    args = parser.parse_args()
    main(args)
    # folder_path = "./model_sonucları/20241029_RSICCformerSadeceSemantik/model_dir"
    #subprocess.run(
    #    f"python eval_v2_CNN_toplayan.py --data_folder {args.data_folder} --terminal_output {folder_path.replace('/model_dir','')} --path {folder_path} --beam_size {args.#beam_size} --data_name {args.data_name}"
    #)
