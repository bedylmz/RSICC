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


import torch
import os
import argparse
from CLIP_modules.modeling import CLIP4IDC
from CLIP_dataloaders.raw_image_util import RawImageExtractor

from CLIP_modules.module_clip import CLIP

# 1. Konfigürasyon (Modeli init etmek için gerekli argümanlar)
# Eğittiğiniz modelin parametreleriyle (katman sayısı vb.) uyumlu olmalıdır.
class ModelConfig:
    cross_model = "cross-base"
    decoder_model = "decoder-base"
    cache_dir = ""
    type_vocab_size = 2
    task_type = "retrieval" 
    linear_patch = "2d" # Eğer 3d kullandıysanız değiştirin
    local_rank = 0
    # Eğitimde kullandığınız diğer önemli configler buraya eklenebilir
    intra_num_hidden_layers = 9 # Varsayılan değerler (eğitimde değiştirdiyseniz güncelleyin)
    pretrained_clip_name = "ViT-B/32"

def load_trained_visual_encoder(checkpoint_path, device):
    """
    Eğitilmiş checkpoint'ten sadece visual encoder'ı döndürür.
    """
    args = ModelConfig()
    
    # Checkpoint'i yükle
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint bulunamadı: {checkpoint_path}")
        
    print(f"Model yükleniyor: {checkpoint_path}")
    
    # State dict'i CPU'ya yükleyerek bellek tasarrufu yapalım
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # Eğer checkpoint optimizer durumunu da içeriyorsa (genelde epoch ile biten dosyalarda olur),
    # sadece model ağırlıklarını almalıyız. Ancak modeling.py içindeki from_pretrained
    # genellikle sadece model weight bekler. Eğer 'model_state_dict' gibi bir key altındaysa:
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    
    # Modeli Başlat (CLIP4IDC yapısı)
    # Not: from_pretrained fonksiyonu modeling.py içinde state_dict parametresi alabiliyor.
    model = CLIP4IDC.from_pretrained(
        args.cross_model, 
        args.decoder_model, 
        state_dict=state_dict, 
        task_config=args
    )
    
    model.to(device)
    model.eval()
    
    # Bütün modeli kullanmak yerine sadece CLIP'in visual modülünü alıyoruz.
    # modeling.py incelemesine göre yapı: model -> clip -> visual
    visual_encoder = model.clip.visual
    
    return visual_encoder

def load_custom_visual_encoder(checkpoint_path, device, intra_num_hidden_layers=9):
    """
    CLIP4IDC wrapper'ı olmadan, INTRA layerları da içeren
    özelleştirilmiş Visual Encoder'ı yükler.
    
    Args:
        intra_num_hidden_layers: Eğitimde kullandığınız intra layer sayısı (varsayılan: 9)
    """
    print(f"Checkpoint yükleniyor: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    # --- 1. Model Parametrelerini Algıla ---
    # Modelin boyutlarını checkpointten okuyoruz
    embed_dim = state_dict["clip.text_projection"].shape[1]
    context_length = state_dict["clip.positional_embedding"].shape[0]
    vocab_size = state_dict["clip.token_embedding.weight"].shape[0]
    transformer_width = state_dict["clip.ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("clip.transformer.resblocks")))

    vision_width = state_dict["clip.visual.conv1.weight"].shape[0]
    vision_layers = len([k for k in state_dict.keys() if k.startswith("clip.visual.") and k.endswith(".attn.in_proj_weight")])
    
    # Patch size hesaplama
    vision_patch_size = state_dict["clip.visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["clip.visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size

    print(f"Model Yapılandırması: Res={image_resolution}, VisLayers={vision_layers}, IntraLayers={intra_num_hidden_layers}")

    # --- 2. CLIP Modelini INTRA Parametresiyle Başlat ---
    # ÖNEMLİ: intra_layers parametresini buraya ekliyoruz!
    model = CLIP(
        embed_dim,
        image_resolution,
        vision_layers,
        vision_width,
        vision_patch_size,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
        intra_layers=intra_num_hidden_layers  # <--- BURASI KRİTİK
    ).float()

    # --- 3. Ağırlıkları Yükle ---
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("clip."):
            new_key = k[5:] 
            new_state_dict[new_key] = v
            
    # strict=False kullanıyoruz çünkü text encoder ağırlıklarını yüklemesek de olur
    # ama visual encoder eksiksiz yüklenecektir.
    msg = model.load_state_dict(new_state_dict, strict=False)
    print("Yükleme Durumu:", msg)
    
    model.to(device)
    model.eval()
    
    # Görsel encoder'ı döndür
    return model.visual

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
    clip_encoder_scheduler,
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

        clip_imgs_A = img_pairs[:, 0, :, :, :]
        clip_imgs_B = img_pairs[:, 1, :, :, :]
        clip_encoded = []

        # imgs_C = img_pairs[:, 2, :, :, :]
        # sem_A = img_pairs[:, 2, :, :, :]
        # sem_B = img_pairs[:, 3, :, :, :]

        # rsformer image encoder 
        #imgs_A = encoder_image(imgs_A)  # imgs_A: [batch_size,1024, 14, 14]
        #imgs_B = encoder_image(imgs_B)  # batch time = 0.35
        # Convert a single tensor image (C,H,W) in range [0,1] or [0,255] to PIL
        
        # --new--
        imgs_A = clip_encoder_image(img_pairs[:, 0, :, :, :])
        imgs_B = clip_encoder_image(img_pairs[:, 1, :, :, :])

        """to_pil = transforms.ToPILImage()

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
            imgs_A,
            imgs_B,
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
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(args.data_folder, args.data_name, "TRAIN", transform=transforms.Compose([normalize])),
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
    clip_encoder_image = load_custom_visual_encoder(
        "/content/RSICC/pytorch_model.bin.0", 
        device, 
        intra_num_hidden_layers=9
    )
    #clip_encoder_image = load_trained_visual_encoder("/content/RSICC/pytorch_model.bin.0", device)
    #print("Visual Encoder başarıyla ayıklandı.")

    clip_encoder_optimizer, clip_encoder_scheduler, clip_encoder_image = prep_optimizer(
            args,
            clip_encoder_image,
            device,
            num_train_optimization_steps
        )

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
            clip_encoder_optimizer=clip_encoder_optimizer,
            encoder_image_lr_scheduler=encoder_image_lr_scheduler,
            clip_encoder_scheduler=clip_encoder_scheduler,
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
