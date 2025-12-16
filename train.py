import sys
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import subprocess
from torch.nn.utils.rnn import pack_padded_sequence
import argparse
from torch.optim.lr_scheduler import StepLR
import datetime

from models import CNN_Encoder
from models_RSICCformerDfusion import *
from datasets import *
from eval import evaluate_transformer
import sys

from utils import *

from torchvision.transforms.functional import to_pil_image

seed = 1
torch.manual_seed(seed)

def train(
    args,
    train_loader,
    encoder_image,
    encoder_feat,
    decoder,
    criterion,
    encoder_feat_optimizer,
    encoder_feat_lr_scheduler,
    decoder_optimizer,
    decoder_lr_scheduler,
    epoch,
    encoder_image_optimizer=None,
    encoder_image_lr_scheduler=None
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
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, (img_pairs, caps, caplens) in enumerate(train_loader):
        #         if i == 20:
        #             break
        data_time.update(time.time() - start)

        # Back prop.
        decoder_optimizer.zero_grad()
        encoder_feat_optimizer.zero_grad()

        # Move to GPU, if available
        img_pairs = img_pairs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        # Eklemek ve carparak eklemeyi de dene
        imgs_A = img_pairs[:, 0, :, :, :]
        imgs_B = img_pairs[:, 1, :, :, :]

        # imgs_C = img_pairs[:, 2, :, :, :]
        # sem_A = img_pairs[:, 2, :, :, :]
        # sem_B = img_pairs[:, 3, :, :, :]

        # rsformer image encoder 
        imgs_A = encoder_image(imgs_A)  # imgs_A: [batch_size,1024, 14, 14]
        imgs_B = encoder_image(imgs_B)  # batch time = 0.35

        
        fused_feat = encoder_feat(
            imgs_A,
            imgs_B
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

        if encoder_image_optimizer is not None:
            encoder_image_optimizer.zero_grad()
  
        loss.backward()

        # Clip gradients
        if args.grad_clip is not None:
            clip_gradient(decoder_optimizer, args.grad_clip)
            if encoder_image_optimizer is not None:
                clip_gradient(encoder_image_optimizer, args.grad_clip)

        # Update weights
        decoder_optimizer.step()
        decoder_lr_scheduler.step()
        
        encoder_feat_optimizer.step()
        encoder_feat_lr_scheduler.step()

        if encoder_image_optimizer is not None:
            encoder_image_optimizer.step()
            
        if encoder_image_lr_scheduler is not None:
            encoder_image_lr_scheduler.step()
   
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


def save_checkpoint(args, data_name, epoch, epochs_since_improvement, 
                    encoder_image, encoder_feat, decoder, 
                    encoder_image_optimizer, encoder_feat_optimizer, decoder_optimizer):
    import torch
    
    state = {
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        
        # --- Modeller ---
        # MCCFormers Feature Encoder
        'encoder_feat': encoder_feat.state_dict(),
        # Transformer Decoder
        'decoder': decoder.state_dict(),

        # --- Optimizerlar ---
        'encoder_feat_optimizer': encoder_feat_optimizer.state_dict(),
        'decoder_optimizer': decoder_optimizer.state_dict(),
    }

    # Eski ResNet encoder varsa (Opsiyonel)
    if encoder_image is not None:
        state['encoder_image'] = encoder_image.state_dict()
    if encoder_image_optimizer is not None:
        state['encoder_image_optimizer'] = encoder_image_optimizer.state_dict()

    # Kayıt Dizini Kontrolü
    directory = args.save_model_path
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 1. En Son Checkpoint'i Kaydet (Her epoch'ta üzerine yazar)
    filename = os.path.join(directory, 'checkpoint_' + data_name + '.pth.tar')
    torch.save(state, filename)


def main(args, meteor_output=None):
    print(args)
    global metrics_list
    print(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))

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

    # set the encoder_dim
    encoder_image_dim = 1024 # resnet101
    # filename = os.listdir(args.checkpoint)
    # checkpoint_path = os.path.join(args.checkpoint, filename[0])
    # print(args.checkpoint + filename[0])
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
        # print(args.checkpoint + filename[0])
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

    print("Checkpoint_savepath:{}".format(args.savepath))
    print(
        "Encoder_image_mode:{}   Encoder_feat_mode:{}   Decoder_mode:{}".format(
            args.encoder_image, args.encoder_feat, args.decoder
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

    # Epochs
    for epoch in range(start_epoch, args.epochs):
        # One epoch's training
        print(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
        train(
            args,
            train_loader=train_loader,
            encoder_image=encoder_image,
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
        )

        # Mevcut metrik hesaplamaları (Evaluate transformer) - LOGLAMA İÇİN KALSIN
        metrics, nochange_metrics, change_metrics = evaluate_transformer(
            args, encoder_image=encoder_image,encoder_feat=encoder_feat, decoder=decoder
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
                            encoder_image_optimizer, encoder_feat_optimizer, decoder_optimizer)
            
         # Early Stopping
        if epochs_since_improvement == args.stop_criteria:
            print(f"Early stopping triggered! Validation metrics hasn't increased for {args.stop_criteria} epochs.")
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 3 == 0:
            adjust_learning_rate(decoder_optimizer, 0.7)
            if args.fine_tune_encoder and encoder_image_optimizer is not None:
                print(encoder_image_optimizer)
                # adjust_learning_rate(encoder_optimizer, 0.8)
    


if __name__ == "__main__":

    folder_path = ""
    
    parser = argparse.ArgumentParser(description="Image_Change_Captioning")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Data parameters
    parser.add_argument("--data_name", default="LEVIR_CC_5_cap_per_img_10_min_word_freq", help="base name shared by data files.")
    parser.add_argument("--data_folder", default="")
    parser.add_argument("--cross_model", default="")
    parser.add_argument("--save_model_path", default="")
    # Model parameters
    parser.add_argument('--encoder_image', default="resnet101", help='which model does encoder use?')
    parser.add_argument("--encoder_feat", default="MCCFormers_diff_as_Q")
    parser.add_argument("--decoder", default="trans")
    parser.add_argument("--n_heads", type=int, default=8, help="Multi-head attention in Transformer.")
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--decoder_n_layers", type=int, default=1)
    parser.add_argument("--feature_dim_de", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=40, help="number of epochs to train for (if early stopping is not triggered).")
    parser.add_argument("--stop_criteria", type=int, default=10, help="training stop if epochs_since_improvement == stop_criteria")
    parser.add_argument("--batch_size", type=int, default=28, help="batch_size")
    parser.add_argument("--print_freq", type=int, default=100, help="print training/validation stats every __ batches.")
    parser.add_argument("--workers", type=int, default=0, help="for data-loading; right now, only 0 works with h5pys in windows.")
    parser.add_argument("--encoder_lr", type=float, default=5e-5, help="learning rate for encoder if fine-tuning.")  
    parser.add_argument("--decoder_lr", type=float, default=5e-5, help="learning rate for decoder.") 
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
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    args = parser.parse_args()
    main(args)