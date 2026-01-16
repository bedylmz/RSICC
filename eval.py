import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import time
from torch.optim.lr_scheduler import StepLR
from torch import nn
from typing import Optional
from classes import *
import logging
import torch.backends.cudnn as cudnn
from models import MCCFormers_diff_as_Q, DecoderTransformer, CNN_Encoder



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BatchNormalize(torch.nn.Module):
    def __init__(self, mean, std, device):
        super().__init__()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, 3, 1, 1).to(device)

    def forward(self, tensor):
        # Tensor shape: [Batch, 3, H, W]
        return (tensor - self.mean) / self.std


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

    with open('eval_results_fortest/' + args.Split +'/'+args.encoder_image + "_" +args.encoder_feat+"_" +args.decoder + '_res.json', 'w') as f:
        json.dump(result_json_file, f)

    with open('eval_results_fortest/' + args.Split +'/'+ args.encoder_image + "_" +args.encoder_feat+"_" +args.decoder + '_gts.json', 'w') as f:
        json.dump(reference_json_file, f)

def get_key(dict_, value):
  return [k for k, v in dict_.items() if v == value]

def evaluate_transformer_caption(
        args: argparse.Namespace = None,
        encoder_feat: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        clip_encoder_image: Optional[nn.Module] = None,
        encoder_image: Optional[nn.Module] = None,
        layerNormalizeLayer: Optional[nn.Module] = None,
        adaptLayer: Optional[nn.Module] = None,        # Burası
        adaptLayerClip: Optional[nn.Module] = None,
        gateSelf: Optional[nn.Module] = None,
        logger = None,

        ):
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

        for i, (img_pairs, caps, caplens, allcaps) in enumerate(
                tqdm(loader, desc=args.Split + " EVALUATING AT BEAM SIZE " + str(beam_size))):
            # 5 image is the same when "shuffle=False" of the dataloader
            if (i + 1) % 5 != 0:
                continue
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

                encoder_out = encoder_feat(
                    final_A,
                    final_B,
                ) # encoder_out: (S, batch, feature_dim) # fused_feat: (S, batch, feature_dim) # buyuk tensor atama yavaslatior (#batch time = 0.5)
            elif(args.eval_just_RSICC):

                imgs_A_resnet = norm_resnet(imgs_A) # ResNet için normalize et
                imgs_B_resnet = norm_resnet(imgs_B)

                imgs_A = encoder_image(imgs_A_resnet)
                imgs_B = encoder_image(imgs_B_resnet)  # encoder_image :[1, 1024,14,14]

                encoder_out = encoder_feat(imgs_A, imgs_B) # encoder_out: (S, batch, feature_dim)
            else:
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

                encoder_out = encoder_feat(
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

    return hypotheses, references

def evaluate_transformer(
        args: argparse.Namespace = None,
        encoder_feat: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        clip_encoder_image: Optional[nn.Module] = None,
        encoder_image: Optional[nn.Module] = None,
        layerNormalizeLayer: Optional[nn.Module] = None,
        adaptLayer: Optional[nn.Module] = None,        # Burası
        adaptLayerClip: Optional[nn.Module] = None,
        gateSelf: Optional[nn.Module] = None,
        logger = None,

        ):
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

        for i, (img_pairs, caps, caplens, allcaps) in enumerate(
                tqdm(loader, desc=args.Split + " EVALUATING AT BEAM SIZE " + str(beam_size))):
            # 5 image is the same when "shuffle=False" of the dataloader
            if (i + 1) % 5 != 0:
                continue
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

                encoder_out = encoder_feat(
                    final_A,
                    final_B,
                ) # encoder_out: (S, batch, feature_dim) # fused_feat: (S, batch, feature_dim) # buyuk tensor atama yavaslatior (#batch time = 0.5)
            elif(args.eval_just_RSICC):

                imgs_A_resnet = norm_resnet(imgs_A) # ResNet için normalize et
                imgs_B_resnet = norm_resnet(imgs_B)

                imgs_A = encoder_image(imgs_A_resnet)
                imgs_B = encoder_image(imgs_B_resnet)  # encoder_image :[1, 1024,14,14]

                encoder_out = encoder_feat(imgs_A, imgs_B) # encoder_out: (S, batch, feature_dim)
            else:
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

                encoder_out = encoder_feat(
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

        # captions
        # save_captions(args, word_map, hypotheses, references)

    logger.info(f"len(nochange_references): {len(nochange_references)}")
    logger.info(f"len(change_references): {len(change_references)}")
    # Calculate BLEU1~4, METEOR, ROUGE_L, CIDEr scores
    if len(nochange_references)>0:
        logger.info('nochange_metric:')
        nochange_metric = get_eval_score(nochange_references, nochange_hypotheses, logger)
        logger.info(f"nochange_acc: {nochange_acc / len(nochange_references)}")
    if len(change_references)>0:
        logger.info('change_metric:')
        change_metric = get_eval_score(change_references, change_hypotheses, logger)
        logger.info(f"change_acc: {change_acc / len(change_references)}")
    logger.info(".......................................................")
    metrics = get_eval_score(references, hypotheses, logger)


    return metrics

def main(args):
    global logger 
    logger = get_logger(os.path.join(args.save_model_path, "log.txt"))
    logger.info(f"Kullanılan Random Seed: {seed}") # İleride gerekirse tekrar üretmek için
    if args.eval_mode:
        logging.disable(logging.CRITICAL)

    logger.info(args)
    global metrics_list

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

        clip = CLIPVisualEncoder(args, args.clip_path)

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

    if(args.dual_branch == True):
        encoder_image_lr_scheduler = (
            StepLR(encoder_image_optimizer, step_size=900, gamma=1) if args.fine_tune_encoder else None
        )
    
    # Move to GPU, if available
    if(args.dual_branch == True or args.eval_just_RSICC == True):
        encoder_image = encoder_image.to(device)
    encoder_feat = encoder_feat.to(device)
    decoder = decoder.to(device)
    
    # Custom dataloaders
    # Burada SADECE boyutlandırma yapıyoruz. Normalizasyonu döngü içine taşıdık.
    # CLIP genelde Bicubic sever, ResNet de buna uyum sağlar.
    train_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC)
    ])

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

    # ---------------------------- EVAL SECTION ----------------------------
    if(args.eval_mode):
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

        

        if(args.dual_branch):
              if(args.gate):
                evaluate_transformer(
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
                evaluate_transformer(
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
                evaluate_transformer(
                        args, 
                        encoder_image=encoder_image, 
                        encoder_feat=encoder_feat,
                        decoder=decoder,
                        logger=logger,
                        )
        else:
                evaluate_transformer(
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
    