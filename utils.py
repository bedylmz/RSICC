import os
import numpy as np
import h5py
import json
import torch
# from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
import imageio
from PIL import Image

from eval_func.bleu.bleu import Bleu
from eval_func.rouge.rouge import Rouge
from eval_func.cider.cider import Cider
from eval_func.meteor.meteor import Meteor
# from eval_func.spice.spice import Spice
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

import os
import numpy as np
import h5py
import json
import torch
# from scipy.misc import imread, imresize

from imageio import imread
from cv2 import resize as imresize
# from skimage.transform import resize

from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample

def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    assert dataset in {'RSICD', 'LEVIR_CC', 'Second_CC_RGB', 'Second_CC_Big','Second_CC_Little'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []

    val_image_paths = []
    val_image_captions = []

    test_image_paths = []
    test_image_captions = []

    word_freq = Counter()  # 创建一个空的Counter类(计数
    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])   # 其中c['tokens']是一个很多单词组成的句子‘列表’
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        if dataset == 'coco':
            path = os.path.join(image_folder, img['filepath'], img['filename'])
        elif dataset == 'LEVIR_CC':
            # FIXME:need to change for levir_CC
            path1 = os.path.join(image_folder, img['split'], 'A', img['filename'])
            path2 = os.path.join(image_folder, img['split'], 'B', img['filename'])
            path = [path1,path2]
        elif dataset == 'Second_CC_RGB':
            path1 = os.path.join(image_folder, img['split'], 'rgb', 'A', img['filename'])
            path2 = os.path.join(image_folder, img['split'], 'rgb', 'B', img['filename'])
            path = [path1,path2]
        elif dataset == 'Second_CC_Little' or dataset == 'Second_CC_Big':
            path1 = os.path.join(image_folder, img['split'], 'rgb', 'A', img['filename'])
            path2 = os.path.join(image_folder, img['split'], 'rgb', 'B', img['filename'])
            path = [path1,path2]
        else:
            path = os.path.join(image_folder, img['filename'])

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            if dataset =='LEVIR_CC' or dataset =='Second_CC_RGB'or dataset =='Second_CC_Little'or dataset =='Second_CC_Big':
                images = h.create_dataset('images', (len(impaths), 2, 3, 256, 256), dtype='uint8')
            else:
                images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                # Read images
                if dataset =='LEVIR_CC' or dataset =='Second_CC_RGB'or dataset =='Second_CC_Little'or dataset =='Second_CC_Big':
                    img_A = imread(impaths[i][0])
                    img_B = imread(impaths[i][1])
                    if len(img_A.shape) == 2:
                        img_A = img_A[:, :, np.newaxis]
                        img_A = np.concatenate([img_A, img_A, img_A], axis=2)
                    if len(img_B.shape) == 2:
                        img_B = img_B[:, :, np.newaxis]
                        img_B = np.concatenate([img_B, img_B, img_B], axis=2)
                    img_A = imresize(img_A, (256, 256))
                    img_A = img_A.transpose(2, 0, 1)
                    img_B = imresize(img_B, (256, 256))
                    img_B = img_B.transpose(2, 0, 1)
                    assert img_A.shape == (3, 256, 256)
                    assert img_B.shape == (3, 256, 256)
                    assert np.max(img_A) <= 255
                    assert np.max(img_B) <= 255

                    # Save image to HDF5 file
                    # images[i][0] = img_A
                    # images[i][1] = img_B
                    images[i] = [img_A,img_B]

                else:
                    img = imread(impaths[i])
                    if len(img.shape) == 2:
                        img = img[:, :, np.newaxis]
                        img = np.concatenate([img, img, img], axis=2)
                    img = imresize(img, (256, 256))
                    img = img.transpose(2, 0, 1)
                    assert img.shape == (3, 256, 256)
                    assert np.max(img) <= 255
                    # Save image to HDF5 file
                    images[i] = img

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)


            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)

def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)

def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)

def get_eval_score(references, hypotheses):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    hypo = [[' '.join(hypo)] for hypo in [[str(x) for x in hypo] for hypo in hypotheses]]
    ref = [[' '.join(reft) for reft in reftmp] for reftmp in
           [[[str(x) for x in reft] for reft in reftmp] for reftmp in references]]

    score = []
    method = []
    for scorer, method_i in scorers:
        score_i, scores_i = scorer.compute_score(ref, hypo)
        score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
        method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)
        print("{} {}".format(method_i, score_i))
    score_dict = dict(zip(method, score))

    return score_dict

def convert2words(sequences, rev_word_map):
    for l1 in sequences:
        caption = ""
        for l2 in l1:
            caption += rev_word_map[l2]
            caption += " "
        print(caption)

#---------------------------CUSTOM TOKENİZER-------------------------------------------

def bridge_embeddings_and_transfer(rsicc_decoder, clip_model, clip_tokenizer, rsicc_word_map):
    """
    RSICC'nin Word-Level kelime haritası ile CLIP'in BPE vektörleri arasında köprü kurar.
    Dimension mismatch (512 vs 1024) durumunda vektörleri tekrarlayarak transfer eder.
    """
    print(f">>> CLIP ve RSICC Tokenizer Köprüsü Kuruluyor...")
    
    # 1. Hedef ve Kaynak Embedding Matrisleri
    rsicc_emb_layer = rsicc_decoder.vocab_embedding
    clip_emb_weight = clip_model.token_embedding.weight # [49408, 512]
    
    # RSICC Embedding boyutları
    vocab_size = rsicc_emb_layer.weight.shape[0]
    embed_dim = rsicc_emb_layer.weight.shape[1] # Muhtemelen 1024
    
    # İstatistikler
    found_count = 0
    total_count = len(rsicc_word_map)
    
    with torch.no_grad():
        for word, rsicc_id in rsicc_word_map.items():
            # CLIP için özel token eşleştirmeleri
            if word == '<start>':
                clip_tokens = clip_tokenizer.encode("<|startoftext|>")
            elif word == '<end>':
                clip_tokens = clip_tokenizer.encode("<|endoftext|>")
            elif word == '<pad>' or word == '<unk>':
                continue
            else:
                clip_tokens = clip_tokenizer.encode(word)
            
            if len(clip_tokens) > 0:
                clip_indices = torch.tensor(clip_tokens).to(clip_emb_weight.device)
                vectors = clip_emb_weight[clip_indices]
                
                # Kelime parçalarının ortalamasını al -> [512]
                avg_vector = torch.mean(vectors, dim=0)
                
                # --- BOYUT UYUŞMAZLIĞI DÜZELTMESİ ---
                if avg_vector.shape[0] != embed_dim:
                    if embed_dim == 1024 and avg_vector.shape[0] == 512:
                         # 512'lik vektörü 2 kere yan yana koyarak 1024 yapıyoruz (Concatenate)
                         # Bu sayede bilgi korunur ve 1024 boyuta ulaşılır.
                         avg_vector = torch.cat([avg_vector, avg_vector], dim=0)
                    else:
                        # Eğer başka bir boyut farkı varsa atla ve uyar
                        # print(f"Skip: {word} dim mismatch {avg_vector.shape} vs {embed_dim}")
                        continue
                
                # Kopyala
                if rsicc_id < vocab_size:
                    rsicc_emb_layer.weight[rsicc_id].copy_(avg_vector)
                    found_count += 1

    print(f">>> Embedding Transfer Tamamlandı: {found_count}/{total_count} kelime CLIP'ten aktarıldı.")

    # ---------------------------------------------------------
    # 2. Transformer Katmanlarını Transfer Et
    # ---------------------------------------------------------
    print(">>> Transformer Katmanları Kontrol Ediliyor...")
    clip_layers = clip_model.transformer.resblocks
    decoder_layers = rsicc_decoder.transformer.layers 
    
    min_layers = min(len(clip_layers), len(decoder_layers))
    
    transferred_layers = 0
    with torch.no_grad():
        for i in range(min_layers):
            c_layer = clip_layers[i]
            d_layer = decoder_layers[i]
            
            # Boyut kontrolü: Eğer CLIP (512) ve Decoder (1024) uyuşmuyorsa transferi atla.
            if c_layer.attn.in_proj_weight.shape != d_layer.self_attn.in_proj_weight.shape:
                continue

            # Boyutlar uyuşuyorsa (örn: feature_dim_de=512 yaptıysanız) transfer et
            d_layer.self_attn.in_proj_weight.data.copy_(c_layer.attn.in_proj_weight.data)
            d_layer.self_attn.in_proj_bias.data.copy_(c_layer.attn.in_proj_bias.data)
            d_layer.self_attn.out_proj.weight.data.copy_(c_layer.attn.out_proj.weight.data)
            d_layer.self_attn.out_proj.bias.data.copy_(c_layer.attn.out_proj.bias.data)

            d_layer.norm1.weight.data.copy_(c_layer.ln_1.weight.data)
            d_layer.norm1.bias.data.copy_(c_layer.ln_1.bias.data)
            d_layer.norm3.weight.data.copy_(c_layer.ln_2.weight.data)
            d_layer.norm3.bias.data.copy_(c_layer.ln_2.bias.data)

            d_layer.linear1.weight.data.copy_(c_layer.mlp.c_fc.weight.data)
            d_layer.linear1.bias.data.copy_(c_layer.mlp.c_fc.bias.data)
            d_layer.linear2.weight.data.copy_(c_layer.mlp.c_proj.weight.data)
            d_layer.linear2.bias.data.copy_(c_layer.mlp.c_proj.bias.data)
            
            transferred_layers += 1
            
    if transferred_layers == 0:
        print(f">>> UYARI: Boyut farkı (512 vs {embed_dim}) nedeniyle Transformer katmanları transfer EDİLEMEDİ.")
        print(">>> Sadece Embedding katmanı (çoğaltılarak) transfer edildi, model sıfırdan öğrenecek.")
    else:
        print(f">>> {transferred_layers} Transformer katmanı başarıyla transfer edildi.")