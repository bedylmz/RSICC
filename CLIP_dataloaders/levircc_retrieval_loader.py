import json
import os
from typing import Any
from typing import Literal

import logging

import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from dataloaders.raw_image_util import RawImageExtractor

logger = logging.getLogger(__name__)


def _extract_raw_sentences(change_captions: dict[str, Any]) -> list[str]:
    sentences_dict = change_captions["sentences"]

    sentences = [" ".join(sentence["tokens"]) for sentence in sentences_dict]
    #sentences = [sentence + " . " for sentence in sentences]
    return sentences


class LEVIRCC_DataLoader(Dataset):
    """LEVIR-CC dataset loader."""

    # images path = C:\Users\TUBITAK\Desktop\datasets\imagesRandAUG\train , A , B 
    # sem images path = C:\Users\TUBITAK\Desktop\datasets\imagesRandAUG\sem\train , A , B 
    # Json Path = C:\Users\TUBITAK\Desktop\datasets\imagesRandAUG\merged.json
    

    max_words = 64
    image_resolution = 256
    change_caption_file_name = "merged.json"
    before_image_folder = "A"
    after_image_folder = "B"

    before_semantic_folder = "A"
    after_semantic_folder = "B"

    def __init__(
        self,
        subset: Literal["train", "val", "test"],
        data_path: str | os.PathLike,
        tokenizer: Any,
    ):
        assert subset in ["train", "val", "test"]

        self.subset = subset
        self.data_path = data_path
        self.default_features_path = os.path.join(self.data_path, subset, "rgb")
        self.default_semantic_features_path = os.path.join(self.data_path, subset ,"sem")

        if tokenizer is not None:
            self.tokenizer = tokenizer
        #else:
        #    self.tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")

        change_caption_file_path = os.path.join(self.data_path, self.change_caption_file_name)

        print("*"*100)
        print(change_caption_file_path)
        print("*"*100)


        with open(change_caption_file_path) as fp:
            change_captions = json.load(fp)["images"]

        self.sentences_dict: list[tuple[str, str]] = []
        self.cut_off_points: list[int] = []
        for change_caption in change_captions:
            if change_caption["split"] != self.subset:
                continue

            image_name = os.path.join(change_caption["filename"])
            raw_sentences = _extract_raw_sentences(change_caption)

            for cap_txt in raw_sentences:
                self.sentences_dict.append((image_name, cap_txt))
            self.cut_off_points.append(len(self.sentences_dict))

        # below variables are used to multi-sentences retrieval
        # self.cut_off_points: used to tag the label when calculate the metric
        # self.sentence_num: used to cut the sentence representation
        # self.image_num: used to cut the image pair representation
        self.multi_sentence_per_pair = True  # !!! important tag for eval
        if self.subset == "val" or self.subset == "test":
            self.sentence_num = len(self.sentences_dict)
            self.image_num = len(self.cut_off_points)

        logger.info(f" {subset} Image number: {len(self.cut_off_points)}")
        logger.info(f" {subset} Total Pairs: {len(self.sentences_dict)}")

        self.sample_len = len(self.sentences_dict)
        self.rawImageExtractor = RawImageExtractor(size=self.image_resolution)
        self.SPECIAL_TOKEN = {
            "CLS_TOKEN": "<|startoftext|>",
            "SEP_TOKEN": "<|endoftext|>",
            "MASK_TOKEN": "[MASK]",
            "UNK_TOKEN": "[UNK]",
            "PAD_TOKEN": "[PAD]",
        }

        # if subset == "test" or subset == "val":
        # self.sample_len = 128

    def __len__(self):
        return self.sample_len

    def _get_text(self, caption):
        k = 1
        pairs_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.int64)

        words = self.tokenizer.tokenize(caption)

        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        # logger.info(f"Input caption words: {words}" + "<>" * 5)
        # logger.info(f"Input indis words: {input_ids}" + "<>" * 5)

        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == self.max_words
        assert len(input_mask) == self.max_words
        assert len(segment_ids) == self.max_words

        pairs_text[0] = np.array(input_ids)
        pairs_mask[0] = np.array(input_mask)
        pairs_segment[0] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment

    def _get_rawimage(self, image_path):
        # Pair x L x T x 3 x H x W
        image = np.zeros(
            (
                1,
                3,
                self.rawImageExtractor.size,
                self.rawImageExtractor.size,
            ),
            dtype=np.float32,
        )

        raw_image_data = self.rawImageExtractor.get_image_data(image_path)
        
        if isinstance(raw_image_data, dict):
            # Bazı modeller dict döndürüyor (örneğin eski CLIP)
            if "pixel_values" in raw_image_data:
                raw_image_data = raw_image_data["pixel_values"]
            elif "image" in raw_image_data:
                raw_image_data = raw_image_data["image"]
            else:
                raise KeyError(f"Beklenmeyen anahtarlar: {raw_image_data.keys()}")
        else:
            # Yeni sürümde AutoProcessor tensor döner
            pass  # zaten tensor olduğu için değiştirmiyoruz

            
        raw_image_data = raw_image_data.reshape(1, 3, 224, 224)


        image[0] = raw_image_data

        return image

    def __getitem__(self, idx):
        image_name, caption = self.sentences_dict[idx]
        bef_image_path = os.path.join(
            self.default_features_path,
            self.before_image_folder,
            image_name,
        )

        aft_image_path = os.path.join(
            self.default_features_path,
            self.after_image_folder,
            image_name,
        )

        bfr_image_semantic_path = os.path.join(
            self.default_semantic_features_path,
            self.before_semantic_folder,
            image_name,
        )

        aft_image_semantic_path = os.path.join(
            self.default_semantic_features_path,
            self.after_semantic_folder,
            image_name,
        )

        pairs_text, pairs_mask, pairs_segment = self._get_text(
            caption,
        )

        # logger.info("$" * 8 + f"Pairs text: {pairs_text}" + "$" * 8)

        bef_image = self._get_rawimage(bef_image_path)
        aft_image = self._get_rawimage(aft_image_path)
        bef_semantic = self._get_rawimage(bfr_image_semantic_path)
        aft_semantic = self._get_rawimage(aft_image_semantic_path)
        image_mask = np.ones(2, dtype=np.int64)

        return (
            pairs_text,
            pairs_mask,
            pairs_segment,
            bef_image,
            aft_image,
            bef_semantic,
            aft_semantic,
            image_mask,
        )
