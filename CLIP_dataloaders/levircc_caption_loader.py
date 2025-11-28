import json
import os
from typing import Any

import logging

import numpy as np
from torch.utils.data import Dataset

from dataloaders.raw_image_util import RawImageExtractor

logger = logging.getLogger(__name__)


def _extract_raw_sentences(change_captions: dict[str, Any]) -> list[str]:
    sentences_dict = change_captions["sentences"]

    sentences = [" ".join(sentence["tokens"]) for sentence in sentences_dict]
    sentences = [sentence + " . " for sentence in sentences]
    return sentences


class LEVIRCC_DataLoader(Dataset):
    """LEVIRCC dataset loader."""

    

    change_caption_file_name = "merged.json"

    before_image_folder = "A"
    after_image_folder = "B"

    before_semantic_folder = "A"
    after_semantic_folder = "B"

    def __init__(
        self,
        subset,
        data_path,
        tokenizer,
        max_words=30,
        image_resolution=224,
    ):
        self.data_path = data_path
        self.default_features_path = os.path.join(self.data_path, "train", "rgb")
        self.semantic_default_features_path = os.path.join(self.data_path, "train" ,"sem")




        logger.warning(">" * 50)
        logger.warning(f"Default features path: {self.default_features_path}")
        logger.warning(">" * 50)
        self.max_words = max_words
        self.tokenizer = tokenizer

        self.subset = subset
        assert self.subset in ["train", "val", "test"]

        change_caption_file_path = os.path.join(self.data_path, self.change_caption_file_name)

        with open(change_caption_file_path) as fp:
            change_captions = json.load(fp)["images"]

        self.sample_len = 0
        self.sentences_dict: list[tuple[str, str]] = []
        self.cut_off_points = []

        for change_caption in change_captions:
            if change_caption["split"] != self.subset:
                continue

            image_name = os.path.join(change_caption["filename"])
            raw_sentences = _extract_raw_sentences(change_caption)
            if self.subset == "test" or self.subset == "val":
                raw_sentences = raw_sentences[:1]

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
            self.image_num = len(change_captions)

            logger.info(f"For {self.subset}, sentence number: {self.sentence_num}")
            logger.info(f"For {self.subset}, image number: {self.image_num}")

        print(f"Image number: {len(change_captions)}")
        print(f"Total Paire: {len(self.sentences_dict)}")

        self.sample_len = len(self.sentences_dict)
        self.rawImageExtractor = RawImageExtractor(size=image_resolution)
        self.SPECIAL_TOKEN = {
            "CLS_TOKEN": "<|startoftext|>",
            "SEP_TOKEN": "<|endoftext|>",
            "MASK_TOKEN": "[MASK]",
            "UNK_TOKEN": "[UNK]",
            "PAD_TOKEN": "[PAD]",
        }

        # # if subset is test or val then return 10 items
        # if self.subset == "test" or self.subset == "val":
        # self.sample_len = 256

    def __len__(self):
        return self.sample_len

    def _get_text(self, caption):
        k = 1
        pairs_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.int64)

        pairs_input_caption_ids = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_output_caption_ids = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_decoder_mask = np.zeros((k, self.max_words), dtype=np.int64)

        words = self.tokenizer.tokenize(caption)

        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
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

        # For generate captions
        if caption is not None:
            # logger.info("#" * 5 + f"Caption: {caption}")
            caption_words = self.tokenizer.tokenize(caption)
        if len(caption_words) > total_length_with_CLS:
            caption_words = caption_words[:total_length_with_CLS]
        input_caption_words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + caption_words
        output_caption_words = caption_words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

        # logger.info(f"Input caption words: {input_caption_words}")
        # logger.info(f"Output caption words: {output_caption_words}" + "<>" * 5)

        # For generate captions
        input_caption_ids = self.tokenizer.convert_tokens_to_ids(input_caption_words)
        output_caption_ids = self.tokenizer.convert_tokens_to_ids(output_caption_words)

        decoder_mask = [1] * len(input_caption_ids)
        while len(input_caption_ids) < self.max_words:
            input_caption_ids.append(0)
            output_caption_ids.append(0)
            decoder_mask.append(0)
        assert len(input_caption_ids) == self.max_words
        assert len(output_caption_ids) == self.max_words
        assert len(decoder_mask) == self.max_words

        pairs_input_caption_ids[0] = np.array(input_caption_ids)
        pairs_output_caption_ids[0] = np.array(output_caption_ids)
        pairs_decoder_mask[0] = np.array(decoder_mask)

        return (
            pairs_text,
            pairs_mask,
            pairs_segment,
            pairs_input_caption_ids,
            pairs_decoder_mask,
            pairs_output_caption_ids,
        )

    def _get_rawimage(self, image_path):
        choice_image_path = [image_path]
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
        raw_image_data = raw_image_data["image"].reshape(1, 3, 224, 224)

        image[0] = raw_image_data

        return image

    def __getitem__(self, idx):
        image_name, caption = self.sentences_dict[idx]

        # logger.info("><" * 30)
        # logger.info(f"Image name: {image_name}")
        # logger.info(f"Caption: {caption}")
        # logger.info("><" * 30)

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

        (
            pairs_text,
            pairs_mask,
            pairs_segment,
            pairs_input_caption_ids,
            pairs_decoder_mask,
            pairs_output_caption_ids,
        ) = self._get_text(caption)

        # logger.info("$" * 8 + f"Pairs text: {pairs_text}" + "$" * 8)
        # logger.info("$" * 8 + f"Pairs input caption: {pairs_input_caption_ids}" + "$" * 8)

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
            pairs_input_caption_ids,
            pairs_decoder_mask,
            pairs_output_caption_ids,
            image_name,
        )
