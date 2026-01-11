import torch
from torch.utils.data import DataLoader

from CLIP_dataloaders.levircc_retrieval_loader import LEVIRCC_DataLoader
from CLIP_dataloaders.levircc_caption_loader import LEVIRCC_DataLoader as LEVIRCC_DataLoader_Caption


def dataloader_levircc_train(args, tokenizer):
    if args.task_type == "retrieval":
        DataSet_DataLoader = LEVIRCC_DataLoader
    else:
        DataSet_DataLoader = LEVIRCC_DataLoader_Caption

    levircc_dataset = DataSet_DataLoader(
        subset="train",
        data_path=args.data_path,
        tokenizer=tokenizer,
    )

    dataloader = DataLoader(
        levircc_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
    )

    return dataloader, len(levircc_dataset)


def dataloader_levircc_test(args, tokenizer, subset="test"):
    if args.task_type == "retrieval":
        DataSet_DataLoader = LEVIRCC_DataLoader
    else:
        DataSet_DataLoader = LEVIRCC_DataLoader_Caption

    if( subset == 'train1'):
        subset = 'train'

    levircc_testset = DataSet_DataLoader(
        subset=subset,  # type: ignore[arg-type]
        data_path=args.data_path,
        tokenizer=tokenizer,
    )
    dataloader_levircc = DataLoader(
        levircc_testset,
        batch_size=args.batch_size_val,
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_levircc, len(levircc_testset)


DATALOADER_DICT = {}

DATALOADER_DICT["levircc"] = {
    "train": dataloader_levircc_train,
    "val": dataloader_levircc_test,
    "train1": dataloader_levircc_test,
    "test": dataloader_levircc_test,
}
