# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from var_model import VARModel
from var_eval_data import VARDataset, VARTorchDataset, VAREvaluator
from utils import TrainingMeter
import torch.distributed as dist
from vision_helpers import GroupedBatchSampler, create_aspect_ratio_groups
from param import args
import numpy as np
import scipy.stats

DataTuple = collections.namedtuple("DataTuple", 'dataset loader')


def get_data_tuple(split: int, bs:int) -> DataTuple:
    dset = VARDataset(split)
    tset = VARTorchDataset(dset)
    print(f"torch dataset for {split} with {len(tset)} is loaded")

    data_loader = DataLoader(
        tset, num_workers=args.num_workers, pin_memory=True, batch_size=bs,
        collate_fn = tset.collate_fn, shuffle=False
    )
    '''else:
        data_loader = DataLoader(
            tset, batch_size=bs,
            shuffle=shuffle, num_workers=args.num_workers,
            drop_last=drop_last, pin_memory=True
        )'''

    return DataTuple(dataset=dset, loader=data_loader)


class VAR:
    def __init__(self):
        # Model
        self.model = VARModel()

        self.model = self.model.to(args.device)

    def predict(self, eval_tuple: DataTuple):

        self.model.eval()
        dset, loader = eval_tuple
        pred_vec = []
        test_ids = []
        for i, datum_tuple in enumerate(tqdm(loader, ncols=80, desc="evaluating")):
            test_id, feats, boxes, txt = datum_tuple
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, txt)

                rank_scores_sigmoid = torch.sigmoid(logit).cpu().numpy()
                pred_vec.append(rank_scores_sigmoid)
                test_ids.extend(test_id)

        pred_vec = np.concatenate(pred_vec, axis=0)
        predictions = dict(zip(test_ids, pred_vec))
        instance_ids = list(set([x[0] for x in dset.answer_key.values()]))
        decoded_predictions = {tuple(dset.answer_key[test_instance_id]): prediction for test_instance_id, prediction in predictions.items()}
        print("decoded_length: ", len(decoded_predictions))

        # image 2 txt
        sim_mat = np.zeros((len(instance_ids), len(instance_ids)))

        for idx1, instance_id1 in enumerate(instance_ids):
            for idx2, instance_id2 in enumerate(instance_ids):
                sim_mat[idx1, idx2] = decoded_predictions[(instance_id1, instance_id2)]

        return sim_mat

    def evaluate(self, eval_tuple):
        """Evaluate all data in data_tuple."""
        sim_mat = self.predict(eval_tuple)
        im2text_ranks = np.diagonal(scipy.stats.rankdata(-sim_mat, axis=1))
        text2im_ranks = np.diagonal(scipy.stats.rankdata(-sim_mat, axis=0))
        p_at_1 = float(100*np.mean(im2text_ranks == 1.0))
        print('im2txt: {:.3f}'.format(np.mean(im2text_ranks)))
        print('txt2im: {:.3f}'.format(np.mean(text2im_ranks)))
        print('p_at_1: {:.3f}'.format(np.mean(p_at_1)))
        return np.mean(im2text_ranks), np.mean(text2im_ranks), np.mean(p_at_1)

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path, map_location="cpu")
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    args.device = torch.device("cuda")
    var = VAR()

    # Load VAR model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        print(f"fine-tuned model from {args.load} is loaded")
        var.load(args.load)

    print("Evaluation process is running")

    im2txt = []
    txt2im = []
    p_at_1 = []
    for i in tqdm(range(1), ncols=80):
        eval_tuple = get_data_tuple(
            split=i, bs=args.batch_size*4
        )

        i2t, t2i, p1 = var.evaluate(eval_tuple)
        im2txt.append(i2t)
        txt2im.append(t2i)
        p_at_1.append(p1)
    print('Final im2txt: {:.3f}'.format(sum(im2txt)/len(im2txt)))
    print('Final txt2im: {:.3f}'.format(sum(txt2im)/len(txt2im)))
    print('Final p_at_1: {:.3f}'.format(sum(p_at_1)/len(p_at_1)))

