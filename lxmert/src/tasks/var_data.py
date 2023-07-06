# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import load_obj_tsv
import os
import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from collections import OrderedDict
import torch
from tqdm import tqdm
from vision_helpers import to_image_list
import json
import random  # for negative sample
from collections import defaultdict
from toolz.sandbox import unzip
from cytoolz import concat

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

def dict_str_key_to_int(target_dict):
    return {int(k) if k.isnumeric() else k :v for k,v in target_dict.items()}

def _has_overlap(la, lb):
    if len(la) < len(lb):
        la, lb = lb, la
    s = set(la)
    return any(b in s for b in lb)

def sample_negative(sample_pool, ground_truths, num_sample):
    """ random and retry """
    outputs = ground_truths[:1]
    while _has_overlap(outputs, ground_truths):
        outputs = random.sample(sample_pool, num_sample)
    return outputs


class VARDataset:
    """
    A VAR data example in json file:
        {
            'inputs': 
            {
                'image': 
                {
                    'url': 'https://cs.stanford.edu/people/rak248/VG_100K_2/2378438.jpg', 
                    'width': 500, 'height': 375
                }, 
                'bboxes': [{'height': 168, 'width': 89, 'left': 329, 'top': 119}], 
                'clue': 'fruit cut in half', 'confidence': 2.0, 'obs_idx': 1
            }, 
            'targets': 
            {
                'inference': 'people going to eat it'
            }, 
            'instance_id': 'c344c9b7dbee3d5d5dfcfc66c245b737', 
            'split_idx': 0
        }
    """
    def __init__(self, splits: str):
        self.splits = splits  # tain, val

        # Loading datasets
        self.data = json.load(open("/hdd/user4/data/sherlock/sherlock_%s.json" % self.splits))
        print("Load %d data from split(s) %s." % (len(self.data), self.splits))

        self.box_dic = json.load(open('/hdd/user4/data/sherlock/image_url2auto_bboxes.json'))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['instance_id']: datum
            for datum in self.data
        }

    def __len__(self):
        return len(self.data)
    
class VARBufferLoader():
    def __init__(self):
        self.key2data = {}

    def load_data(self, name, number):
        if 'val' in name:
            path ="/hdd/user4/workspace/ExtractFeatures/val_obj36.tsv"
        else:
            path ="/hdd/user4/workspace/ExtractFeatures/train_obj36.tsv"
        key = "%s_%d" % (path, number)
        if key not in self.key2data:
            self.key2data[key] = load_obj_tsv(
                path,
                topk=number
            )
        return self.key2data[key]

var_buffer_loader = VARBufferLoader()

class VARTorchDataset(Dataset):
    def __init__(self, dataset: VARDataset, neg_sample_size=1, valid=""):
        super().__init__()
        self.raw_dataset = dataset
        self.neg_sample_size = neg_sample_size  #####

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = -1

        img_data = []
        if 'val' in dataset.splits:
            img_data.extend(var_buffer_loader.load_data('val', -1))
        else:
            img_data.extend(var_buffer_loader.load_data('train', topk))

        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        ###
        self.vcr_dir='/hdd/user16/HT/data/vcr/vcr1images/vcr1images/'
        self.vg_dir='/hdd/user4/data/vg_raw_images/'
        # self.input_raw_images = args.input_raw_images
        # self.vqa_style_transform = args.vqa_style_transform
        # self.image_size_min = args.image_size_min
        # self.image_size_max = args.image_size_max

        self.data = self.raw_dataset.data
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]
        img_id = os.path.splitext(self.url2filepath(datum['inputs']['image']['url']))[0].split('/')[-1]

        #####
        id_pairs = [(item, img_id)]
        if self.neg_sample_size > 0:
            neg_sample_img_ids = sample_negative(list(self.imgid2img.keys()), [img_id], self.neg_sample_size)
            neg_sample_txt_ids = sample_negative(list(range(len(self.data))), [item], self.neg_sample_size)
            id_pairs.extend([(item, neg_img_id) for neg_img_id in neg_sample_img_ids] + [(neg_txt_id, img_id) for neg_txt_id in neg_sample_txt_ids])

        inputs = self._collect_input(id_pairs)
        assert len(inputs) == (1 + 2*self.neg_sample_size)
        return inputs  # inputs = [(txt_id, feats, boxes, txt), (txt_id, feats, boxes, txt), (txt_id, feats, boxes, txt)]

    def url2filepath(self, url):
        if 'VG_' in url:
            return self.vg_dir + '/'.join(url.split('/')[-2:])
        else:
            # http://s3-us-west-2.amazonaws.com/ai2-rowanz/vcr1images/lsmdc_3023_DISTRICT_9/3023_DISTRICT_9_01.21.02.808-01.21.16.722@5.jpg
            if 'vcr1images' in self.vcr_dir:
                return self.vcr_dir + '/'.join(url.split('/')[-2:])
            else:
                return self.vcr_dir + '/'.join(url.split('/')[-3:])

    #####
    def _collect_input(self, id_pairs):
        inputs = []
        for txt_id, img_id in id_pairs:
            # Get image info
            img_info = self.imgid2img[img_id]
            obj_num = img_info['num_boxes']
            boxes = img_info['boxes'].copy()
            feats = img_info['features'].copy()
            assert len(boxes) == len(feats) == obj_num

            # Normalize the boxes (to 0 ~ 1)
            img_h, img_w = img_info['img_h'], img_info['img_w']
            boxes = boxes.copy()
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h
            np.testing.assert_array_less(boxes, 1+1e-5)
            np.testing.assert_array_less(-boxes, 0+1e-5)

            example = self.data[txt_id]
            txt = example['targets']['inference']

            inputs.append((txt_id, torch.tensor(feats), torch.tensor(boxes), txt))
        return inputs
        
    def collate_fn(self, inputs):
        (txt_id, feats, boxes, txt,) = map(list, unzip(concat(i for i in inputs)))
        # len(feats), len(boxes), len(txt) = batch_size

        feats = torch.stack(feats, dim=0)
        boxes = torch.stack(boxes, dim=0)
        sample_size = len(inputs[0])
  
        return txt_id, feats, boxes, txt, sample_size
    
class VAREvaluator:
    def __init__(self, dataset: VARDataset):
        self.dataset = dataset

    def evaluate(self, txtid2ans: dict):
        score = 0.
        for txtid, ans in txtid2ans.items():
            if ans == 0:
                score += 1
        return score / len(txtid2ans)
