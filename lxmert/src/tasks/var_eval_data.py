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
    A VAR evaluation instance data example in json file:
        {
            'image': 
            {
                'url': 'http://s3-us-west-2.amazonaws.com/ai2-rowanz/vcr1images/lsmdc_0014_Ist_das_Leben_nicht_schoen/0014_Ist_das_Leben_nicht_schoen_00.52.45.070-00.52.46.702@0.jpg',
                'width': 1920,
                'height': 1080
            },
            'region': [{'height': 1010, 'width': 864, 'left': 419, 'top': 71}, {'height': 612, 'width': 409, 'left': 1302, 'top': 468}],
            'inference': 'it is currently raining',
            'test_id': '000023912abd16f5d52662966729deda',
            'extra_info': {'split': 0, 'task': 'retrieval'}
        }
    Answer-key data in json file: value[0] == value[1] 에서 right score
        {
            '000023912abd16f5d52662966729deda': ['c344c9b7dbee3d5d5dfcfc66c245b737', 'c344c9b7dbee3d5d5dfcfc66c245b737']
            ...
        }
    """
    def __init__(self, split):
        # Loading datasets
        #################################### instances.json으로 수정
        self.data = json.load(open(f'/hdd/user4/workspace/Zeroshot/sherlock/leaderboard_eval/val_retrieval/val_retrieval_{split}_instances.json'))
        print("Load %d data from split %s" % (len(self.data), str(split)))
        
        self.answer_key = json.load(open(f'/hdd/user4/workspace/Zeroshot/sherlock/leaderboard_eval/val_retrieval/val_retrieval_{split}_answer_key.json'))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['test_id']: datum
            for datum in self.data
        }

    def __len__(self):
        return len(self.data)
    
class VARBufferLoader():
    def __init__(self):
        self.key2data = {}

    def load_data(self, number):
        path ="/hdd/user4/workspace/ExtractFeatures/val_obj36.tsv"
        key = "%s_%d" % (path, number)
        if key not in self.key2data:
            self.key2data[key] = load_obj_tsv(
                path,
                topk=number
            )
        return self.key2data[key]

var_buffer_loader = VARBufferLoader()

class VARTorchDataset(Dataset):
    def __init__(self, dataset: VARDataset):
        super().__init__()
        self.raw_dataset = dataset

        img_data = []
        img_data.extend(var_buffer_loader.load_data(-1))

        self.answer_key = self.raw_dataset.answer_key

        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        self.vcr_dir='/hdd/user16/HT/data/vcr/vcr1images/vcr1images/'
        self.vg_dir='/hdd/user4/data/vg_raw_images/'
        
        self.data = self.raw_dataset.data
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]
        test_id = datum['test_id']
        img_id = os.path.splitext(self.url2filepath(datum['image']['url']))[0].split('/')[-1]

        #####
        inputs = []
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

        txt = datum['inference']

        inputs.append((test_id, torch.tensor(feats), torch.tensor(boxes), txt))

        return inputs  # inputs = [(test_id, feats, boxes, txt)]
    
    def url2filepath(self, url):
        if 'VG_' in url:
            return self.vg_dir + '/'.join(url.split('/')[-2:])
        else:
            # http://s3-us-west-2.amazonaws.com/ai2-rowanz/vcr1images/lsmdc_3023_DISTRICT_9/3023_DISTRICT_9_01.21.02.808-01.21.16.722@5.jpg
            if 'vcr1images' in self.vcr_dir:
                return self.vcr_dir + '/'.join(url.split('/')[-2:])
            else:
                return self.vcr_dir + '/'.join(url.split('/')[-3:])


    def collate_fn(self, inputs):
        (test_id, feats, boxes, txt,) = map(list, unzip(concat(i for i in inputs)))
        # len(feats), len(boxes), len(txt) = batch_size

        feats = torch.stack(feats, dim=0)
        boxes = torch.stack(boxes, dim=0)
        test_id = np.array(test_id)
  
        return test_id, feats, boxes, txt
    
class VAREvaluator:
    def __init__(self, dataset: VARDataset):
        self.dataset = dataset
        

    def evaluate(self, txtid2ans: dict):
        score = 0.
        for txtid, ans in txtid2ans.items():
            if ans == 0:
                score += 1
        return score / len(txtid2ans)
