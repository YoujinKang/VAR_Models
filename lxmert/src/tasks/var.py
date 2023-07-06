# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from var_model import VARModel
from var_data import VARDataset, VARTorchDataset, VAREvaluator
from utils import TrainingMeter
from param import args

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False, exhaustive = False, neg_sample_size = 1) -> DataTuple:
    dset = VARDataset(splits)
    tset = VARTorchDataset(dset, valid=splits, neg_sample_size=neg_sample_size)
    print(f"torch dataset for {splits} with {len(tset)} is loaded")
    evaluator = VAREvaluator(dset)

    train_sampler = torch.utils.data.RandomSampler(tset)
    if not shuffle:
        train_sampler = torch.utils.data.SequentialSampler(tset)

    train_batch_sampler = torch.utils.data.BatchSampler(
    train_sampler, bs, drop_last=True)

    data_loader = DataLoader(
        tset,
        batch_sampler=train_batch_sampler, num_workers=args.num_workers, pin_memory=True,
        collate_fn = tset.collate_fn
    )
    '''else:
        data_loader = DataLoader(
            tset, batch_size=bs,
            shuffle=shuffle, num_workers=args.num_workers,
            drop_last=drop_last, pin_memory=True
        )'''

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class VAR:
    def __init__(self):
        # Datasets
        self.train_tuple = get_data_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if "val" in args.valid:
            self.valid_tuple = get_data_tuple(
                args.valid, bs=args.batch_size*4,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None
        
        # Model
        self.model = VARModel()

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
            print(f"Model is loaded from {args.load_lxmert}")
        
        # print(summary(self.model))
        # print(self.model)

        self.model = self.model.to(args.device)
        # Loss and Optimizer
       
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(list(self.model.parameters()), args.lr)
        
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple, margin=0.2):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader), ncols=80, desc="training")) if args.tqdm else (lambda x: x)

        best_valid = 0.
        train_meter = TrainingMeter()
        self.optim.zero_grad()
        for epoch in range(args.epochs):
            quesid2ans = {}
            n_hard_ex = 0  #####
            for i, (txt_id, feats, boxes, txt, sample_size) in iter_wrapper(enumerate(loader)):
                self.model.train()
                #self.optim.zero_grad()

                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, txt)
                
                #####
                rank_scores_sigmoid = torch.sigmoid(logit)
                scores = rank_scores_sigmoid.contiguous().view(-1, sample_size)
                pos = scores[:, :1]
                neg = scores[:, 1:]
                loss = torch.clamp(margin + neg - pos, 0)  
                loss = loss.mean()
                #####

                # loss = loss * logit.size(1) 
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
               
                loss.backward()
                
                if ((i + 1) % args.gradient_accumulation_steps == 0) or (args.gradient_accumulation_steps == 1):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                    
                    self.optim.step()
                    # self.scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    #self.optim.step()
                    self.optim.zero_grad()

                train_meter.update(
                    {'loss': loss.detach().mean().item() * args.gradient_accumulation_steps / logit.size(1)}
                )
                
                if (i + 1) % 1000 == 0:
                    if self.valid_tuple is not None:  # Do Validation
                        valid_score = self.evaluate(eval_tuple)
                        if valid_score > best_valid:
                            best_valid = valid_score
                            self.save("BEST")

                        log_validation = "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                            "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)
                        print(log_validation)

            log_str = "\nEpoch %d: \n" % (epoch)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")
                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)
            
            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        txtid2ans = {}
        for i, datum_tuple in enumerate(tqdm(loader, ncols=80, desc="evaluating")):
            txt_id, feats, boxes, txt, sample_size = datum_tuple
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, txt)

                rank_scores_sigmoid = torch.sigmoid(logit)
                scores = rank_scores_sigmoid.contiguous().view(-1, sample_size)
                label = torch.argmax(scores, dim=-1)

                for tid, l in zip(txt_id, label.cpu().numpy()):
                    txtid2ans[tid] = l
        if dump is not None:
            evaluator.dump_result(txtid2ans, dump)
        return txtid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        txtid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(txtid2ans)


    def save(self, name):
        save_path = os.path.join(self.output + "/%s.pth" % name)
        torch.save(self.model.state_dict(), save_path)
        print(f"Save model in {save_path}")

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("/%s.pth" % path, map_location="cpu")
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    args.device = torch.device("cuda")
    var = VAR()

    # Load VAR model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        print(f"fine-tuned model from {args.load} is loaded")
        var.load(args.load)

    # Test or Train
    if args.test is not None:
        print("Evaluation process is running")
        result = var.evaluate(
            get_data_tuple('val', bs=args.batch_size,
                            shuffle=False, drop_last=False, neg_sample_size = 1000),
            dump=os.path.join(args.output, 'minival_predict.json')
        )
        print(result)
    else:
        print('Splits in Train data:', var.train_tuple.dataset.splits)
        if var.valid_tuple is not None:
            print('Splits in Valid data:', var.valid_tuple.dataset.splits)
            #print("Valid Oracle: %0.2f" % (var.oracle_score(var.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        var.train(var.train_tuple, var.valid_tuple)