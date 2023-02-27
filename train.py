import numpy as np
import os
import csv
import copy
import logging
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from dataset import TrainDataset, TestDataset, ApplyWeightedRandomSampler
from utils import  AvgrageMeter, performances_cross_db, compute_video_score
from model.model import BaseModel, BaseMixModel


def main(train_csv_1, train_csv_2, test_csv, log_file, args):
    # train_csv_1 and train_csv_2 are for synthetic and real data respectively
    # WeightedRandomSampler to balance the attack and bonafide in a mini-batch
    train_dataset_1 = TrainDataset(csv_file=train_csv_1, input_shape=args.input_shape)
    train_loader_1 = DataLoader(train_dataset_1, batch_size=args.batch_size,  sampler=ApplyWeightedRandomSampler(train_csv_1),
                                num_workers=4, pin_memory=True, drop_last=True)

    train_dataset_2 = TrainDataset(csv_file=train_csv_2, input_shape=args.input_shape)
    train_loader_2 = DataLoader(train_dataset_2, batch_size=args.batch_size, sampler=ApplyWeightedRandomSampler(train_csv_2),
                                num_workers=4, pin_memory=True, drop_last=True)

    test_dataset = TestDataset(csv_file=test_csv, input_shape=args.input_shape)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    checkpoint_save_dir = os.path.join('checkpoints/MixStyle', args.prefix)
    print('Checkpoint folder', checkpoint_save_dir)
    if not os.path.isdir(checkpoint_save_dir):
        os.makedirs(checkpoint_save_dir)

    model = torch.nn.DataParallel(BaseMixModel(model_name=args.model_name,  pretrained=False, num_classes=2, ms_layers=["layer1", "layer2"]))
    model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    cen_criterion = torch.nn.CrossEntropyLoss().cuda()

    scaler = GradScaler()

    for epoch in range(1, args.max_epoch+1):
        if os.path.isfile(os.path.join(checkpoint_save_dir, '{}.pth'.format(epoch))):
            model.load_state_dict(torch.load(os.path.join(checkpoint_save_dir, '{}.pth'.format(epoch))))
            continue
        else:
            print('-------------- train ------------------------')
            train_epoch(epoch, model, train_loader_1, train_loader_2, optimizer, cen_criterion, optimizer.param_groups[0]['lr'], scaler, checkpoint_save_dir, log_file)

            print ('------------ test  -------------------')
            AUC, HTER = test_epoch(model, test_loader)

            lr_scheduler.step()
            log_file.write(f'Test: AUC_1= {AUCs[0]:.4f}, HTER_1= {HTERS[0]:.4f}\n')
            log_file.flush()

def train_epoch(epoch, model, train_loader_1, train_loader_2, optimizer, cen_criterion, current_lr, scaler, checkpoint_save_dir, log_file):
    loss_total = AvgrageMeter()

    model.train()
    iter_data_1 = iter(train_loader_1)
    iter_data_2 = iter(train_loader_2)

    num_iter = len(train_loader_1)
    for i in range(1, num_iter):
        if i % len(train_loader_2) == 0:
            iter_data_2 = iter(train_loader_2)

        data_1= next(iter_data_1)
        input_1, label_1 = data_1["images"].cuda(), data_1["labels"].cuda()

        data_2 = next(iter_data_2)
        input_2 = data_2["images"].cuda()

        input_data = torch.cat((input_1, input_2), dim=0)

        optimizer.zero_grad()
        #### only synthetic data is used to backpropagrate loss
        output = model(input_data)
        output_1, output_2 = output.chunk(2)
        loss = cen_criterion(output_1, label_1.to(torch.int64))

        if i % 20 == 0:
            print(f'Iteration [{i}/{num_iter}]:  Loss: {loss.item():.4f}')

        loss_total.update(loss.data, input_1.shape[0])

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    torch.save(model.state_dict(), os.path.join(checkpoint_save_dir, f'{epoch}.pth'))
    log_file.write('Epoch: %d, Loss: %.4f, lr: %.6f \n' % (epoch, loss_total.avg, current_lr))
    log_file.flush()

def test_epoch(model, data_loader, video_format=True):
    model.eval()

    raw_test_scores, gt_labels = [], []
    raw_scores_dict = []
    raw_test_video_ids = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            raw, labels, img_pathes = data["images"].cuda(), data["labels"], data["img_path"]
            output = model(raw)

            raw_scores = output.softmax(dim=1)[:, 1].cpu().data.numpy()
            raw_test_scores.extend(raw_scores)
            gt_labels.extend(labels.data.numpy())

            for j in range(raw.shape[0]):
                image_name = os.path.splitext(os.path.basename(img_pathes[j]))[0]
                video_id = os.path.join(os.path.dirname(img_pathes[j]), image_name.rsplit('_', 1)[0])
                raw_test_video_ids.append(video_id)

        if video_format: # compute mean prediction score of all frames for each video
            raw_test_scores, gt_labels, _ = compute_video_score(raw_test_video_ids, raw_test_scores, gt_labels)

        raw_test_stats = [np.mean(raw_test_scores), np.std(raw_test_scores)]
        raw_test_scores = ( raw_test_scores - raw_test_stats[0]) / raw_test_stats[1]

        AUC, _, _, HTER = performances_cross_db(raw_test_scores, gt_labels)

    return AUC, HTER

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    #torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == "__main__":


    torch.cuda.empty_cache()
    set_seed(seed=777)

    import argparse
    parser = argparse.ArgumentParser(description='SynPAD Training with MixStyle')
    parser.add_argument("--prefix", default='SynFacePAD', type=str, help="log description")
    parser.add_argument("--model_name", default='resnet18', type=str, help="model backbone")
    parser.add_argument("--train_csv_1", default='synthetic data csv', type=str, help="csv contains training data")
    parser.add_argument("--train_csv_2", default='real data csv', type=str, help="csv contains training data")

    parser.add_argument("--lr", default=0.001, type=float, help="initial learning rate")
    parser.add_argument("--input_shape", default=(224, 224), type=tuple, help="Neural Network input shape")
    parser.add_argument("--max_epoch", default=80, type=int, help="maximum epochs")
    parser.add_argument("--batch_size", default=128, type=int, help="train batch size")

    args = parser.parse_args()

    test_csv = args.train_csv_2  #this real data is used to obtain real image style (without label) to synthetic and also used for testing

    logging_filename = os.path.join('logs/MixStyle',  '{}.txt'.format(args.prefix))
    if not os.path.isdir('logs/MixStyle'):
        os.makedirs('logs/MixStyle')
    log_file = open(logging_filename, 'a')
    log_file.write(f"Real PAD data: {args.train_csv_2} \n model_name: {args.model_name} pretrained, lr: {args.lr}, prefix: {args.prefix}, bs: {args.batch_size} \n")
    log_file.flush()

    main(train_csv_1=args.train_csv_1,  # synthetic data
         train_csv_2=args.train_csv_2,  # real data
         test_csv=test_csv,
         log_file=log_file,
         args=args)
