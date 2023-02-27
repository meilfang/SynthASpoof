import numpy as np
import os
import csv
import copy
from tqdm import tqdm
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from dataset import TestDataset
from utils import performances_cross_db, compute_video_score
from model.model import BaseMixModel

def main(test_csv, args):

    test_dataset = TestDataset(csv_file=test_csv, input_shape=args.input_shape)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = torch.nn.DataParallel(BaseMixModel(model_name='resnet18',  pretrained=False, num_classes=2, ms_layers=[]))
    model = model.cuda()

    model.load_state_dict(torch.load(args.model_path))
    print ('------------ test  -------------------')

    AUC, HTER = test_model(model, test_loader)
    print('AUC, HTER:', AUC, HTER)

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
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument("--input_shape", default=(224, 224), type=tuple, help="Neural Network input shape")
    parser.add_argument("--batch_size", default=512, type=int, help="train batch size")

    parser.add_argument("--test_csv", type=str, help="csv file for testing")
    parser.add_argument("--model_path", type=str, help="test model weight path")

    args = parser.parse_args()

    main(test_csv=args.test_csv, args=args)
