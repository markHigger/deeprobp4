from typing import Any, Optional, Tuple

import json
import os
import shutil
import time

import rob599
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import optim
from torchvision import transforms
from torchvision.datasets.utils import check_integrity, download_and_extract_archive



import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.ops import box_iou
import sys, os
import json
import random
import cv2
from PIL import Image
import trimesh
import pyrender
import tqdm

_HOUGHVOTING_NUM_INLIER = 500
_HOUGHVOTING_DIRECTION_INLIER = 0.9
_LABEL2MASK_THRESHOL = 500


def hello_helper():
    print("Hello from p4_helpers.py!")

    
def verify_notebook_cells(notebook_path, expected_count):
    """
    Checks the number of cells in notebook file.

    Inputs:
    - notebook_path: File path to notebook file
    - expected_count: Integer number of cells to expect in this file
    """
    notebook_json = json.loads(open(notebook_path).read())
    num_cells = len(notebook_json['cells'])
    
    if num_cells == expected_count:
        print('Notebook has expected cell count!')
    elif num_cells > expected_count:
        print(f"Notebook has {num_cells-expected_count} unexpected cell(s)")
    elif num_cells < expected_count:
        print(f"Notebook is missing {expected_count-num_cells} cell(s)")


def loss_cross_entropy(scores, labels):
    """
    scores: a tensor [batch_size, num_classes, height, width]
    labels: a tensor [batch_size, num_classes, height, width]
    """

    cross_entropy = -torch.sum(labels * torch.log(scores + 1e-10), dim=1)
    loss = torch.div(torch.sum(cross_entropy), torch.sum(labels)+1e-10)

    return loss


def loss_Rotation(pred_R, gt_R, label, model):
    """
    pred_R: a tensor [N, 3, 3]
    gt_R: a tensor [N, 3, 3]
    label: a tensor [N, ]
    model: a tensor [N_cls, 1024, 3]
    """
    device = pred_R.device
    models_pcd = model[label - 1].to(device)
    gt_points = models_pcd @ gt_R
    pred_points = models_pcd @ pred_R
    loss = ((pred_points - gt_points) ** 2).sum(dim=2).sqrt().mean()
    return loss


def IOUselection(pred_bbxes, gt_bbxes, threshold):
    """
        pred_bbx is N_pred_bbx * 6 (batch_ids, x1, y1, x2, y2, cls)
        gt_bbx is gt_bbx * 6 (batch_ids, x1, y1, x2, y2, cls)
        threshold : threshold of IOU for selection of predicted bbx
    """
    device = pred_bbxes.device
    output_bbxes = torch.empty((0, 6)).to(device = device, dtype =torch.float)
    for pred_bbx in pred_bbxes:
        for gt_bbx in gt_bbxes:
            if pred_bbx[0] == gt_bbx[0] and pred_bbx[5] == gt_bbx[5]:
                iou = box_iou(pred_bbx[1:5].unsqueeze(dim=0), gt_bbx[1:5].unsqueeze(dim=0)).item()
                if iou > threshold:
                    output_bbxes = torch.cat((output_bbxes, pred_bbx.unsqueeze(dim=0)), dim=0)
    return output_bbxes


def HoughVoting(label, centermap, num_classes=10):
    """
    label [bs, 3, H, W]
    centermap [bs, 3*maxinstance, H, W]
    """
    batches, H, W = label.shape
    x = np.linspace(0, W - 1, W)
    y = np.linspace(0, H - 1, H)
    xv, yv = np.meshgrid(x, y)
    xy = torch.from_numpy(np.array((xv, yv))).to(device = label.device, dtype=torch.float32)
    x_index = torch.from_numpy(x).to(device = label.device, dtype=torch.int32)
    centers = torch.zeros(batches, num_classes, 2)
    depths = torch.zeros(batches, num_classes)
    for bs in range(batches):
        for cls in range(1, num_classes + 1):
            if (label[bs] == cls).sum() >= _LABEL2MASK_THRESHOL:
                pixel_location = xy[:2, label[bs] == cls]
                pixel_direction = centermap[bs, (cls-1)*3:cls*3][:2, label[bs] == cls]
                y_index = x_index.unsqueeze(dim=0) - pixel_location[0].unsqueeze(dim=1)
                y_index = torch.round(pixel_location[1].unsqueeze(dim=1) + (pixel_direction[1]/pixel_direction[0]).unsqueeze(dim=1) * y_index).to(torch.int32)
                mask = (y_index >= 0) * (y_index < H)
                count = y_index * W + x_index.unsqueeze(dim=0)
                center, inlier_num = torch.bincount(count[mask]).argmax(), torch.bincount(count[mask]).max()
                center_x, center_y = center % W, torch.div(center, W, rounding_mode='trunc')
                if inlier_num > _HOUGHVOTING_NUM_INLIER:
                    centers[bs, cls - 1, 0], centers[bs, cls - 1, 1] = center_x, center_y
                    xyplane_dis = xy - torch.tensor([center_x, center_y])[:, None, None].to(device = label.device)
                    xyplane_direction = xyplane_dis/(xyplane_dis**2).sum(dim=0).sqrt()[None, :, :]
                    predict_direction = centermap[bs, (cls-1)*3:cls*3][:2]
                    inlier_mask = ((xyplane_direction * predict_direction).sum(dim=0).abs() >= _HOUGHVOTING_DIRECTION_INLIER) * label[bs] == cls
                    depths[bs, cls - 1] = centermap[bs, (cls-1)*3:cls*3][2, inlier_mask].mean()
    return centers, depths


def infinite_loader(loader):
    """Get an infinite stream of batches from a data loader."""
    while True:
        yield from loader


def train_classifier(
    classifier,
    train_loader,
    learning_rate: float = 5e-3,
    weight_decay: float = 1e-4,
    max_iters: int = 5000,
    log_period: int = 20,
    device: str = "cpu",
):
    """
    Train the classifier. We use Adam with momentum and step decay.
    """

    classifier.to(device=device)

    # Optimizer: use SGD with momentum.
    # Use SGD with momentum:
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, classifier.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    # LR scheduler: use step decay at 70% and 90% of training iters.
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(0.6 * max_iters), int(0.9 * max_iters)]
    )

    # Keep track of training loss for plotting.
    loss_history = []

    train_loader = infinite_loader(train_loader)
    classifier.train()

    for _iter in range(max_iters):
        # Ignore first arg (image path) during training.
        images, gt_classes = next(train_loader)

        images = images.to(device)

        # Dictionary of loss scalars.
        losses = classifier.loss(images, gt_classes)

        # Ignore keys like "proposals" in RPN.
        losses = {k: v for k, v in losses.items() if "loss" in k}

        optimizer.zero_grad()
        total_loss = sum(losses.values())
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # Print losses periodically.
        if _iter % log_period == 0:
            loss_str = f"[Iter {_iter}][loss: {total_loss:.3f}]"
            for key, value in losses.items():
                loss_str += f"[{key}: {value:.3f}]"

            print(loss_str)
            loss_history.append(total_loss.item())

    # Plot training loss.
    plt.title("Training loss history")
    plt.xlabel(f"Iteration (x {log_period})")
    plt.ylabel("Loss")
    plt.plot(loss_history)
    plt.show()


def inference_classifier(
    classifier,
    test_loader
):
    """
    Evaluate the classifier. We use SGD with momentum and step decay.
    """

    classifier.to(device=device)

    train_loader = infinite_loader(train_loader)
    classifier.test()

    for _iter in range(max_iters):
        # Ignore first arg (image path) during training.
        images, gt_classes = next(train_loader)

        images = images.to(device)

        # Dictionary of loss scalars.
        losses = classifier.loss(images, gt_classes)

        # Ignore keys like "proposals" in RPN.
        losses = {k: v for k, v in losses.items() if "loss" in k}

        optimizer.zero_grad()
        total_loss = sum(losses.values())
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # Print losses periodically.
        if _iter % log_period == 0:
            loss_str = f"[Iter {_iter}][loss: {total_loss:.3f}]"
            for key, value in losses.items():
                loss_str += f"[{key}: {value:.3f}]"

            print(loss_str)
            loss_history.append(total_loss.item())

    # Plot training loss.
    plt.title("Training loss history")
    plt.xlabel(f"Iteration (x {log_period})")
    plt.ylabel("Loss")
    plt.plot(loss_history)
    plt.show()


#        Loss Functions from P1         #
def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[torch.arange(N), y]
    margins = (x - correct_class_scores[:, None] + 1.0).clamp(min=0.)
    margins[torch.arange(N), y] = 0.
    loss = margins.sum() / N
    num_pos = (margins > 0).sum(dim=1)
    dx = torch.zeros_like(x)
    dx[margins > 0] = 1.
    dx[torch.arange(N), y] -= num_pos.to(dx.dtype)
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for
      the jth class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label
      for x[i] and 0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - x.max(dim=1, keepdim=True).values
    Z = shifted_logits.exp().sum(dim=1, keepdim=True)
    log_probs = shifted_logits - Z.log()
    probs = log_probs.exp()
    N = x.shape[0]
    loss = (-1.0 / N) * log_probs[torch.arange(N), y].sum()
    dx = probs.clone()
    dx[torch.arange(N), y] -= 1
    dx /= N
    return loss, dx
