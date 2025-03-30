"""
Implements the PoseCNN network architecture in PyTorch.
"""
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
import torchvision.models as models
from torchvision.ops import RoIPool

import numpy as np
import random
import statistics
import time
from typing import Dict, List, Callable, Optional

from rob599 import quaternion_to_matrix
from rob599.p4_helpers import HoughVoting, _LABEL2MASK_THRESHOL, loss_cross_entropy, loss_Rotation, IOUselection


def hello_pose_cnn():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from pose_cnn.py!")


class FeatureExtraction(nn.Module):
    """
    Feature Embedding Module for PoseCNN. Using pretrained VGG16 network as backbone.
    """    
    def __init__(self, pretrained_model):
        super(FeatureExtraction, self).__init__()
        embedding_layers = list(pretrained_model.features)[:30]
        ## Embedding Module from begining till the first output feature map
        self.embedding1 = nn.Sequential(*embedding_layers[:23])
        ## Embedding Module from the first output feature map till the second output feature map
        self.embedding2 = nn.Sequential(*embedding_layers[23:])

        for i in [0, 2, 5, 7, 10, 12, 14]:
            self.embedding1[i].weight.requires_grad = False
            self.embedding1[i].bias.requires_grad = False
    
    def forward(self, datadict):
        """
        feature1: [bs, 512, H/8, W/8]
        feature2: [bs, 512, H/16, W/16]
        """ 
        feature1 = self.embedding1(datadict['rgb'])
        feature2 = self.embedding2(feature1)
        return feature1, feature2

class SegmentationBranch(nn.Module):
    """
    Instance Segmentation Module for PoseCNN. 
    """    
    def __init__(self, num_classes = 10, hidden_layer_dim = 64):
        super(SegmentationBranch, self).__init__()

        ######################################################################
        # TODO: Initialize instance segmentation branch layers for PoseCNN.  #
        #                                                                    #
        # 1) Both feature1 and feature2 should be passed through a 1x1 conv  #
        # + ReLU layer (seperate layer for each feature).                    #
        #                                                                    #
        # 2) Next, intermediate features from feature1 should be upsampled   #
        # to match spatial resolution of features2.                          #
        #                                                                    #
        # 3) Intermediate features should be added, element-wise.            #
        #                                                                    #
        # 4) Final probability map generated by 1x1 conv+ReLU -> softmax     #
        #                                                                    #
        # It is recommended that you initialize each convolution kernel with #
        # the kaiming_normal initializer and each bias vector to zeros.      #
        #                                                                    #
        # Note: num_classes passed as input does not include the background  #
        # our desired probability map should be over classses and background #
        # Input channels will be 512, hidden_layer_dim gives channels for    #
        # each embedding layer in this network.                              #
        ######################################################################
        # Replace "pass" statement with your code
        self.num_classes = num_classes
        self.conv_feat1 = nn.Conv2d(512, hidden_layer_dim, kernel_size=1)
        kaiming_normal_(self.conv_feat1.weight, mode='fan_in', nonlinearity='relu')
        self.conv_feat1.bias.data.fill_(0)

        self.conv_feat2 = nn.Conv2d(512, hidden_layer_dim, kernel_size=1)
        kaiming_normal_(self.conv_feat2.weight, mode='fan_in', nonlinearity='relu')
        self.conv_feat2.bias.data.fill_(0)
    
        self.conv_out = nn.Conv2d(hidden_layer_dim, num_classes + 1, kernel_size=1)
        kaiming_normal_(self.conv_out.weight, mode='fan_in', nonlinearity='relu')
        self.conv_out.bias.data.fill_(0)

        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################


    def forward(self, feature1, feature2):
        """
        Args:
            feature1: Features from feature extraction backbone (B, 512, h, w)
            feature2: Features from feature extraction backbone (B, 512, h//2, w//2)
        Returns:
            probability: Segmentation map of probability for each class at each pixel.
                probability size: (B,num_classes+1,H,W)
            segmentation: Segmentation map of class id's with highest prob at each pixel.
                segmentation size: (B,H,W)
            bbx: Bounding boxs detected from the segmentation. Can be extracted 
                from the predicted segmentation map using self.label2bbx(segmentation).
                bbx size: (N,6) with (batch_ids, x1, y1, x2, y2, cls)
        """
        probability = None
        segmentation = None
        bbx = None
        
        ######################################################################
        # TODO: Implement forward pass of instance segmentation branch.      #
        ######################################################################
        # Replace "pass" statement with your code
        feat1 = nn.functional.relu(self.conv_feat1(feature1))
        feat2 = nn.functional.relu(self.conv_feat2(feature2))
        feat2_upsampled = nn.functional.interpolate(feat2, size=feat1.shape[2:])

        combo_feat = feat1 + feat2_upsampled
        combo_upsample = nn.functional.interpolate(combo_feat, scale_factor = 8.0)
        out = nn.functional.relu(self.conv_out(combo_upsample))

        probability = nn.functional.softmax(out, dim=1)
        segmentation = torch.argmax(probability, dim=1)
        bbx = self.label2bbx(segmentation)
        
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        return probability, segmentation, bbx
    
    def label2bbx(self, label):
        bbx = []
        bs, H, W = label.shape
        device = label.device
        label_repeat = label.view(bs, 1, H, W).repeat(1, self.num_classes, 1, 1).to(device)
        label_target = torch.linspace(0, self.num_classes - 1, steps = self.num_classes).view(1, -1, 1, 1).repeat(bs, 1, H, W).to(device)
        mask = (label_repeat == label_target)
        for batch_id in range(mask.shape[0]):
            for cls_id in range(mask.shape[1]):
                if cls_id != 0: 
                    # cls_id == 0 is the background
                    y, x = torch.where(mask[batch_id, cls_id] != 0)
                    if y.numel() >= _LABEL2MASK_THRESHOL:
                        bbx.append([batch_id, torch.min(x).item(), torch.min(y).item(), 
                                    torch.max(x).item(), torch.max(y).item(), cls_id])
        bbx = torch.tensor(bbx).to(device)
        return bbx
        
        
class TranslationBranch(nn.Module):
    """
    3D Translation Estimation Module for PoseCNN. 
    """    
    def __init__(self, num_classes = 10, hidden_layer_dim = 128):
        super(TranslationBranch, self).__init__()
        
        ######################################################################
        # TODO: Initialize layers of translation branch for PoseCNN.         #
        # It is recommended that you initialize each convolution kernel with #
        # the kaiming_normal initializer and each bias vector to zeros.      #
        ######################################################################
        # Replace "pass" statement with your code
        self.num_classes = num_classes
        self.conv_feat1 = nn.Conv2d(512, hidden_layer_dim, kernel_size=1)
        kaiming_normal_(self.conv_feat1.weight, mode='fan_in', nonlinearity='relu')
        self.conv_feat1.bias.data.fill_(0)

        self.conv_feat2 = nn.Conv2d(512, hidden_layer_dim, kernel_size=1)
        kaiming_normal_(self.conv_feat2.weight, mode='fan_in', nonlinearity='relu')
        self.conv_feat2.bias.data.fill_(0)
    
        self.conv_out = nn.Conv2d(hidden_layer_dim, 3*num_classes, kernel_size=1)
        kaiming_normal_(self.conv_out.weight, mode='fan_in', nonlinearity='relu')
        self.conv_out.bias.data.fill_(0)

        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    def forward(self, feature1, feature2):
        """
        Args:
            feature1: Features from feature extraction backbone (B, 512, h, w)
            feature2: Features from feature extraction backbone (B, 512, h//2, w//2)
        Returns:
            translation: Map of object centroid predictions.
                translation size: (N,3*num_classes,H,W)
        """
        translation = None
        ######################################################################
        # TODO: Implement forward pass of translation branch.                #
        ######################################################################
        # Replace "pass" statement with your code
        feat1 = nn.functional.relu(self.conv_feat1(feature1))
        feat2 = nn.functional.relu(self.conv_feat2(feature2))
        feat2_upsampled = nn.functional.interpolate(feat2, size=feat1.shape[2:])

        combo_feat = feat1 + feat2_upsampled
        combo_upsample = nn.functional.interpolate(combo_feat, scale_factor = 8.0)
        translation = nn.functional.relu(self.conv_out(combo_upsample))
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################
        return translation

class RotationBranch(nn.Module):
    """
    3D Rotation Regression Module for PoseCNN. 
    """    
    def __init__(self, feature_dim = 512, roi_shape = 7, hidden_dim = 4096, num_classes = 10):
        super(RotationBranch, self).__init__()

        ######################################################################
        # TODO: Initialize layers of rotation branch for PoseCNN.            #
        # It is recommended that you initialize each convolution kernel with #
        # the kaiming_normal initializer and each bias vector to zeros.      #
        ######################################################################
        # Replace "pass" statement with your code
        self.num_classes = num_classes
        self.roi_shape = roi_shape
        self.RoI_feat1 = RoIPool(roi_shape, 1)

        self.RoI_feat2 = RoIPool(roi_shape, 1)
        
        self.fc_layer1 = nn.Linear(feature_dim*roi_shape*roi_shape, hidden_dim)
        kaiming_normal_(self.fc_layer1.weight, mode='fan_in', nonlinearity='relu')
        self.fc_layer1.bias.data.fill_(0)

        self.fc_layer2 = nn.Linear(hidden_dim, hidden_dim)
        kaiming_normal_(self.fc_layer2.weight, mode='fan_in', nonlinearity='relu')
        self.fc_layer2.bias.data.fill_(0)
        
        self.fc_layer3 = nn.Linear(hidden_dim, 4*num_classes)
        kaiming_normal_(self.fc_layer3.weight, mode='fan_in', nonlinearity='relu')
        self.fc_layer3.bias.data.fill_(0)

        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################


    def forward(self, feature1, feature2, bbx):
        """
        Args:
            feature1: Features from feature extraction backbone (B, 512, h, w)
            feature2: Features from feature extraction backbone (B, 512, h//2, w//2)
            bbx: Bounding boxes of regions of interst (N, 5) with (batch_ids, x1, y1, x2, y2)
        Returns:
            quaternion: Regressed components of a quaternion for each class at each ROI.
                quaternion size: (N,4*num_classes)
        """
        quaternion = None

        ######################################################################
        # TODO: Implement forward pass of rotation branch.                   #
        ######################################################################
        # Replace "pass" statement with your code
        bbx = bbx.to(feature1.dtype)
        feat1 = self.RoI_feat1(feature1, bbx)
        feat2 = self.RoI_feat2(feature2, bbx)

        feat_combo = feat1 + feat2
        feat_flat = feat_combo.view(feat_combo.shape[0], -1)
        
        layer1_out = nn.functional.leaky_relu(self.fc_layer1(feat_flat))
        layer2_out = nn.functional.leaky_relu(self.fc_layer2(layer1_out))
        quaternion = nn.functional.leaky_relu(self.fc_layer3(layer2_out))
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        return quaternion

class PoseCNN(nn.Module):
    """
    PoseCNN
    """
    def __init__(self, pretrained_backbone, models_pcd, cam_intrinsic):
        super(PoseCNN, self).__init__()

        self.iou_threshold = 0.7
        self.models_pcd = models_pcd
        self.cam_intrinsic = cam_intrinsic

        ######################################################################
        # TODO: Initialize layers and components of PoseCNN.                 #
        #                                                                    #
        # Create an instance of FeatureExtraction, SegmentationBranch,       #
        # TranslationBranch, and RotationBranch for use in PoseCNN           #
        ######################################################################
        # Replace "pass" statement with your code
        self.FeatExtract = FeatureExtraction(pretrained_backbone)
        self.SegmentBranch = SegmentationBranch()
        self.TransBranch = TranslationBranch()
        self.RotationBranch = RotationBranch()
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################


    def forward(self, input_dict):
        """
        input_dict = {
            'rgb',
            'depth',
            'objs_id',
            'mask',
            'bbx',
            'RTs'
        }
        """

        if self.training:
            loss_dict = {
                "loss_segmentation": 0,
                "loss_centermap": 0,
                "loss_R": 0
            }

            gt_bbx = self.getGTbbx(input_dict)

            ######################################################################
            # TODO: Implement PoseCNN's forward pass for training.               #
            #                                                                    #
            # Model should extract features, segment the objects, identify roi   #
            # object bounding boxes, and predict rotation and translations for   #
            # each roi box.                                                      #
            #                                                                    #
            # The training loss for semantic segmentation should be stored in    #
            # loss_dict["loss_segmentation"] and calculated using the            #
            # loss_cross_entropy(.) function.                                    #
            #                                                                    #
            # The training loss for translation should be stored in              #
            # loss_dict["loss_centermap"] using the L1loss function.             #
            #                                                                    #
            # The training loss for rotation should be stored in                 #
            # loss_dict["loss_R"] using the given loss_Rotation function.        #
            ######################################################################
            # Important: the rotation loss should be calculated only for regions
            # of interest that match with a ground truth object instance.
            # Note that the helper function, IOUselection, may be used for 
            # identifying the predicted regions of interest with acceptable IOU 
            # with the ground truth bounding boxes.
            # If no ROIs result from the selection, don't compute the loss_R

            #for key, val in input_dict.items():
            #    print(f"{key}: shape is {val.shape}")
            
            # Replace "pass" statement with your code
            feat1, feat2 = self.FeatExtract(input_dict)
            probability, segmentation, bbx = self.SegmentBranch(feat1, feat2)
            loss_dict["loss_segmentation"] = loss_cross_entropy(probability, input_dict['label'])
            translation = self.TransBranch(feat1, feat2)
            testTranslation = self.estimateTrans(translation, bbx, segmentation)
            l1_loss = torch.nn.L1Loss()
            loss_dict["loss_centermap"] = l1_loss(testTranslation, input_dict["centermaps"])
            bbx_selected = IOUselection(bbx, gt_bbx, self.iou_threshold)
            if(bbx_selected.shape[0] > 0):
                loss_dict["loss_R"] = 0.0
                rotations = self.RotationBranch(feat1, feat2, bbx_selected[:,0:5])
                predRot,label_pred = self.estimateRotation(rotations, bbx_selected)
                #print(predRot.shape)
                gtRot = self.gtRotation(bbx_selected, input_dict)
                loss_dict['loss_R'] = loss_Rotation(predRot, gtRot, label_pred, self.models_pcd)

            ######################################################################
            #                            END OF YOUR CODE                        #
            ######################################################################
            
            return loss_dict
        else:
            output_dict = None
            segmentation = None

            with torch.no_grad():
                ######################################################################
                # TODO: Implement PoseCNN's forward pass for inference.              #
                ######################################################################
                # Replace "pass" statement with your code
                feat1, feat2 = self.FeatExtract(input_dict)
                probability, segmentation, bbx = self.SegmentBranch(feat1, feat2)
                translation = self.TransBranch(feat1, feat2)
                #pred_T = self.estimateTrans(translation, bbx, segmentation)
                rotations = self.RotationBranch(feat1, feat2, bbx[:,0:5])
                predRot, label_pred = self.estimateRotation(rotations, bbx)
                # print(torch.max(segmentation))
                # print(translation)
                pred_centers, pred_depth = HoughVoting(segmentation, translation)
                output_dict = self.generate_pose(predRot, pred_centers, pred_depth, bbx)

                ######################################################################
                #                            END OF YOUR CODE                        #
                ######################################################################

            return output_dict, segmentation
    
    def estimateTrans(self, translation_map, filter_bbx, pred_label):
        """
        translation_map: a tensor [batch_size, num_classes * 3, height, width]
        filter_bbx: N_filter_bbx * 6 (batch_ids, x1, y1, x2, y2, cls)
        label: a tensor [batch_size, num_classes, height, width]
        """
        N_filter_bbx = filter_bbx.shape[0]
        pred_Ts = torch.zeros(N_filter_bbx, 3)
        for idx, bbx in enumerate(filter_bbx):
            batch_id = int(bbx[0].item())
            cls = int(bbx[5].item())
            trans_map = translation_map[batch_id, (cls-1) * 3 : cls * 3, :]
            label = (pred_label[batch_id] == cls).detach()
            pred_T = trans_map[:, label].mean(dim=1)
            pred_Ts[idx] = pred_T
        return pred_Ts

    def gtTrans(self, filter_bbx, input_dict):
        N_filter_bbx = filter_bbx.shape[0]
        gt_Ts = torch.zeros(N_filter_bbx, 3)
        for idx, bbx in enumerate(filter_bbx):
            batch_id = int(bbx[0].item())
            cls = int(bbx[5].item())
            gt_Ts[idx] = input_dict['RTs'][batch_id][cls - 1][:3, [3]].T
        return gt_Ts 

    def getGTbbx(self, input_dict):
        """
            bbx is N*6 (batch_ids, x1, y1, x2, y2, cls)
        """
        gt_bbx = []
        objs_id = input_dict['objs_id']
        device = objs_id.device
        ## [x_min, y_min, width, height]
        bbxes = input_dict['bbx']
        for batch_id in range(bbxes.shape[0]):
            for idx, obj_id in enumerate(objs_id[batch_id]):
                if obj_id.item() != 0:
                    # the obj appears in this image
                    bbx = bbxes[batch_id][idx]
                    gt_bbx.append([batch_id, bbx[0].item(), bbx[1].item(),
                                  bbx[0].item() + bbx[2].item(), bbx[1].item() + bbx[3].item(), obj_id.item()])
        return torch.tensor(gt_bbx).to(device=device, dtype=torch.int16)
        
    def estimateRotation(self, quaternion_map, filter_bbx):
        """
        quaternion_map: a tensor [batch_size, num_classes * 3, height, width]
        filter_bbx: N_filter_bbx * 6 (batch_ids, x1, y1, x2, y2, cls)
        """
        N_filter_bbx = filter_bbx.shape[0]
        pred_Rs = torch.zeros(N_filter_bbx, 3, 3)
        label = []
        for idx, bbx in enumerate(filter_bbx):
            batch_id = int(bbx[0].item())
            cls = int(bbx[5].item())
            quaternion = quaternion_map[idx, (cls-1) * 4 : cls * 4]
            quaternion = nn.functional.normalize(quaternion, dim=0)
            pred_Rs[idx] = quaternion_to_matrix(quaternion)
            label.append(cls)
        label = torch.tensor(label)
        return pred_Rs, label

    def gtRotation(self, filter_bbx, input_dict):
        N_filter_bbx = filter_bbx.shape[0]
        gt_Rs = torch.zeros(N_filter_bbx, 3, 3)
        for idx, bbx in enumerate(filter_bbx):
            batch_id = int(bbx[0].item())
            cls = int(bbx[5].item())
            gt_Rs[idx] = input_dict['RTs'][batch_id][cls - 1][:3, :3]
        return gt_Rs 

    def generate_pose(self, pred_Rs, pred_centers, pred_depths, bbxs):
        """
        pred_Rs: a tensor [pred_bbx_size, 3, 3]
        pred_centers: [batch_size, num_classes, 2]
        pred_depths: a tensor [batch_size, num_classes]
        bbx: a tensor [pred_bbx_size, 6]
        """        
        output_dict = {}
        for idx, bbx in enumerate(bbxs):
            bs, _, _, _, _, obj_id = bbx
            R = pred_Rs[idx].numpy()
            center = pred_centers[bs, obj_id - 1].numpy()
            depth = pred_depths[bs, obj_id - 1].numpy()
            if (center**2).sum().item() != 0:
                T = np.linalg.inv(self.cam_intrinsic) @ np.array([center[0], center[1], 1]) * depth
                T = T[:, np.newaxis]
                if bs.item() not in output_dict:
                    output_dict[bs.item()] = {}
                output_dict[bs.item()][obj_id.item()] = np.vstack((np.hstack((R, T)), np.array([[0, 0, 0, 1]])))
        return output_dict


def eval(model, dataloader, device, alpha = 0.35):
    import cv2
    model.eval()

    sample_idx = random.randint(0,len(dataloader.dataset)-1)
    ## image version vis
    rgb = torch.tensor(dataloader.dataset[sample_idx]['rgb'][None, :]).to(device)
    inputdict = {'rgb': rgb}
    pose_dict, label = model(inputdict)
    poselist = []
    rgb =  (rgb[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return dataloader.dataset.visualizer.vis_oneview(
        ipt_im = rgb, 
        obj_pose_dict = pose_dict[0],
        alpha = alpha
        )

