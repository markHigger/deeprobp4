import random

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid
from torchvision.transforms import functional as tvf

from torchvision.ops import box_iou
import sys, os
import trimesh
import pyrender
import tqdm

"""
General utilities to help with implementation
"""


def reset_seed(number):
    """
    Reset random seed to the specific number

    Inputs:
    - number: A seed number to use
    """
    random.seed(number)
    np.random.seed(number)
    torch.manual_seed(number)
    return


def tensor_to_image(tensor):
    """
    Convert a torch tensor into a numpy ndarray for visualization.

    Inputs:
    - tensor: A torch tensor of shape (3, H, W) with
      elements in the range [0, 1]

    Returns:
    - ndarr: A uint8 numpy array of shape (H, W, 3)
    """
    tensor = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    ndarr = tensor.to("cpu", torch.uint8).numpy()
    return ndarr


def format_gt_RTs(RTs):
    return {idx+1: np.concatenate((RTs[idx],[[0,0,0,1]])) for idx in range(len(RTs))}

def visualize_dataset(pose_dataset, num_samples = 4, alpha = 0.5):
    """
    Make a grid-shape image to plot

    Inputs:
    - pose_dataset: instance of PROPSPoseDataset

    Outputs:
    - A grid-image that visualize num_samples
      number of image and pose label samples
    """
    plt.text(300, -40, 'RGB', ha="center")
    plt.text(950, -40, 'Pose', ha="center")
    plt.text(1600, -40, 'Depth', ha="center")
    plt.text(2250, -40, 'Segmentation', ha="center")
    plt.text(2900, -40, 'Centermaps[0]', ha="center")

    samples = []
    for sample_i in range(num_samples):
        sample_idx = random.randint(0,len(pose_dataset)-1)
        sample = pose_dataset[sample_idx]
        rgb = (sample['rgb'].transpose(1, 2, 0) * 255).astype(np.uint8)
        depth = ((np.tile(sample['depth'], (3, 1, 1)) / sample['depth'].max()) * 255).astype(np.uint8)
        segmentation = (sample['label']*np.arange(11).reshape((11,1,1))).sum(0,keepdims=True).astype(np.float64)
        segmentation /= segmentation.max()
        segmentation = (np.tile(segmentation, (3, 1, 1)) * 255).astype(np.uint8)
        ctrs = sample['centermaps'].reshape(10,3,480,640)[0]
        ctrs -= ctrs.min()
        ctrs /= ctrs.max()
        ctrs = (ctrs * 255).astype(np.uint8)
        pose_dict = format_gt_RTs(sample['RTs'])
        render = pose_dataset.visualizer.vis_oneview(
            ipt_im = rgb, 
            obj_pose_dict = pose_dict,
            alpha = alpha
            )
        samples.append(torch.tensor(rgb.transpose(2, 0, 1)))
        samples.append(torch.tensor(render.transpose(2, 0, 1)))
        samples.append(torch.tensor(depth))
        samples.append(torch.tensor(segmentation))
        samples.append(torch.tensor(ctrs))
    img = make_grid(samples, nrow=5).permute(1, 2, 0)
    return img



def chromatic_transform(image):
    """
    Add the hue, saturation and luminosity to the image.

    This is adapted from implicit-depth repository, ref: https://github.com/NVlabs/implicit_depth/blob/main/src/utils/data_augmentation.py

    Parameters
    ----------

    image: array, required, the given image.

    Returns
    -------

    The new image after augmentation in HLS space.
    """
    # Set random hue, luminosity and saturation which ranges from -0.1 to 0.1
    d_h = (np.random.rand(1) - 0.5) * 0.1 * 180
    d_l = (np.random.rand(1) - 0.5) * 0.2 * 256
    d_s = (np.random.rand(1) - 0.5) * 0.2 * 256
    # Convert the BGR to HLS
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)
    # Add the values to the image H, L, S
    new_h = (h + d_h) % 180
    new_l = np.clip(l + d_l, 0, 255)
    new_s = np.clip(s + d_s, 0, 255)
    # Convert the HLS to BGR
    new_hls = cv2.merge((new_h, new_l, new_s)).astype('uint8')
    new_image = cv2.cvtColor(new_hls, cv2.COLOR_HLS2BGR)
    return new_image



def add_noise(image, level = 0.1):
    """
    Add noise to the image.

    This is adapted from implicit-depth repository, ref: https://github.com/NVlabs/implicit_depth/blob/main/src/utils/data_augmentation.py

    Parameters
    ----------

    image: array, required, the given image;

    level: float, optional, default: 0.1, the maximum noise level.

    Returns
    -------

    The new image after augmentation of adding noises.
    """
    # random number
    r = np.random.rand(1)

    # gaussian noise
    if r < 0.9:
        row,col,ch= image.shape
        mean = 0
        noise_level = random.uniform(0, level)
        sigma = np.random.rand(1) * noise_level * 256
        gauss = sigma * np.random.randn(row,col) + mean
        gauss = np.repeat(gauss[:, :, np.newaxis], ch, axis=2)
        noisy = image + gauss
        noisy = np.clip(noisy, 0, 255)
    else:
        # motion blur
        sizes = [3, 5, 7, 9, 11, 15]
        size = sizes[int(np.random.randint(len(sizes), size=1))]
        kernel_motion_blur = np.zeros((size, size))
        if np.random.rand(1) < 0.5:
            kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
        else:
            kernel_motion_blur[:, int((size-1)/2)] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        noisy = cv2.filter2D(image, -1, kernel_motion_blur)

    return noisy.astype('uint8')


class Visualize:
    def __init__(self, object_dict, cam_intrinsic, resolution):
        '''
        object_dict is a dict store object labels, object names and object model path, 
        example:
        object_dict = {
                    1: ["beaker_1", path]
                    2: ["dropper_1", path]
                    3: ["dropper_2", path]
                }
        '''
        self.objnode = {}
        self.render = pyrender.OffscreenRenderer(resolution[0], resolution[1])
        self.scene = pyrender.Scene()
        cam = pyrender.camera.IntrinsicsCamera(cam_intrinsic[0, 0],
                                               cam_intrinsic[1, 1], 
                                               cam_intrinsic[0, 2], 
                                               cam_intrinsic[1, 2], 
                                               znear=0.05, zfar=100.0, name=None)
        self.intrinsic = cam_intrinsic
        Axis_align = np.array([[1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])
        self.nc = pyrender.Node(camera=cam, matrix=Axis_align)
        self.scene.add_node(self.nc)

        for obj_label in object_dict:
            objname = object_dict[obj_label][0]
            objpath = object_dict[obj_label][1]
            tm = trimesh.load(objpath)
            mesh = pyrender.Mesh.from_trimesh(tm, smooth = False)
            node = pyrender.Node(mesh=mesh, matrix=np.eye(4))
            node.mesh.is_visible = False
            self.objnode[obj_label] = {"name":objname, "node":node, "mesh":tm}
            self.scene.add_node(node)
        self.cmp = self.color_map(N=len(object_dict))
        self.object_dict = object_dict

    def vis_oneview(self, ipt_im, obj_pose_dict, alpha = 0.5, axis_len=30):
        '''
        Input:
            ipt_im: numpy [H, W, 3]
                input image
            obj_pose_dict:
                is a dict store object poses within input image
                example:
                poselist = {
                    15: numpy_pose 4X4,
                    37: numpy_pose 4X4,
                    39: numpy_pose 4X4,
                }
            alpha: float [0,1]
                alpha for labels' colormap on image 
            axis_len: int
                pixel lengths for draw axis
        '''
        img = ipt_im.copy()
        for obj_label in obj_pose_dict:
            if obj_label in self.object_dict:
                pose = obj_pose_dict[obj_label]
                node = self.objnode[obj_label]['node']
                node.mesh.is_visible = True
                self.scene.set_pose(node, pose=pose)
        full_depth = self.render.render(self.scene, flags = pyrender.constants.RenderFlags.DEPTH_ONLY)
        for obj_label in obj_pose_dict:
            if obj_label in self.object_dict:
                node = self.objnode[obj_label]['node']
                node.mesh.is_visible = False
        for obj_label in self.object_dict:
            node = self.objnode[obj_label]['node']
            node.mesh.is_visible = False
        for obj_label in obj_pose_dict:
            if obj_label in self.object_dict:
                node = self.objnode[obj_label]['node']
                node.mesh.is_visible = True
                depth = self.render.render(self.scene, flags = pyrender.constants.RenderFlags.DEPTH_ONLY)
                node.mesh.is_visible = False
                mask = np.logical_and(
                    (np.abs(depth - full_depth) < 1e-6), np.abs(full_depth) > 0.2
                )
                if np.sum(mask) > 0:
                    color = self.cmp[obj_label - 1]
                    img[mask, :] = alpha * img[mask, :] + (1 - alpha) * color[:]
                    obj_pose = obj_pose_dict[obj_label]
                    obj_center = self.project2d(self.intrinsic, obj_pose[:3, -1])
                    rgb_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
                    for j in range(3):
                        obj_xyz_offset_2d = self.project2d(self.intrinsic, obj_pose[:3, -1] + obj_pose[:3, j] * 0.001)
                        obj_axis_endpoint = obj_center + (obj_xyz_offset_2d - obj_center) / np.linalg.norm(obj_xyz_offset_2d - obj_center) * axis_len
                        cv2.arrowedLine(img, (int(obj_center[0]), int(obj_center[1])), (int(obj_axis_endpoint[0]), int(obj_axis_endpoint[1])), rgb_colors[j], thickness=2, tipLength=0.15)  
        return img

    def color_map(self, N=256, normalized=False):
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)
        dtype = 'float32' if normalized else 'uint8'
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3
            cmap[i] = np.array([r, g, b])
        cmap = cmap/255 if normalized else cmap
        return cmap
    
    def project2d(self, intrinsic, point3d):
        return (intrinsic @ (point3d / point3d[2]))[:2]

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))





def visualize_patches(imgs, patches):
    """
    Make a grid-shape plot of image-patches pairs

    Inputs:
    - imgs: set of [batch, 3, width, height] image data
    - patches: paired patches of imgs in [batch, patches_per_img, C, patch_size, patch_size] shape

    Outputs:
    - Matplotlib figure showing images and patches
    """
    imgs = imgs.clone().detach() 
    imgs += torch.tensor([0.47028715, 0.40359595, 0.35609495],device=imgs.device).reshape(-1,3,1,1) # Assumes props statistics
    imgs = imgs.clamp(min=0, max=1).cpu()

    patches = patches.clone().detach()
    patches += torch.tensor([0.47028715, 0.40359595, 0.35609495],device=patches.device).reshape(-1,1,3,1,1) # Assumes props statistics
    patches = patches.clamp(min=0, max=1).cpu()

    N, patches_per_img = patches.shape[:2]
    
    fig = plt.figure(figsize=(8,4), constrained_layout=True)
    subfigs = fig.subfigures(N//2, 2)
    for i in range(N//2):
        for j in range(2):
            ax = subfigs[i][j].subplots(1,2)
            
            idx = i*2+j
            img = make_grid(imgs[idx], nrow=1, normalize=False, pad_value=1.0).permute(1,2,0)
            patch_img = make_grid(patches[idx], nrow=patches_per_img//8, normalize=False, pad_value=1.0).permute(1,2,0)
            
            ax[0].imshow(img.cpu())
            ax[0].set_ylabel(f'Sample {i*2+j+1}')
            ax[0].tick_params(right=False, top= False,left=False, bottom=False)
            ax[0].set_xticklabels([])
            ax[0].set_yticklabels([])
            ax[0].set_title('Image')
            
            ax[1].imshow(patch_img)
            ax[1].axis('off')
            ax[1].set_title('Patches')

    return fig



def attention_rollout(attentions):
    """
    Implementation of Attention Rollout, based on Abnar et al. https://arxiv.org/pdf/2005.00928
    See also: https://samiraabnar.github.io/articles/2020-04/attention_flow

    Inputs:
    - attentions: list of attention maps shaped (N, H, S, S)
    
    Outputs:
    - output: tensor of shape (N, S, S)
    """
    output = torch.eye(attentions[0].size(-1)).to(attentions[0].device)
    for layer in attentions:
        layer_mean = layer.mean(dim=1)  # Average over heads
        layer_mean = layer_mean + torch.eye(layer_mean.size(-1)).to(layer_mean.device)
        layer_mean = layer_mean / layer_mean.sum(dim=-1, keepdim=True)
        output = torch.matmul(output, layer_mean)
    return output



def visualize_attention(imgs, attentions, true_cls, pred_cls):
    """
    Make a grid-shape plot of image-attention pairs

    Inputs:
    - imgs: set of [batch, 3, width, height] image data
    - attentions: list of attention maps shaped (N, H, S, S) assuming cls_token in first dimension

    Outputs:
    - Matplotlib figure showing images and patches
    """
    classes = [
                "master_chef_can",
                "cracker_box",
                "sugar_box",
                "tomato_soup_can",
                "mustard_bottle",
                "tuna_fish_can",
                "gelatin_box",
                "potted_meat_can",
                "mug",
                "large_marker"
        ]

    true_cls = true_cls.cpu()
    pred_cls = pred_cls.cpu()

    imgs = imgs.clone().detach() 
    imgs += torch.tensor([0.47028715, 0.40359595, 0.35609495],device=imgs.device).reshape(-1,3,1,1) # Assumes props statistics
    imgs = imgs.clamp(min=0, max=1).cpu()
    
    N = imgs.shape[0]
    input_h, input_w = imgs.shape[-2:]
    patch_h, patch_w = 2*[int((attentions[0].shape[-1]-1) ** 0.5)] # Assume square input image with single cls_token
    cls_attn_maps = attention_rollout(attentions)[:, 0, 1:].reshape(-1, 1, patch_h, patch_w)
    cls_attn_maps = tvf.resize(cls_attn_maps, (input_h, input_w))
    
    cls_attn_maps_norm = cls_attn_maps.clone()
    cls_attn_maps_norm -= cls_attn_maps_norm.flatten(start_dim=1).min(dim=1).values.reshape(-1,1,1,1)
    cls_attn_maps_norm /= cls_attn_maps_norm.flatten(start_dim=1).max(dim=1).values.reshape(-1,1,1,1)
    
    cls_attn_maps = cls_attn_maps.cpu().detach()
    cls_attn_maps_norm = cls_attn_maps_norm.cpu().detach()
    
    fig = plt.figure(figsize=(16,4), constrained_layout=True)
    subfigs = fig.subfigures(N//2, 3)
    for i in range(N//2):
        for j in range(2):
            ax = subfigs[i][j].subplots(1,3)
            
            idx = i*2+j

            ax[0].imshow(imgs[idx].permute(1,2,0).cpu())
            pred_string=classes[pred_cls[idx]]
            true_string=classes[true_cls[idx]]
            ax[0].set_ylabel(f"Pred:{pred_string}\nTrue:{true_string}", fontsize=8, rotation=0, labelpad=35)
            ax[0].tick_params(right=False, top= False,left=False, bottom=False)
            ax[0].set_xticklabels([])
            ax[0].set_yticklabels([])
            ax[0].set_title('Image', fontsize=12)
            
            ax[1].imshow(cls_attn_maps[idx,0])
            ax[1].axis('off')
            ax[1].set_title('Raw\nAttention', fontsize=12)

            ax[2].imshow((imgs[idx]*cls_attn_maps_norm[idx]).permute(1,2,0))
            ax[2].axis('off')
            ax[2].set_title('Image-\nAttention', fontsize=12)

    return fig
