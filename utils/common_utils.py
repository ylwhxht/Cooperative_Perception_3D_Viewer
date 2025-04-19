# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
Common utilities
"""

import numpy as np
import torch
from shapely.geometry import Polygon
import json
from collections import OrderedDict
import struct
import re
import yaml
import numpy as np
import os
import math
import torch.nn.functional as F
def load_yaml(file, opt=None):
    """
    Load yaml file and return a dictionary.

    Parameters
    ----------
    file : string
        yaml file path.

    opt : argparser
         Argparser.
    Returns
    -------
    param : dict
        A dictionary that contains defined parameters.
    """
    if opt and opt.model_dir:
        file = os.path.join(opt.model_dir, 'config.yaml')

    stream = open(file, 'r')
    loader = yaml.Loader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    param = yaml.load(stream, Loader=loader)
    if "yaml_parser" in param:
        param = eval(param["yaml_parser"])(param)

    return param
def read_pcd_bin(pcd_file, type = 'binary'):
    if type == 'binary':
        with open(pcd_file, 'rb') as f:
            data = f.read()
            data_binary = data[data.find(b"DATA binary") + 12:]
            points = np.frombuffer(data_binary, dtype=np.float32).reshape(-1, 4)
            points = points.astype(np.float32)
    else:
        points = []
        f = open(pcd_file, 'r')
        data = f.readlines()
    
        f.close()
        line = data[22]
        # print line
        line = line.strip('\n')
        i = line.split(' ')
        
        for line in data[24:]:
            line = line.strip('\n')
            xyz = line.split(' ')
            x, y, z, t = [np.float32(eval(i)) for i in xyz[:4]]
            points.append([x, y, z, t])
        points = np.array(points)
    t = points[:,-1]
    for i in range(len(points)):
        binary_str = struct.pack('>f', float(t[i]))
        binary_int = int.from_bytes(binary_str, byteorder='big')
        points[i,-1] = (binary_int>>8)&0xFF > 0

    return points

def mask_ego_fov_flag(lidar, ego_yam, yam):
    """
    Args:
        lidar: lidar point clouds in ego lidar pose 
        ego_params : epo params
    Returns:
        mask of fov lidar point clouds <<in ego coords>>
    """
    xyz = lidar[:,:3]
    ego_params = load_yaml(ego_yam)
    params = load_yaml(yam)
    xyz_hom = np.concatenate(
        [xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1)
    intrinsic = np.array(ego_params['camera0']['intrinsic'])
    img_shape = [600,800,3]
    
    cpose = ego_params['camera0']['cords']
    lpose = params['lidar_pose']
    ext_matrix = x1_to_x2(cpose, lpose) @ np.array(
                [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
                dtype=np.float32)
    ext_matrix = np.linalg.inv(ext_matrix)[:3,:4]
    img_pts = (intrinsic @ ext_matrix @ xyz_hom.T).T
    depth = img_pts[:, 2]
    uv = img_pts[:, :2] / depth[:, None]

    val_flag_1 = np.logical_and(uv[:, 0] >= 0, uv[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(uv[:, 1] >= 0, uv[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(depth > 0, val_flag_merge)
    return lidar[pts_valid_flag]

def project_points_by_matrix_torch(points, transformation_matrix):
    """
    Project the points to another coordinate system based on the
    transfomration matrix.

    Parameters
    ----------
    points : torch.Tensor
        3D points, (N, 3)

    transformation_matrix : torch.Tensor
        Transformation matrix, (4, 4)

    Returns
    -------
    projected_points : torch.Tensor
        The projected points, (N, 3)
    """
    # convert to homogeneous  coordinates via padding 1 at the last dimension.
    # (N, 4)
    points_homogeneous = F.pad(points, (0, 1), mode="constant", value=1)
    # (N, 4)
    projected_points = torch.einsum("ik, jk->ij", points_homogeneous,
                                    transformation_matrix)
    return projected_points[:, :3]

def x_to_world(pose):
    """
    The transformation matrix from x-coordinate system to carla world system

    Parameters
    ----------
    pose : list
        [x, y, z, roll, yaw, pitch]

    Returns
    -------
    matrix : np.ndarray
        The transformation matrix.
    """
    x, y, z, roll, yaw, pitch = pose[:]

    # used for rotation matrix
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    matrix = np.identity(4)
    # translation matrix
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    # rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r -s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r -s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix

def x1_to_x2(x1, x2):
    """
    Transformation matrix from x1 to x2.

    Parameters
    ----------
    x1 : list
        The pose of x1 under world coordinates.
    x2 : list
        The pose of x2 under world coordinates.

    Returns
    -------
    transformation_matrix : np.ndarray
        The transformation matrix.

    """
    x1_to_world = x_to_world(x1)
    x2_to_world = x_to_world(x2)
    world_to_x2 = np.linalg.inv(x2_to_world)

    transformation_matrix = np.dot(world_to_x2, x1_to_world)
    return transformation_matrix


def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    return data

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def check_contain_nan(x):
    if isinstance(x, dict):
        return any(check_contain_nan(v) for k, v in x.items())
    if isinstance(x, list):
        return any(check_contain_nan(itm) for itm in x)
    if isinstance(x, int) or isinstance(x, float):
        return False
    if isinstance(x, np.ndarray):
        return np.any(np.isnan(x))
    return torch.any(x.isnan()).detach().cpu().item()


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), radians, angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa, sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3].float(), rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def rotate_points_along_z_2d(points, angle):
    """
    Rorate the points along z-axis.
    Parameters
    ----------
    points : torch.Tensor / np.ndarray
        (N, 2).
    angle : torch.Tensor / np.ndarray
        (N,)

    Returns
    -------
    points_rot : torch.Tensor / np.ndarray
        Rorated points with shape (N, 2)

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    # (N, 2, 2)
    rot_matrix = torch.stack((cosa, sina, -sina, cosa), dim=1).view(-1, 2,
                                                                    2).float()
    points_rot = torch.einsum("ik, ikj->ij", points.float(), rot_matrix)
    return points_rot.numpy() if is_numpy else points_rot


def remove_ego_from_objects(objects, ego_id):
    """
    Avoid adding ego vehicle to the object dictionary.

    Parameters
    ----------
    objects : dict
        The dictionary contained all objects.

    ego_id : int
        Ego id.
    """
    if ego_id in objects:
        del objects[ego_id]


def retrieve_ego_id(base_data_dict):
    """
    Retrieve the ego vehicle id from sample(origin format).

    Parameters
    ----------
    base_data_dict : dict
        Data sample in origin format.

    Returns
    -------
    ego_id : str
        The id of ego vehicle.
    """
    ego_id = None

    for cav_id, cav_content in base_data_dict.items():
        if cav_content['ego']:
            ego_id = cav_id
            break
    return ego_id


def compute_iou(box, boxes):
    """
    Compute iou between box and boxes list
    Parameters
    ----------
    box : shapely.geometry.Polygon
        Bounding box Polygon.

    boxes : list
        List of shapely.geometry.Polygon.

    Returns
    -------
    iou : np.ndarray
        Array of iou between box and boxes.

    """
    # Calculate intersection areas
    if np.any(np.array([box.union(b).area for b in boxes])==0):
        print('debug')
    iou = [box.intersection(b).area / box.union(b).area for b in boxes]

    return np.array(iou, dtype=np.float32)


def convert_format(boxes_array):
    """
    Convert boxes array to shapely.geometry.Polygon format.
    Parameters
    ----------
    boxes_array : np.ndarray
        (N, 4, 2) or (N, 8, 3).

    Returns
    -------
        list of converted shapely.geometry.Polygon object.

    """
    polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in
                boxes_array]
    return np.array(polygons)


def torch_tensor_to_numpy(torch_tensor):
    """
    Convert a torch tensor to numpy.

    Parameters
    ----------
    torch_tensor : torch.Tensor

    Returns
    -------
    A numpy array.
    """
    return torch_tensor.numpy() if not torch_tensor.is_cuda else \
        torch_tensor.cpu().detach().numpy()


def get_voxel_centers(voxel_coords,
                      downsample_times,
                      voxel_size,
                      point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        downsample_times:
        voxel_size:
        point_cloud_range:

    Returns:

    """
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers

def merge_features_to_dict(processed_feature_list, merge=None):
    """
    Merge the preprocessed features from different cavs to the same
    dictionary.

    Parameters
    ----------
    processed_feature_list : list
        A list of dictionary containing all processed features from
        different cavs.
    merge : "stack" or "cat". used for images

    Returns
    -------
    merged_feature_dict: dict
        key: feature names, value: list of features.
    """

    merged_feature_dict = OrderedDict()

    for i in range(len(processed_feature_list)):
        for feature_name, feature in processed_feature_list[i].items():
            if feature_name not in merged_feature_dict:
                merged_feature_dict[feature_name] = []
            if isinstance(feature, list):
                merged_feature_dict[feature_name] += feature
            else:
                merged_feature_dict[feature_name].append(feature) # merged_feature_dict['coords'] = [f1,f2,f3,f4]
    
    # stack them
    # it usually happens when merging cavs images -> v.shape = [N, Ncam, C, H, W]
    # cat them
    # it usually happens when merging batches cav images -> v is a list [(N1+N2+...Nn, Ncam, C, H, W))]
    if merge=='stack': 
        for feature_name, features in merged_feature_dict.items():
            merged_feature_dict[feature_name] = torch.stack(features, dim=0)
    elif merge=='cat':
        for feature_name, features in merged_feature_dict.items():
            merged_feature_dict[feature_name] = torch.cat(features, dim=0)

    return merged_feature_dict