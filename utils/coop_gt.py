import re
import yaml
import numpy as np
from . import box_utils 
import os
import math
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

def get_pose(file):
    cur_params = load_yaml(file)
    return cur_params['lidar_pose']

def get_gt(file, reference_lidar_pose):
    tmp_object_dict = {}
    cur_params = load_yaml(file)
    tmp_object_dict.update(cur_params['vehicles'])
    output_dict = {}
    GT_RANGE = [0, -60, -3, 140, 35, 1]
    filter_range = GT_RANGE
    box_utils.project_world_objects(tmp_object_dict,
                                        output_dict,
                                        reference_lidar_pose,
                                        filter_range,
                                        'lwh')

    object_np = np.zeros((100,7))
    mask = np.zeros(100)
    object_ids = []

    for i, (object_id, object_bbx) in enumerate(output_dict.items()):
        object_np[i] = object_bbx[0, :]
        mask[i] = 1
        object_ids.append(object_id)

    return object_np[mask==1, :]