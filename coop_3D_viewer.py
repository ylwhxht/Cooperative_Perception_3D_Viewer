from sys import implementation
from viewer.viewer import Viewer
import numpy as np
import os
import torch
from utils.common_utils import x1_to_x2, project_points_by_matrix_torch, read_pcd_bin, mask_ego_fov_flag
from utils.coop_gt import get_pose, get_gt

#######################----Parameter settings here----########################
data_path = './demo'
max_agent = 5 # max agent number
ego = 1 # which is the ego agent
#data collection from every agent
pose = [] 
lpc = [] 
rpc = [] 
boxes = [] 
permu = [1,0,2,3,4] #Deciding on the order of gradually increasing agent
FOV_mask = True
show_lidar = True
show_radar = True
#######################----Parameter settings here----########################


def load_point_cloud(path, transformation_matrix = None):
    pc = read_pcd_bin(path, type='ascii')
    if FOV_mask:
        pc = mask_ego_fov_flag(pc, ego_yaml, curr_yaml)
    if transformation_matrix is not None:
        pc[:, :3] = np.array(project_points_by_matrix_torch(torch.tensor(pc[:, :3]), transformation_matrix.to(torch.float32)))
    return pc




if __name__ == '__main__':
    vi = Viewer(bg=(245,245,245))
    ego_yaml = os.path.join(data_path, str(ego) + '.yaml')
    ego_pose = get_pose(ego_yaml)
    for i in range(max_agent):
        
        curr_yaml = os.path.join(data_path, str(i) + '.yaml')
        curr_pose = get_pose(curr_yaml)
        if i != ego:
            transformation_matrix = torch.tensor(x1_to_x2(curr_pose, ego_pose))
        else:
            transformation_matrix = None
        if show_lidar:
            lpc_path = os.path.join(data_path, str(i) + '_lidar.pcd')
            lpc.append(load_point_cloud(lpc_path, transformation_matrix))
        if show_lidar:
            rpc_path = os.path.join(data_path, str(i) + '_radar.pcd')
            rpc.append(load_point_cloud(rpc_path, transformation_matrix))
        
        boxes.append(get_gt(curr_yaml, ego_pose))

    for num_agent in range(1, max_agent + 1): 
        for i in range(num_agent):
            idx = permu[i]
            vi.add_points(lpc[idx][:, :3], scatter_filed=lpc[idx][:, 2], radius=3)
            vi.add_points(rpc[idx][:, :3], scatter_filed=rpc[idx][:, 2], radius=7)
            vi.add_3D_boxes(boxes[idx], color='red', show_corner_spheres=False, show_heading=False, line_width=4)

        for i in range(num_agent):
            idx = permu[i]
            vi.add_3D_boxes(boxes[idx], color='red', show_corner_spheres=False, show_heading=False, line_width=4)
        vi.show_3D()