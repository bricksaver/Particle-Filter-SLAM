import numpy as np
import matplotlib as plt
import pr2_utils

def add_texture(rgbImg, disparityImg, vTc, pose_of_highest_weight_particle, texture_map, inverseK, MAP):
    # rgbImg = rgb image
    # depthImg = depth image
    # T_depth_to_body = transformation from depth to body coordinates
    # pose_of_highest_weight_particle = best robot pose
    # T_rgb_to_body = transformation from rgb to body coordinates
    # texture_map = current texture map

    # UPDATE TEXTURE MAP
    x_vehicle = pose_of_highest_weight_particle[0]
    y_vehicle = pose_of_highest_weight_particle[1]
    theta_vehicle = pose_of_highest_weight_particle[2]
    wTv = np.array([[np.cos(theta_vehicle), -np.sin(theta_vehicle), 0, x_vehicle],
                                [np.sin(theta_vehicle), np.cos(theta_vehicle), 0, y_vehicle],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])

    return texture_map