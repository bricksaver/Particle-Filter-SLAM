#################### IMPORT PACKAGES ####################

from pr2_utils import read_data_from_csv
from map_update import update_map
from particle_filter_prediction import particle_filter_prediction
from particle_filter_update import particle_filter_update
from resample_particles import resample_particles
from add_texture import add_texture
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import add_texture
import os

#################### LOAD PARAMETERS AND DATA ####################

# fog - gives angle in radians
fog_rotation_mtx = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 3x3
fog_translation_mtx = np.array([-0.335, -0.035, 0.78])  # 3x1
fog_timestamps, fog_data = read_data_from_csv('data/sensor_data/fog.csv')  # radians #1160508x1 #1160508x3
fog_delta_roll = fog_data[:, 0]  # 1160508x1
fog_delta_pitch = fog_data[:, 1]  # 1160508x1
fog_delta_yaw = fog_data[:, 2]  # 1160508x1

# encoder - gives ticks given timestamp difference
encoder_resolution = 4096
encoder_left_wheel_diameter = 0.623479
encoder_right_wheel_diameter = 0.622806
encoder_wheel_base = 1.52439
encoder_timestamps, encoder_data = read_data_from_csv('data/sensor_data/encoder.csv')  # 116048x1 #116048x2
encoder_left_count = encoder_data[:, 0]  # 116048x1
encoder_right_count = encoder_data[:, 1]  # 116048x1

# lidar - gives observation scans
lidar_timestamps, lidar_data = read_data_from_csv('data/sensor_data/lidar.csv')  # 115865x286
lidar_rotation_mtx = np.array(
    [[0.00130201, 0.79097, 0.605167], [0.999999, -0.000418027, -0.00160026], [-0.00102038, 0.605169, -0.796097]])
lidar_translation_mtx = np.array([0.8349, -0.0126869, 1.76416])

# stereo - gives timestamps of rgb stereo images
path = '/Users/benja/Documents/Grad School/ECE 276A/Projects/ECE276A_PR2/code/data/stereo_images/stereo_left'
stereo_timestamps = os.listdir(path)
# stereo camera to vehicle frame transform
vTc = np.array([[-0.00680499, -0.0153215, 0.99985, 1.64239],
                [-0.999977, 0.000334627, -0.00680066, 0.247401],
                [-0.000230383, -0.999883, -0.0153234, 1.58411],
                [0, 0, 0, 1]])

# transform from lidar frame to vehicle (body) frame [Rt,pt;0,0,0,1]
vTl = np.array([[0.00130201, 0.79097, 0.605167, 0.8349],
                [0.999999, -0.000418027, -0.00160026, -0.0126869],
                [-0.00102038, 0.605169, -0.796097, 1.76416],
                [0, 0, 0, 1]])
# print(vTl)

#################### SYNC TIMESTAMPS #################### DECIDED TO DO BELOW PER ITERATION INSTEAD ####################
#################### Comment out when changing code ####################
#################### or takes forever to run ####################
# Match Timestamps - gets closest encoder timestamps and gets encoder data closest to fog timestamp
# output: relevant encoder data with size of fog (fog is dataset with most samples)
'''
num_fog_obs = len(fog_timestamps)
encoder_left_count_matched = np.zeros(num_fog_obs)
for i in range(num_fog_obs):
    diff = abs(encoder_timestamps - fog_timestamps[i])
    index1 = np.argmin(diff)
    encoder_left_count_matched[i] = encoder_left_count[index1]

# Match Timestamps - gets closest lidar timestamps and gets lidar data closest to fog timestamp
# output: relevant lidar data with size of fog (fog is dataset with most samples)

num_fog_obs = len(fog_timestamps)
#print(lidar_timestamps)
lidar_data_matched = np.zeros((num_fog_obs,286))
for i in range(num_fog_obs):
    diff = abs(lidar_timestamps - fog_timestamps[i])
    index1 = np.argmin(diff)
    lidar_data_matched[i][:] = lidar_data[index1][:]
'''
# Initialize Occupancy-Grid Map (unit: cells)
# use sample code from pr2_utils
MAP = {}
MAP['res'] = 1  # meters
MAP['xmin'] = -100  # meters
MAP['ymin'] = -1200
MAP['xmax'] = 1300
MAP['ymax'] = 200
MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))  # 1001
MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))  # 1001
MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.float32)  # DATA TYPE: char or int8
texturemap = np.zeros((MAP['sizex'], MAP['sizey'], 3), dtype=np.uint8)  # texture map

# At this point, Map is completely black w/ all 0 values

# initialize all particle filter particles with pose [x,y,theta] = [0,0,0] in world frame
num_particles = 10  # Change value as needed
particle_poses = np.zeros((3, num_particles))  # particle_poses = (x, y, theta)' x num_particles
particle_weights = np.array([1 / num_particles] * num_particles).reshape(1,
                                                                         num_particles)  # all particles initialized with same weight # all 0.25
# print(particle_weights.shape)

# Initialize Map
# use sample code from pr2_utils
angles = np.linspace(-5, 185, 286) / 180 * np.pi
ranges = lidar_data[0, :]
validIdxs = np.logical_and((ranges < 70), (ranges > 2))
ranges = ranges[validIdxs]
angles = angles[validIdxs]
idx_highest_weight_particle = np.argmax(particle_weights)  # idx_highest_weight_particle  = 0 for initial
# pose_of_highest_weight_particle = particle_poses[:,idx_highest_weight_particle ]
# Add first lidar scan to map
pose_of_highest_weight_particle = [0, 0, 0]  # initial best particle pose
MAP = update_map(pose_of_highest_weight_particle, ranges, angles, vTl, MAP)

'''
# Plot Occupancy Grid Map
fig1 = plt.figure()
plt.imshow(MAP['map'], cmap="gray")
plt.title("Occupancy grid map")
plt.show(block=True)
'''

# Get Number of FOG Measurements
fog_num_measurements = len(fog_timestamps)
# print('lidar timestamps shape:',lidar_timestamps.shape) #115865,1

# Parameter whether to do texture mapping or not - texture mapping currently not finished adding
do_texture_mapping = 1  # 0=no, 1=yes

################### RUN MULTIPLE PREDICT AND UPDATE ITERATIONS ###################
# Predict Particle Motion and add noise using Encoder and FOG
# Encoder -> Differential-Drive Model -> velocity
# FOG -> Differential-Drive Model -> angular velocity

# initialize variable to store all past vehicle trajectory cell indices
trajectory = np.array([[0], [0]])

# tau = 0.001  # seconds
num_fog_obs = len(fog_timestamps)

# initialize time synced encoder and lidar variables
encoder_left_count_matched = np.zeros(num_fog_obs)
lidar_data_matched = np.zeros((num_fog_obs, 286))

for i in range(fog_num_measurements):

    # Match encoder timestamps to fog timestamps
    diff = abs(encoder_timestamps - fog_timestamps[i])
    index1 = np.argmin(diff)
    encoder_left_count_matched[i] = encoder_left_count[index1]
    # Match lidar timestamps to fog timestamps
    diff = abs(lidar_timestamps - fog_timestamps[i])
    index1 = np.argmin(diff)
    lidar_data_matched[i][:] = lidar_data[index1][:]

    if (i % 500 == 0):
        #################### DISPLAY OCCUPANCY GRID MAP ####################
        #################### INCLUDING VEHICLE TRAJECTORY AND LIDAR SCANS ####################
        print('iteration:', i)

        # calculate pmf of each occupancy-grid map cell using sigmoid eq. from slides
        # and determine which ones are walls and maps
        map_result = ((np.exp(MAP['map']) / (1 + np.exp(MAP['map']))) < 0.13).astype(np.int)
        wall_result = ((np.exp(MAP['map']) / (1 + np.exp(MAP['map']))) > 0.87).astype(np.int)

        # convert lidar observation end points to grid units in world frame
        lidar_x_end_points = trajectory[0, :]
        lidar_y_end_points = trajectory[1, :]
        # print('lidar_x_end_points:',lidar_x_end_points)
        # print('lidar_y_end_points:',lidar_y_end_points)

        # Convert lidar observation end points from world frame (meters) to map frame grid units (cells)
        # use sample code from pr2_utils
        lidar_x_end_points = np.ceil((lidar_x_end_points - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
        lidar_y_end_points = np.ceil((lidar_y_end_points - MAP['ymin']) / MAP['res']).astype(np.int16) - 1
        # print('lidar_x_end_points:', lidar_x_end_points)
        # print('lidar_y_end_points:', lidar_y_end_points)

        # Obtain only lidar observation end points within MAP bounds
        # use sample code from pr2_utils
        idx_in_bound_lidar_end_points = np.logical_and(
            np.logical_and(np.logical_and((lidar_x_end_points > 1), (lidar_y_end_points > 1)),
                           (lidar_x_end_points < MAP['sizex'])),
            (lidar_y_end_points < MAP['sizey']))
        # print(idx_in_bound_lidar_end_points)

        # Update Occupancy-Grid Map values at cells with in-bound lidar endponts
        map_result[
            lidar_x_end_points[idx_in_bound_lidar_end_points], lidar_y_end_points[idx_in_bound_lidar_end_points]] = 2
        wall_result[
            lidar_x_end_points[idx_in_bound_lidar_end_points], lidar_y_end_points[idx_in_bound_lidar_end_points]] = 2

        # plot occupancy grid map
        plt.imshow(map_result, cmap="gray")
        plt.show()
        plt.pause(0.001)

    ################### PREDICTION ###################

    # calculate vehicle velocity and angular velocity from FOG gyroscope data
    # 10 in denominator accounts for how encoder data is called every iteration, but there is supposed to
    # be 10x fog data happening for every 1 encoder data
    '''
    tau = (encoder_timestamps[i] - encoder_timestamps[i - 1]) / 10 ** 9
    if i != 0:
        left_linear_velocity = (math.pi * encoder_left_wheel_diameter * (
                    encoder_left_count[i] - encoder_left_count[i - 1])) / (encoder_resolution * tau)
    else:
        left_linear_velocity = (math.pi * encoder_left_wheel_diameter * encoder_left_count[i]) / (
                    encoder_resolution * tau)
    if i != 0:
        right_linear_velocity = (math.pi * encoder_right_wheel_diameter * (
                    encoder_right_count[i] - encoder_right_count[i - 1])) / (encoder_resolution * tau)
    else:
        right_linear_velocity = (math.pi * encoder_right_wheel_diameter * encoder_right_count[i]) / (
                    encoder_resolution * tau)
    linear_velocity = (left_linear_velocity + right_linear_velocity) / 2
    angular_velocity = fog_delta_yaw[i]
    '''
    # Choose matched or unmatched encoder left count to use
    encoder_left_count_in_use = encoder_left_count_matched[i]
    encoder_left_count_in_use_prev = encoder_left_count_matched[i - 1]
    # encoder_left_count_in_use = encoder_left_count[i]
    # encoder_left_count_in_use_prev = encoder_left_count[i-1]
    # Calculate v_tau
    if i != 0:
        v_tau = (math.pi * encoder_left_wheel_diameter * (
                    encoder_left_count_in_use - encoder_left_count_in_use_prev)) / (encoder_resolution)
    else:
        v_tau = (math.pi * encoder_left_wheel_diameter * (encoder_left_count_in_use)) / (encoder_resolution)
    # print('v_tau:', v_tau)
    # angular_velocity = fog_delta_yaw[i]/tau

    delta_theta = fog_delta_yaw[i]
    # delta_theta = fog_delta_yaw[i]
    # print('delta_theta:',delta_theta)
    # predict vehicle particle pose
    particle_poses = particle_filter_prediction(particle_poses, delta_theta, v_tau)
    # print('Predicted particle_poses:',particle_poses)

    # Do update step every 10 since there's new lidar data about every 10 fog timestamps so updating
    # more often than that is pointless. Can also decrease how often updates are done to decrease run-ti
    if (i % 10 == 0):
        ################### UPDATE ####################
        # Finds particle with highest weight and updates vehicle trajectory and MAP with it
        # note: map adds lidars scans from highest weight particle pose

        # initialize lidar angles
        angles = np.linspace(-5, 185, 286) / 180 * np.pi  # radians

        # Choose matched or unmatched lidar scan data to use
        ranges = lidar_data_matched[i, :]
        # ranges = lidar_data[i, :]

        # filter out only range indexes within valid lidar range
        validIdxs = np.logical_and((ranges < 70), (ranges > 2))
        # store filtered lidar scan indices
        ranges = ranges[validIdxs]
        angles = angles[validIdxs]

        # update particle weights
        particle_poses, particle_weights = particle_filter_update(particle_poses, particle_weights, ranges, angles, MAP,
                                                                  vTl, num_particles)
        # print("this is after update: ", particle_poses[1, :])
        # print('updated particle_poses:',particle_poses)
        # print('updated_weight:',weight)

        # find index of particle with highest weight (the particle most likely representing the correct vehicle pose)
        idx_highest_weight_particle = np.argmax(particle_weights)
        # print('idx_highest_weight_particle :',idx_highest_weight_particle )
        # store pose of highest weight particle
        pose_of_highest_weight_particle = particle_poses[:, idx_highest_weight_particle]
        # print('pose_of_highest_weight_particle best:',pose_of_highest_weight_particle)
        # update vehicle trajectory with pose of highest weight particle
        trajectory = np.hstack((trajectory, pose_of_highest_weight_particle[0:2].reshape(2, 1)))
        # print('trajectory:',trajectory)
        # update map with trajectory of highest weight particle
        MAP = update_map(pose_of_highest_weight_particle, ranges, angles, vTl, MAP)

        # texture mapping
        '''
        if (do_texture_mapping):
            # find depth and rgb images with clsest timestamps to current fog time
            stereo_timestamps = str(stereo_timestamps).replace('.png','')
            diff = abs(stereo_timestamps - fog_timestamps[i])
            index3 = np.argmin(diff)

            img_disp = Image.open('data/stereo_images/disparity_images/%s' % stereo_timestamps[i])
            disp = np.array(img_disp, np.uint16)
            img_rgb = Image.open('data/stereo_images/stereo_left/%s' % stereo_timestamps[i])
            rgb = np.array(img_rgb, np.uint16)

            texturemap = add_texture(img_rgb, img_disp, vTc, pose_of_highest_weight_particle, texturemap, invK, MAP)
        '''

        # resample particles if number of particles gets low
        num_particles_eff = 1 / np.dot(particle_weights.reshape(1, num_particles),
                                       particle_weights.reshape(num_particles, 1))
        if num_particles_eff < 5:
            particle_poses, particle_weights = resample_particles(particle_poses, particle_weights, num_particles)
