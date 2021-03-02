import numpy as np
import pr2_utils
import math

def particle_filter_update(particle_poses, particle_weights, ranges, angles, MAP, vTl, num_particles):
    # particle_poses = current vehicle particle state in world frame
    # particle_weights = particle weights
    # ranges = distances of lidar observations
    # angles = angles of lidar observations
    # MAP = occupancy grid map
    # vTl = transform matrix for lidar frame to vehicle (body) frame
    # num_particles = number of particles

    # End point of lidar observations
    lidar_x_end_points = ranges*np.cos(angles)
    lidar_y_end_points = ranges*np.sin(angles)

    # store positions of surrounding 9x9 map pixels
    x_indices_of_surrounding_nine_cells = np.arange(-4*MAP['res'], 5*MAP['res'], MAP['res'])
    y_indices_of_surrounding_nine_cells = np.arange(-4*MAP['res'], 5*MAP['res'], MAP['res'])

    # Use sigmoid function (given in slides) to find pmf og grid cells
    # Consider cells with > 0.5 pmf to be walls (as shown in lecture slides)
    map_wall_pmfs = ((np.exp(MAP['map'])/(1 + np.exp(MAP['map']))) < 0.5).astype(np.int)

    # x and y position of every map cell
    map_x_idxs = np.arange(MAP['xmin'], MAP['xmax']+MAP['res'], MAP['res'])
    map_y_idxs = np.arange(MAP['ymin'], MAP['ymax']+MAP['res'], MAP['res'])

    # initialize matrix to store cell coordinates in vehicle frame
    lidar_state = np.ones((4,np.size(lidar_x_end_points)))
    # convert to x,y cell coordinates from lidar frame to vehicle frame
    lidar_state[0,:] = lidar_x_end_points
    lidar_state[1,:] = lidar_y_end_points
    # transform from lidar frame to vehicle frame
    s_vehicle = np.dot(vTl, lidar_state)

    # initialize variables to store correlation
    correlation = np.zeros(num_particles)

  	# number of particles to iterate through
    numParticles = np.shape(particle_poses)[1]

    ################## UPDATE THE POSE OF EACH PARTICLE ##################
    for i in range(numParticles):
    	# get particle pose
        particle_pose = particle_poses[:,i]

 		# get individual x,y,theta pose elements of particle
        x_particle_pose = particle_pose[0]
        y_particle_pose = particle_pose[1]
        theta_w = particle_pose[2]
        #print('x_particle_pose:',x_particle_pose)
        #print('y_particle_pose:',y_particle_pose)
        #print('theta_w:',theta_w)

        # calculate vehicle frame to world frame transformation matrix
        wTv = np.array([[np.cos(theta_w),-np.sin(theta_w),0,x_particle_pose], [np.sin(theta_w), np.cos(theta_w), 0, y_particle_pose], [0,0,1,0], [0,0,0,1]])

        # transform coordinates from vehicle (body) frame to world frame
        particle_pose_world_frame = np.dot(wTv,s_vehicle)

        # get transformed particle pose elements
        map_x_end_points_world_frame = particle_pose_world_frame[0,:]
        map_y_end_points_world_frame = particle_pose_world_frame[1,:]
        # stack all y idexes values, not in world frame coordinates, back into original form
        lidar_y_all_idxs = np.stack((map_x_end_points_world_frame,map_y_end_points_world_frame))

        # calculate map correlation
        map_correlation = pr2_utils.mapCorrelation(map_wall_pmfs, map_x_idxs, map_y_idxs, lidar_y_all_idxs, x_indices_of_surrounding_nine_cells, y_indices_of_surrounding_nine_cells)

        # find index of highest value correlation
        correlation[i] = np.max(map_correlation)

    # update particle weights using softmax function per lecture slides
    map_x_endpoints = np.exp(correlation - np.max(correlation))
    soft_norm_corr = map_x_endpoints / map_x_endpoints.sum()
    updated_particle_weights = particle_weights*soft_norm_corr/np.sum(particle_weights*soft_norm_corr)

    return particle_poses, updated_particle_weights

