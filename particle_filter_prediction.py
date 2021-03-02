import numpy as np
import pr2_utils

def particle_filter_prediction(particle_poses, delta_theta, v_tau):
    # particle_poses = current vehicle particle state in world frame
    # pose_change = lidar odometry data containing change in x, y, theta
    # num_particles = num of particles

    # Current Vehicle Pose
    x_w = particle_poses[0,:]
    y_w = particle_poses[1,:]
    theta_w = particle_poses[2,:]

    # Predicted Pose Change in Vehicle Pose using Differential-Drive Model
    #print('v_tau:',v_tau)
    delta_x = v_tau*np.cos(theta_w + delta_theta)
    #print('delta_x:', delta_x)
    delta_y = v_tau*np.sin(theta_w + delta_theta)
    #print('delta_y:', delta_y)
    #print('theta_w:', theta_w)
    delta_theta = delta_theta
    #delta_theta = tau*angular_velocity #angular velocity is: w = delta_theta / delta_t

    # Add predicted pose change to current pose for every particle and ...
    # ... Add gaussian noise to predicted pose change (supposed to help somehow)
    num_particles = np.shape(particle_poses)[1] # number of particles
    mu = 0
    x_w = x_w + delta_x + (np.array([np.random.normal(mu, abs(np.max(delta_x))/10)]))[0]
    #print('x_w:', x_w)
    y_w = y_w + delta_y + (np.array([np.random.normal(mu, abs(np.max(delta_y))/10)]))[0]
    theta_w = theta_w + delta_theta + (np.array([np.random.normal(mu, abs(np.max(delta_theta))/10)]))[0]

    # Convert New Vehicle Pose to World Frame
    particle_poses_new = np.ones((3,num_particles))
    particle_poses_new[0,:] = x_w
    particle_poses_new[1,:] = y_w
    particle_poses_new[2,:] = theta_w
    return particle_poses_new