import numpy as np
import matplotlib as plt
import pr2_utils


def update_map(pose_of_highest_weight_particle, ranges, angles, vTl, MAP):
    # pose_of_highest_weight_particle = pose of highest weight particle (also where vehicle most likely is)
    # ranges = distance of lidar observations
    # angles = angles of lidar observations
    # vTl = transformation from lidar frame to vehicle (body) frame
    # MAP = world map

    # calculate body to world transform
    x_vehicle = pose_of_highest_weight_particle[0]
    y_vehicle = pose_of_highest_weight_particle[1]
    theta_vehicle = pose_of_highest_weight_particle[2]

    # ending point of lidar observation
    lidar_best_pose_x_end_point = ranges * np.cos(angles)
    lidar_best_pose_y_end_point = ranges * np.sin(angles)

    # Convert to x,y,z coordinates - z coordinate doesn't really matter though and can be ignored
    vehicle_pose = np.ones((4, np.size(lidar_best_pose_x_end_point)))
    vehicle_pose[0, :] = lidar_best_pose_x_end_point
    vehicle_pose[1, :] = lidar_best_pose_y_end_point
    vehicle_pose[2, :] = 1.76416

    # wTv only has change in z axis so calculated using z-axis rotation matrix from lecture slides
    wTv = np.array([[np.cos(theta_vehicle), -np.sin(theta_vehicle), 0, x_vehicle],
                    [np.sin(theta_vehicle), np.cos(theta_vehicle), 0, y_vehicle], [0, 0, 1, 0], [0, 0, 0, 1]])

    # transform vehicle pose from lidar frame to world frame
    vehicle_pose = np.dot(wTv, np.dot(vTl, vehicle_pose))

    # set starting point of lidar observation in map frame as current vehicle pose in map frame
    lidar_x_start_point = np.ceil((x_vehicle - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
    lidar_y_start_point = np.ceil((y_vehicle - MAP['ymin']) / MAP['res']).astype(np.int16) - 1

    # get best lidar end point pose elements
    lidar_best_pose_x_end_point = vehicle_pose[0, :]
    lidar_best_pose_y_end_point = vehicle_pose[1, :]

    # transform end point of best pose lidar observation end points from world frame to map frame
    lidar_best_pose_x_end_point = np.ceil((lidar_best_pose_x_end_point - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
    lidar_best_pose_y_end_point = np.ceil((lidar_best_pose_y_end_point - MAP['ymin']) / MAP['res']).astype(np.int16) - 1

    # calculate cell values for each lidar scan
    num_lidar_scans = np.size(ranges)
    for i in range(num_lidar_scans):

        # use Bresenham's line rasterization algorithm to determine cells lidar scans passed through
        passed_points = pr2_utils.bresenham2D(lidar_x_start_point, lidar_y_start_point, lidar_best_pose_x_end_point[i],
                                              lidar_best_pose_y_end_point[i])

        # store x,y coordinates of cells passed through individually
        x_passed_points = passed_points[0, :].astype(np.int16)
        y_passed_points = passed_points[1, :].astype(np.int16)

        # filter only passed-through cells which lie within map bounds
        idx_in_bound_lidar_end_points = np.logical_and(
            np.logical_and(np.logical_and((x_passed_points > 1), (y_passed_points > 1)),
                           (x_passed_points < MAP['sizex'])), (y_passed_points < MAP['sizey']))

        # update Map cell log-odds values for which a lidar scan passed through
        MAP['map'][
            x_passed_points[idx_in_bound_lidar_end_points], y_passed_points[idx_in_bound_lidar_end_points]] += np.log(
            1 / 4)

        # update all Map cell log-odds values additionally to prevent log-odds values from getting too large so that
        # if cell was empty, it's log-odds value goes back to 0 and if it is full, it doubles
        if ((lidar_best_pose_x_end_point[i] > 1) and (lidar_best_pose_x_end_point[i] < MAP['sizex']) and (
                lidar_best_pose_y_end_point[i] > 1) and (lidar_best_pose_y_end_point[i] < MAP['sizey'])):
            MAP['map'][lidar_best_pose_x_end_point[i], lidar_best_pose_y_end_point[i]] += 2 * np.log(4)

    # Limit MAP range to prevent over-confidence
    MAP['map'] = np.clip(MAP['map'], 10 * np.log(1 / 4), 10 * np.log(4))

    return MAP