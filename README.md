# Particle-Filter-SLAM
Particle filter prediction and update steps to localize differential-drive model vehicle

HOW_TO_RUN_CODE
- You shouldn't need to change anything. It should run out of the box.
- If any questions, please email me at bmc011@ucsd.edu


DESCRIPTION OF FILES
main.py
- initializes map
- syncs encoder and lidar data to FOG data per timestamps
- runs prediction() from particle_filter.py
- runs update() from particle_filter.py
- runs update_map() from map_update.py

particle_filter_prediction.py
- particle_filter_prediction() - function for particle filter prediction step

particle_filter_update.py
- particle_filter_update() - function for particle filter update step

map_update.py
- update_map() - function for updating map

resample_particles.py
- resample_particles() - function for resampling particle filter particle weights

add_texture.py (INCOMPLETE)
- add_texture() - function to add color to map lidar scans

pr2_utils.py
- tic()
- toc()
- compute_stereo() - modified to calculate and save disparity images from given stereo images
- read_data_from_csv() - used to import lidar, encoder, fog data in main.py
- mapcorrelation() - used to find correlation between 
- bresenham2D() - used to find cells passed through by lidar scans
- test_bresenham2D() 
- test_mapCorrelation() - borrowed a lot of code from here for map_update's update_map() function
- show_lidar()

DESCRIPTION OF DATA/PARAM FILES (NOT INCLUDED IN SUBMISSION)

DATA
encoder.csv
- [timestamp, left count, right count]
fog.csv
- [timestamp, delta roll, delta pitch, delta yaw]
lidar.csv
- [timestamp, 286 lidar values]
stereo_images folder
- 

PARAMETERS
EncoderParameters.txt
  - Encoder resolution
  - Encoder left wheel diameter
  - Encoder right wheel diameter
  - Encoder wheel base
Lidar_param.txt
  - FOV
  - Start angle
  - End angle
  - Angular resolution
  - Max range
stereo_param.txt
  - Stereo camera baseline
Vehicle2FOG.txt
  - Rotation matrix
  - Translation vector
Vehicle2Lidar.txt
  - Rotation matrix
  - Translation vector
