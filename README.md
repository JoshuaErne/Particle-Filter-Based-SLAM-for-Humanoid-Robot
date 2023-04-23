# Particle Filter-Based SLAM for Humanoid Robot
## Overview
The Simultaneous Localization and Mapping (SLAM) problem is a common challenge faced by robotics systems. It involves the robot accurately determining its own location and creating a map of its environment at the same time. One popular algorithm for solving the SLAM problem is the particle filter (PF), which is often used in conjunction with the Kalman filter (KF) and the Extended Kalman Filter (EKF). In this project, I have implemented SLAM using a particle filter on data collected from a humanoid robot named THOR, which was developed by researchers at the University of Pennsylvania and the University of California, Los Angeles. 

### Hardware Setup of THOR

The humanoid THOR has a [Hokuyo LiDAR sensor](https://hokuyo-usa.com/products/lidar-obstacle-detection) on its head. This LiDAR is a planar LiDAR sensor and returns 1080 readings at each instant, each reading being the distance of some physical object along a ray that shoots off at an angle between (-135, 135) degrees with discretization of 0.25 degrees in an horizontal plane.

![image](https://user-images.githubusercontent.com/38180831/205529594-cda1e25e-a384-4bf2-b0ca-0a23c0719735.png)

Here I use the position and orientation of the head of the robot to calculate the orientation of the LiDAR in the body frame. The second kind of observations I used pertain to the location of the robot. I  directly used the (x,y,θ) pose of the robot in the world coordinates (θ denotes yaw). These poses were created presumably on the robot by running a filter on the IMU data (such estimates are called odometry estimates), and these poses will not be extremely accurate.

You can read more about the hardware in this paper - [THOR-OP humanoid robot for DARPA Robotics Challenge Trials 2013](https://ieeexplore.ieee.org/document/7057369)

### Coordinate frames
The body frame is at the top of the head (X axis pointing forwards, Y axis pointing left and Z axis pointing upwards), the top of the head is at a height of 1.263m from the ground. The transformation from the body frame to the LiDAR frame depends upon the angle of the head (pitch) and the angle of the neck (yaw) and the height of the LiDAR above the head (which is 0.15m). The world coordinate frame where we want to build the map has its origin on the ground plane, i.e., the origin of the body frame is at a height of 1.263m with respect to the world frame at location (x,y,θ).

## Running the code
1. You can choose to run the different parts of the SLAM algorithm (dynamic step and observation step) either separately or together. To do this, pass a mode argument, either 'dynamics', 'observation', or 'slam', in the main function of `main.py`.

2. Run the `main.py` file and set the datasets you want to use by passing the idx argument corresponding to the desired dataset.

## Results
#### The odometry and dynamics plots for dynamics step:
##### Map 1
![image](/Images/traj1.png)
![image](/Images/map1.png)

##### Map 2
![image](/Images/traj2.png)
![image](/Images/map2.png)

##### Map 3
![image](/Images/traj3.png)
![image](/Images/map3.png)

##### Map 4
![image](/Images/traj4.png)
![image](/Images/map4.png)
