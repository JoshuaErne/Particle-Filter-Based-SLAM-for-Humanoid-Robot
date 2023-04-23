
from operator import not_
import os, sys, pickle, math
from re import I
from copy import deepcopy

from scipy import io
import numpy as np
import matplotlib.pyplot as plt

from load_data import load_lidar_data, load_joint_data, joint_name_to_index
from utils import *

import logging
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

from itertools import accumulate
class map_t:
    """
    This will maintain the occupancy grid and log_odds. You do not need to change anything
    in the initialization
    """
    def __init__(s, resolution=0.05):
        s.resolution = resolution
        s.xmin, s.xmax = -20, 20
        s.ymin, s.ymax = -20, 20
        s.szx = int(np.ceil((s.xmax-s.xmin)/s.resolution+1))
        s.szy = int(np.ceil((s.ymax-s.ymin)/s.resolution+1))

        # binarized map and log-odds
        s.cells = np.zeros((s.szx, s.szy), dtype=np.int8)
        s.log_odds = np.zeros(s.cells.shape, dtype=np.float64)

        # value above which we are not going to increase the log-odds
        # similarly we will not decrease log-odds of a cell below -max
        s.log_odds_max = 5e6
        # number of observations received yet for each cell
        s.num_obs_per_cell = np.zeros(s.cells.shape, dtype=np.uint64)

        # we call a cell occupied if the probability of
        # occupancy P(m_i | ... ) is >= occupied_prob_thresh
        s.occupied_prob_thresh = 0.6
        s.log_odds_thresh = np.log(s.occupied_prob_thresh/(1-s.occupied_prob_thresh))

    def grid_cell_from_xy(s, x, y):
        """
        x and y are 1-dimensional arrays, compute the cell indices in the map corresponding
        to these (x,y) locations. You should return an array of shape 2 x len(x). Be
        careful to handle instances when x/y go outside the map bounds, you can use
        np.clip to handle these situations.
        """
        #### TODO: XXXXXXXXXXX
        #raise NotImplementedError
        
        x_lower_thresh = np.floor((x - s.xmin) / s.resolution)
        y_lower_thresh = np.floor((y - s.ymin) / s.resolution)

        #return an array of shape 2 x len(x)
        new_xmax       = 2 * s.xmax
        x_upper_thresh = np.ceil(new_xmax / s.resolution)
        new_ymax       = 2 * s.ymax
        y_upper_thresh = np.ceil(new_ymax / s.resolution)
        
        x_new = np.clip(x_lower_thresh, 0, x_upper_thresh)
        y_new = np.clip(y_lower_thresh, 0, y_upper_thresh)
        
        cell_indices = np.vstack((x_new, y_new))
    
        return cell_indices.astype(int)

class slam_t:
    """
    s is the same as self. In Python it does not really matter
    what we call self, s is shorter. As a general comment, (I believe)
    you will have fewer bugs while writing scientific code if you
    use the same/similar variable names as those in the mathematical equations.
    """
    def __init__(s, resolution=0.05, Q=1e-3*np.eye(3),
                 resampling_threshold=0.3):
        s.init_sensor_model()

        # dynamics noise for the state (x,y,yaw)
        #s.Q = 1e-8*np.eye(3)
        s.Q = Q

        # we resample particles if the effective number of particles
        # falls below s.resampling_threshold*num_particles
        s.resampling_threshold = resampling_threshold

        # initialize the map
        s.map = map_t(resolution)

    def read_data(s, src_dir, idx=0, split='train'):
        """
        src_dir: location of the "data" directory
        """
        logging.info('> Reading data')
        s.idx = idx
        s.lidar = load_lidar_data(os.path.join(src_dir,
                                               'data/%s/%s_lidar%d'%(split,split,idx)))
        s.joint = load_joint_data(os.path.join(src_dir,
                                               'data/%s/%s_joint%d'%(split,split,idx)))

        # finds the closets idx in the joint timestamp array such that the timestamp
        # at that idx is t
        s.find_joint_t_idx_from_lidar = lambda t: np.argmin(np.abs(s.joint['t']-t))

    def init_sensor_model(s):
        # lidar height from the ground in meters
        s.head_height = 0.93 + 0.33
        s.lidar_height = 0.15

        # dmin is the minimum reading of the LiDAR, dmax is the maximum reading
        s.lidar_dmin = 1e-3
        s.lidar_dmax = 30
        s.lidar_angular_resolution = 0.25
        # these are the angles of the rays of the Hokuyo
        s.lidar_angles = np.arange(-135,135+s.lidar_angular_resolution,
                                   s.lidar_angular_resolution)*np.pi/180.0

        # sensor model lidar_log_odds_occ is the value by which we would increase the log_odds
        # for occupied cells. lidar_log_odds_free is the value by which we should decrease the
        # log_odds for free cells (which are all cells that are not occupied)
        s.lidar_log_odds_occ = np.log(9)
        s.lidar_log_odds_free = np.log(1/9.)

    def init_particles(s, n=100, p=None, w=None, t0=0):
        """
        n: number of particles
        p: xy yaw locations of particles (3xn array)
        w: weights (array of length n)
        """
        s.n = n
        s.p = deepcopy(p) if p is not None else np.zeros((3,s.n), dtype=np.float64)
        s.w = deepcopy(w) if w is not None else np.ones(n)/float(s.n)

    @staticmethod
    def stratified_resampling(p, w):
        """
        resampling step of the particle filter, takes p = 3 x n array of
        particles with w = 1 x n array of weights and returns new particle
        locations (number of particles n remains the same) and their weights
        """
        #### TODO: XXXXXXXXXXX
        resampled_weights = np.ones(p.shape[1]) / p.shape[1] 
        resampled_particles = np.zeros(p.shape)

        spaced_pts = []
        for i in range(p.shape[1]):
            spaced_pts.append(i)
        spaced_pts = np.array(spaced_pts)

        randomness   = np.random.uniform(0,1,p.shape[1])
        rand        = (spaced_pts + randomness)/p.shape[1]
        i = 0
        
        accumulate_w = np.array(list(accumulate(w)))   # Creating a cumulative sum of weights
        accumulate_w = accumulate_w / accumulate_w[-1] #Normalizing

        for j in range(p.shape[1]):
            while accumulate_w[i] < rand[j]:
                i = i + 1
            resampled_particles[:, j] = p[:, i]

        return resampled_particles, resampled_weights

    @staticmethod
    def log_sum_exp(w):
        return w.max() + np.log(np.exp(w-w.max()).sum())

    def rays2world(s, t, p, d, head_angle=0, neck_angle=0, angles=None):
        """
        p is the pose of the particle (x,y,yaw)
        angles = angle of each ray in the body frame (this will usually
        be simply s.lidar_angles for the different lidar rays)
        d = is an array that stores the distance of along the ray of the lidar, for each ray (the length of d has to be equal to that of angles, this is s.lidar[t]['scan'])
        Return an array 2 x num_rays which are the (x,y) locations of the end point of each ray
        in world coordinates
        """
        #### TODO: XXXXXXXXXXX
        #raise NotImplementedError
        rpy = s.lidar[t]['rpy']
        # make sure each distance >= dmin and <= dmax, otherwise something is wrong in reading
        # the data
        for i in range(len(d)):
            if d[i]<s.lidar_dmin:
                d[i] = s.lidar_dmin
            elif d[i]>s.lidar_dmax:
                d[i] = s.lidar_dmax

        locations = np.zeros((3, len(d)))

        xs  = d * np.cos(angles)
        ys  = d * np.sin(angles)
        zs  = np.zeros_like(xs)
        ones = np.ones_like(xs)

        pts = np.vstack((xs,ys,zs,ones))

        # 1. from lidar distances to points in the LiDAR frame
        # 2. from LiDAR frame to the body frame
        lidar_topofhead      = euler_to_se3(0, 0, 0,                    np.array([0, 0, s.lidar_height]))
        head_body            = euler_to_se3(0, head_angle, neck_angle,  np.array([0, 0, 0.33]))
        body_world           = euler_to_se3(rpy[0],rpy[1], rpy[2],      np.array([p[0],p[1],s.head_height-0.33])
)                    
        
        # 3. from body frame to world frame
        lidar_world         = body_world @ head_body @ lidar_topofhead

        for i in range(len(xs)):
            transformed_pt = lidar_world @ pts[:,i]
            locations[:,i] = transformed_pt[:3].reshape(-1)

        true_pts   = locations[:, locations[2] > 0]

        return true_pts[:2]

    def get_control(s, t):
        """
        Use the pose at time t and t-1 to calculate what control the robot could have taken
        at time t-1 at state (x,y,th)_{t-1} to come to the current state (x,y,th)_t. We will
        assume that this is the same control that the robot will take in the function dynamics_step
        below at time t, to go to time t-1. need to use the smart_minus_2d function to get the difference of the two poses and we will simply set this to be the control (delta x, delta y, delta theta)
        """

        if t == 0:
            return np.zeros(3)

        #### TODO: XXXXXXXXXXX
        #raise NotImplementedError

        xyth = s.lidar[t]['xyth']
        xyth_previous = s.lidar[t - 1]['xyth']

        return smart_minus_2d(xyth, xyth_previous)

    def dynamics_step(s, t):
        """"
        Compute the control using get_control and perform that control on each particle 
        to get the updated locations of the particles in the particle filter, 
        remember to add noise using the smart_plus_2d function to each particle
        """
        #### TODO: XXXXXXXXXXX
        #raise NotImplementedError

        control     = s.get_control(t)

        #no. of particles
        particles   = s.n

        #S.Q = dynamics noise for the state (x,y,yaw)
        for i in range(particles):

            #Noise needs to be updated within the loop otherwise the slam performs poorly
            Noise = np.random.multivariate_normal([0, 0, 0], s.Q)
            s.p[:, i] = smart_plus_2d(s.p[:, i], control)
            s.p[:, i] = smart_plus_2d(s.p[:, i], Noise)


    @staticmethod
    def update_weights(w, obs_logp):
        """
        Given the observation log-probability and the weights of particles w, calculate the
        new weights as discussed in the writeup. Make sure that the new weights are normalized
        """
        #### TODO: XXXXXXXXXXX
        #raise NotImplementedError
        w_update = np.multiply(w, np.exp(obs_logp))
        w_update = w_update / np.sum(w_update)
        return w_update


    def observation_step(s, path, not_in_map, t):
        """
        This function does the following things
            1. updates the particles using the LiDAR observations
            2. updates map.log_odds and map.cells using occupied cells as shown by the LiDAR data

        Some notes about how to implement this.
            1. As mentioned in the writeup, for each particle
                (a) First find the head, neck angle at t (this is the same for every particle)
                (b) Project lidar scan into the world frame (different for different particles)
                (c) Calculate which cells are obstacles according to this particle for this scan,
                calculate the observation log-probability
            2. Update the particle weights using observation log-probability
            3. Find the particle with the largest weight, and use its occupied cells to update the map.log_odds and map.cells.
        You should ensure that map.cells is recalculated at each iteration (it is simply the binarized version of log_odds). 
        map.log_odds is of course maintained across iterations.
        """
        #### TODO: XXXXXXXXXXX
        #raise NotImplementedError
        #(a) First find the head, neck angle at t (this is the same for every particle
        
        # Update odometry for plotting

        necks   = s.joint['head_angles'][0][s.find_joint_t_idx_from_lidar(t)]
        heads   = s.joint['head_angles'][1][s.find_joint_t_idx_from_lidar(t)]
        
        
        #(b) Project lidar scan into the world frame (different for different particles)
        log_p = np.zeros(s.p.shape[1])

        if t == 0:
            lidar_scan      = s.rays2world(t, s.p[:, 0], s.lidar[t]['scan'], heads, necks, s.lidar_angles)
            occupied_grid   = s.map.grid_cell_from_xy(lidar_scan[0],lidar_scan[1])
            
            s.map.cells = np.zeros_like(s.map.cells)
            
            for i in occupied_grid.T:
                s.map.cells[i[0],i[1]] = 1

            path            = s.map.grid_cell_from_xy(s.p.T[0][0], s.p.T[0][1])
            path            = path.reshape(2, 1)

            #Initial grid locations
            grid_locations_0    = s.map.grid_cell_from_xy(s.p.T[0][0],s.p.T[0][1])
            
            #####FREEE COORDNIATES BETWEEN OBSTACLES AND US ###########
            
            r1 = grid_locations_0

            for obstacle in occupied_grid.T:
                step_size  = int(np.linalg.norm(obstacle - grid_locations_0.T))

                xy_coor1 = (np.linspace(grid_locations_0[0], obstacle[0], step_size, endpoint=False, dtype=int)).T
                xy_coor2 = (np.linspace(grid_locations_0[1], obstacle[1], step_size, endpoint=False, dtype=int)).T

                r1 = np.hstack((r1, np.vstack((xy_coor1,xy_coor2))))

            free_coor = np.unique(r1, axis=1)

            ############################################################

            for i in free_coor.T:
                s.map.cells[i[0],i[1]] = 0

            not_in_map = np.ones(s.map.cells.shape)
            
            for i in free_coor.T:
                not_in_map[i[0],i[1]] = 0
            
            for i in occupied_grid.T:
                not_in_map[i[0],i[1]] = 0

            not_in_map = np.argwhere(not_in_map)



        elif t!=0:

            for i,p in enumerate(s.p.T):
                lidar_scan      = s.rays2world(t, p, s.lidar[t]['scan'], heads, necks, s.lidar_angles)
                occupied_grid_t = s.map.grid_cell_from_xy(lidar_scan[0],lidar_scan[1])
                for j in occupied_grid_t.T:
                    log_p[i] += s.map.cells[j[0], j[1]]
            
            s.w                 = s.update_weights(s.w, log_p - s.log_sum_exp(log_p))
            grid_locations_T    = s.map.grid_cell_from_xy(s.p[:,np.argmax(s.w)][0],s.p[:,np.argmax(s.w)][1])
            path                = np.hstack((path, grid_locations_T.reshape(2, 1)))


            lidar_scan          = s.rays2world(t, s.p[:, np.argmax(s.w)], s.lidar[t]['scan'], heads, necks, s.lidar_angles)
            occup_coor_T        = s.map.grid_cell_from_xy(lidar_scan[0],lidar_scan[1])

            
            r1 = grid_locations_T
            
            #####FREEE COORDNIATES BETWEEN OBSTACLES AND US ###########
            for obstacle in occupied_grid_t.T:
                step_size  = int(np.linalg.norm(obstacle - grid_locations_T.T))

                xy_coor1 = (np.linspace(grid_locations_T[0], obstacle[0], step_size, endpoint=False, dtype=int)).T
                xy_coor2 = (np.linspace(grid_locations_T[1], obstacle[1], step_size, endpoint=False, dtype=int)).T

                r1 = np.hstack((r1, np.vstack((xy_coor1,xy_coor2))))

            free_coor_T = np.unique(r1, axis=1)

            ############################################################
            for i in occup_coor_T.T:
                s.map.log_odds[i[0],i[1]] += s.lidar_log_odds_occ

            for i in free_coor_T.T:
                s.map.log_odds[i[0],i[1]] += s.lidar_log_odds_free

            s.map.log_odds = np.clip(s.map.log_odds, -s.map.log_odds_max, s.map.log_odds_max)

            free_threshold = s.map.log_odds_thresh * -1

            s.map.cells = np.zeros_like(s.map.cells)
            s.map.cells[s.map.log_odds >= s.map.log_odds_thresh]       = 1
            s.map.cells[s.map.log_odds <= free_threshold       ]       = 0

            not_in_map_1  = s.map.log_odds > free_threshold
            not_in_map_2  = s.map.log_odds < s.map.log_odds_thresh
            not_in_map    = np.logical_and(not_in_map_1, not_in_map_2)

            s.resample_particles()

        return path.copy(), not_in_map.copy()
        
    def resample_particles(s):
        """
        Resampling is a (necessary) but problematic step which introduces a lot of variance
        in the particles. We should resample only if the effective number of particles
        falls below a certain threshold (resampling_threshold). A good heuristic to
        calculate the effective particles is 1/(sum_i w_i^2) where w_i are the weights
        of the particles, if this number of close to n, then all particles have about
        equal weights and we do not need to resample
        """
        e = 1/np.sum(s.w**2)
        logging.debug('> Effective number of particles: {}'.format(e))
        if e/s.n < s.resampling_threshold:
            s.p, s.w = s.stratified_resampling(s.p, s.w)
            logging.debug('> Resampling')