
import click, tqdm, random

from slam import *

def run_dynamics_step(src_dir, log_dir, idx, split, t0=0, draw_fig=False):
    """
    This function is for you to test your dynamics update step. It will create
    two figures after you run it. The first one is the robot location trajectory
    using odometry information obtained form the lidar. The second is the trajectory
    using the PF with a very small dynamics noise. The two figures should look similar.
    """
    slam = slam_t(Q=1e-8*np.eye(3))
    slam.read_data(src_dir, idx, split)

    # trajectory using odometry (xy and yaw) in the lidar data
    d = slam.lidar
    xyth = []
    for p in d:
        xyth.append([p['xyth'][0], p['xyth'][1],p['xyth'][2]])
    xyth = np.array(xyth)

    plt.figure(1); plt.clf();
    plt.title('Trajectory using onboard odometry')
    plt.plot(xyth[:,0], xyth[:,1])
    logging.info('> Saving odometry plot in '+os.path.join(log_dir, 'odometry_%s_%02d.jpg'%(split, idx)))
    plt.savefig(os.path.join(log_dir, 'odometry_%s_%02d.jpg'%(split, idx)))

    # dynamics propagation using particle filter
    # n: number of particles, w: weights, p: particles (3 dimensions, n particles)
    # S covariance of the xyth location
    # particles are initialized at the first xyth given by the lidar
    # for checking in this function
    n = 3
    w = np.ones(n)/float(n)
    p = np.zeros((3,n), dtype=np.float64)
    slam.init_particles(n,p,w)
    slam.p[:,0] = deepcopy(slam.lidar[0]['xyth'])

    print('> Running prediction')
    t0 = 0
    T = len(d)
    ps = deepcopy(slam.p)   # maintains all particles across all time steps
    plt.figure(2); plt.clf();
    ax = plt.subplot(111)
    for t in tqdm.tqdm(range(t0+1,T)):
        slam.dynamics_step(t)
        ps = np.hstack((ps, slam.p))

        if draw_fig:
            ax.clear()
            ax.plot(slam.p[0], slam.p[0], '*r')
            plt.title('Particles %03d'%t)
            plt.draw()
            plt.pause(0.01)

    plt.plot(ps[0], ps[1], '*c')
    plt.title('Trajectory using PF')
    logging.info('> Saving plot in '+os.path.join(log_dir, 'dynamics_only_%s_%02d.jpg'%(split, idx)))
    plt.savefig(os.path.join(log_dir, 'dynamics_only_%s_%02d.jpg'%(split, idx)))

def run_observation_step(src_dir, log_dir, idx, split, is_online=False):
    """
    This function is for you to debug your observation update step
    It will create three particles np.array([[0.2, 2, 3],[0.4, 2, 5],[0.1, 2.7, 4]])
    * Note that the particle array has the shape 3 x num_particles so
    the first particle is at [x=0.2, y=0.4, z=0.1]
    This function will build the first map and update the 3 particles for one time step.
    After running this function, you should get that the weight of the second particle is the largest since it is the closest to the origin [0, 0, 0]
    """
    path       = np.array([0,0])
    not_in_map = np.array([])
    odom       = None

    slam = slam_t(resolution=0.05)
    slam.read_data(src_dir, idx, split)

    # t=0 sets up the map using the yaw of the lidar, do not use yaw for
    # other timestep
    # initialize the particles at the location of the lidar so that we have some
    # occupied cells in the map to calculate the observation update in the next step
    
    t0 = 0
    xyth = slam.lidar[t0]['xyth']
    xyth[2] = slam.lidar[t0]['rpy'][2]
    logging.debug('> Initializing 1 particle at: {}'.format(xyth))
    slam.init_particles(n=1,p=xyth.reshape((3,1)),w=np.array([1]))

    path, not_in_map = slam.observation_step(path, not_in_map, t=0)
    logging.info('> Particles\n: {}'.format(slam.p))
    logging.info('> Weights: {}'.format(slam.w))

    # reinitialize particles, this is the real test
    logging.info('\n')
    n = 3
    w = np.ones(n)/float(n)
    p = np.array([[2, 0.2, 3],[2, 0.4, 5],[2.7, 0.1, 4]])
    slam.init_particles(n, p, w)

    path, not_in_map = slam.observation_step(path, not_in_map, t=1)
    logging.info('> Particles\n: {}'.format(slam.p))
    logging.info('> Weights: {}'.format(slam.w))

def run_slam(src_dir, log_dir, idx, split):
    """
    This function runs slam. We will initialize the slam just like the observation_step
    before taking dynamics and observation updates one by one. You should initialize
    the slam with n=100 particles, you will also have to change the dynamics noise to
    be something larger than the very small value we picked in run_dynamics_step function
    above.
    """
    path       = None
    not_in_map = np.array([])

    slam = slam_t(resolution=0.05, Q=np.diag([2e-3, 2e-3, 2e-3]))
    slam.read_data(src_dir, idx, split)

    # t=0 sets up the map using the yaw of the lidar, do not use yaw for
    # other timestep
    # initialize the particles at the location of the lidar so that we have some
    # occupied cells in the map to calculate the observation update in the next step
    t0 = 0
    xyth = slam.lidar[t0]['xyth']
    xyth[2] = slam.lidar[t0]['rpy'][2]
    logging.debug('> Initializing 1 particle at: {}'.format(xyth))
    slam.init_particles(n=1, p=xyth.reshape((3, 1)), w=np.array([1]))

    path, not_in_map = slam.observation_step(path, not_in_map, t=0)

    n = 100
    w = np.ones(n)/float(n)
    p = np.zeros((3, n))
    slam.init_particles(n, p, w)

    plt.figure()
    plt.ion()

    ###########################################
    timestep = 10

    mapp                = plt.imshow((1-slam.map.cells)*255, cmap='gray')
    current_position,   = plt.plot(path[1,:], path[0,:], '*', markersize=1, color='red')

    imu_data            = None
    imu_0               = slam.map.grid_cell_from_xy(slam.lidar[t0]['xyth'][:2][0],slam.lidar[t0]['xyth'][:2][1])
    imu_0               = np.array([[imu_0[0]], [imu_0[1]]])
    imu_data            = imu_0

    imu,                = plt.plot(imu_data[1,:], imu_data[0,:], '*', markersize=1, color='blue')

    

    for t in range(1, len(slam.lidar),timestep):
        
        slam.dynamics_step(t)
        path, not_in_map = slam.observation_step(path, not_in_map, t)

        final_cells             = (1 - slam.map.cells) * 255
        final_cells[not_in_map] = 255//2
        
        imu_t         = slam.map.grid_cell_from_xy(slam.lidar[t]['xyth'][:2][0],slam.lidar[t]['xyth'][:2][1])
        imu_t         = np.array([[imu_t[0]], [imu_t[1]]])
        imu_data      = np.hstack((imu_data,imu_t))
        
        mapp.set_data(final_cells)
        current_position.set_data(path[1, :], path[0, :])
        imu.set_data(imu_data[1, :], imu_data[0, :])

        #plt.draw()
        plt.pause(0.001)

    plt.savefig(os.path.join(log_dir, 'Slam_%02d.jpg'%(idx)))



@click.command()
@click.option('--src_dir', default='./ESE650/HW3/ese650_hw3_p2/p2/', help='data directory', type=str)
@click.option('--log_dir', default='./ESE650/HW3/ese650_hw3_p2/p2/logs', help='directory to save logs', type=str)
@click.option('--idx', default='0', help='dataset number', type=int)
@click.option('--split', default='train', help='train/test split', type=str)
@click.option('--mode', default='observation',
              help='choices: dynamics OR observation OR slam', type=str)
def main(src_dir, log_dir, idx, split, mode):
    # Run python main.py --help to see how to provide command line arguments

    if not mode in ['slam', 'dynamics', 'observation']:
        raise ValueError('Unknown argument --mode %s'%mode)
        sys.exit(1)

    np.random.seed(42)
    random.seed(42)

    if mode == 'dynamics':
        run_dynamics_step(src_dir, log_dir, idx, split)
        sys.exit(0)
    elif mode == 'observation':
        run_observation_step(src_dir, log_dir, idx, split)
        sys.exit(0)
    else:
        p = run_slam(src_dir, log_dir, idx, split)
        return p

if __name__=='__main__':
    main()
