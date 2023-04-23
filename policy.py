import numpy as np
import numpy as np
#import seaborn as sns
import matplotlib.pylab as plt
import os

def Q_init():
    #DEFINING THE INITIAL STATESPACE
    statespace              = np.ones((10,10))*-1
    statespace[0:,0]        = -10
    statespace[0:,-1]       = -10
    statespace[0,0:]        = -10
    statespace[-1,0:]       = -10
    
    statespace[4:8,4]       = -10
    statespace[7,5]         = -10
    
    statespace[2,3:7]       = -10
    
    statespace[4,7]         = -10
    statespace[5,7]         = -10
    
    statespace[1,8]         = 10
    
    # print('q_init')
    # print(statespace)
    return np.flip(statespace,0)


def TransitionMatrix():

    #TRANSITION MATRIX FOR 4 DIRECTIONS: NORTH, SOUTH, EAST WEST
    transition_matrix = np.zeros((100,100,4))

    #OBSTACLES AT THE RIGHT MOST ED
    last_column = np.array([9,19,29,39,49,59,69,79,89,99])

    ############################################################################################################

    #North
    for i in range(10,transition_matrix.shape[0]):
        if i == 88:
            transition_matrix[i,i,0] = 1
        
        else:
            transition_matrix[i,i-10,0] = 0.7
        
            if i%10 == 0:
                transition_matrix[i,i,0]  = 0.15
                transition_matrix[i,i+1,0]  = 0.15
            
            elif (i%10!=0) & (i not in last_column):
                transition_matrix[i,i,0]  = 0.1
                transition_matrix[i,i+1,0]  = 0.1
                transition_matrix[i,i-1,0]  = 0.1

            elif (i in last_column):
                transition_matrix[i,i,0]  = 0.15
                transition_matrix[i,i-1,0]  = 0.15

    for i in range(10):
        if i == 0:
            transition_matrix[i,i,0] = 0.5
            transition_matrix[i,i+1,0] = 0.5
        elif 0 < i < 9:
            transition_matrix[i,i,0] = 0.33
            transition_matrix[i,i-1,0] = 0.33
            transition_matrix[i,i+1,0] = 0.33
        elif i == 9:
            transition_matrix[i,i,0] = 0.5
            transition_matrix[i,i-1,0] = 0.5
    
    # print('North')
    # print(np.sum(transition_matrix[:,:,0],1))
    ############################################################################################################
    #South
    for i in range(transition_matrix.shape[0]-10):
        if i == 88:
            transition_matrix[i,i,1] = 1

        else:
            transition_matrix[i,i+10,1] = 0.7
        
            if i%10 == 0:
                transition_matrix[i,i,1]  = 0.15
                transition_matrix[i,i+1,1]  = 0.15
            
            elif (i%10!=0) & (i not in last_column):
                transition_matrix[i,i,1]    = 0.1
                transition_matrix[i,i+1,1]  = 0.1
                transition_matrix[i,i-1,1]  = 0.1

            elif (i in last_column):
                transition_matrix[i,i,1]    = 0.15
                transition_matrix[i,i-1,1]  = 0.15

    for i in range(90,transition_matrix.shape[0]):
        if i == 90:
            transition_matrix[i,i,1]   = 0.5
            transition_matrix[i,i+1,1] = 0.5
        elif 90 < i < 99:
            transition_matrix[i,i,1]   = 0.33
            transition_matrix[i,i-1,1] = 0.33
            transition_matrix[i,i+1,1] = 0.33
        elif i == 99:
            transition_matrix[i,i,1]   = 0.5
            transition_matrix[i,i-1,1] = 0.5
    
    # print('South')
    # print(np.sum(transition_matrix[:,:,1],1))
    
    ############################################################################################################    
    #East
    for i in range(transition_matrix.shape[0]):
        if i == 88:
            transition_matrix[i,i,2] = 1
        else:
            if i not in last_column:
                transition_matrix[i,i+1,2] = 0.7
            
            if (i in last_column) & (i > 9) & (i<99):
                transition_matrix[i,i-10,2]  = 0.33
                transition_matrix[i,i+10,2]  = 0.33
                transition_matrix[i,i,2]     = 0.33
            
            elif (i in last_column) & (i == 9):
                transition_matrix[i,i,2]     = 0.5
                transition_matrix[i,i+10,2]  = 0.5

            elif (i in last_column) & (i == 99):
                transition_matrix[i,i,2]     = 0.5
                transition_matrix[i,i-10,2]  = 0.5 
                
    for i in range(10-1):
        transition_matrix[i,i+10,2]  = 0.15
        transition_matrix[i,i,2]  = 0.15
    
    for i in range(90,transition_matrix.shape[0]-1):
        transition_matrix[i,i-10,2]  = 0.15
        transition_matrix[i,i,2]  = 0.15

    for i in range(10,transition_matrix.shape[0]-10):
        if (i not in last_column)  & (i!=88):
            transition_matrix[i,i-10,2]  = 0.1
            transition_matrix[i,i+10,2]  = 0.1
            transition_matrix[i,i,2]     = 0.1

    # print('East')
    # print(np.sum(transition_matrix[:,:,2],1))

    ############################################################################################################
    #West
    for i in range(transition_matrix.shape[0]):
        if i == 88:
            transition_matrix[i,i,3] = 1
        else:
            if i%10 != 0:
                transition_matrix[i,i-1,3] = 0.7
            
            if (i%10 == 0) & (i > 0) & (i<90):
                transition_matrix[i,i-10,3]  = 0.33
                transition_matrix[i,i+10,3]  = 0.33
                transition_matrix[i,i,3]     = 0.33
            
            elif (i%10 == 0) & (i == 0):
                transition_matrix[i,i,3]     = 0.5
                transition_matrix[i,i+10,3]  = 0.5

            elif (i%10 == 0) & (i == 90):
                transition_matrix[i,i,3]     = 0.5
                transition_matrix[i,i-10,3]  = 0.5 


    for i in range(1,10):
        transition_matrix[i,i+10,3]  = 0.15
        transition_matrix[i,i,3]  = 0.15
    
    for i in range(91,transition_matrix.shape[0]):
        transition_matrix[i,i-10,3]  = 0.15
        transition_matrix[i,i,3]  = 0.15

    for i in range(10,transition_matrix.shape[0]-10):
        if (i%10 != 0) & (i!=88):
            transition_matrix[i,i-10,3]  = 0.1
            transition_matrix[i,i+10,3]  = 0.1
            transition_matrix[i,i,3]     = 0.1

    #print('West')
    #print(np.sum(transition_matrix[:,:,3],1))
    return transition_matrix



def policy_eval(transition_matrix,q,gamma):
    
    #THIS STEP IS ONLY FOR DEFINING THE INITIAL J_PI[0] AND THE 1ST POLICY ITERATION 
    #EQ. 5.22 (2) from notes J_pi[0] = (I-gamma*T)^-1 *q
    J_east  = np.linalg.inv(np.eye(100) - gamma * transition_matrix[:,:,2]) @ q.reshape(100,1)
    J_pi_0  = J_east 

    return J_pi_0

def policy_improvement_loop(J_pi, gamma, q_init, u_k, transition_matrix):
    #THE ONLY DIFFERENCE BETWEEN THIS FUNCTION AND THE PREVIOUS INITIAL FUNCTION IS THE UPDATE STEP FOR J_pi
    #eq 5.22 Update step for J_pi =  qu + gamma * Transition_matrix * J_pi_prev

    J_pi_north       = q_init.reshape(100,1) + gamma * transition_matrix[:,:,0] @ J_pi.reshape(100,1)
    J_pi_south       = q_init.reshape(100,1) + gamma * transition_matrix[:,:,1] @ J_pi.reshape(100,1)
    J_pi_east        = q_init.reshape(100,1) + gamma * transition_matrix[:,:,2] @ J_pi.reshape(100,1)
    J_pi_west        = q_init.reshape(100,1) + gamma * transition_matrix[:,:,3] @ J_pi.reshape(100,1)

    J_combo          = np.hstack((J_pi_north,J_pi_south,J_pi_east,J_pi_west))

    #Policy IMprovement step 
    # U(X) = argmin E[q(x,u) + J_pi(k)(f(x,u) +e)]
    J_new_north      = q_init.reshape(100,1) + gamma * J_combo[:,0].reshape(100,1)
    J_new_south      = q_init.reshape(100,1) + gamma * J_combo[:,1].reshape(100,1)
    J_new_east       = q_init.reshape(100,1) + gamma * J_combo[:,2].reshape(100,1)
    J_new_west       = q_init.reshape(100,1) + gamma * J_combo[:,3].reshape(100,1)

    J_new            = np.hstack((J_new_north,J_new_south,J_new_east,J_new_west))

    #U_k is the argmin of the above equations. It is the policy (N,S,E,W) that the robot can take at each state
    u_k              = (np.argmax(J_new,1) + 1).reshape(10,10)

    #THE NEW J_pi[1] IS INDEXED BASED ON THE BEST CONTROL ACTION
    J_pi             = np.amax(J_new,axis = 1)
    #for i in range(u_north.shape[0]):
    
    return  J_pi, u_k

def policyplot(u_k):
    #Returns the final Policy and the direction of the robot to move at each state
    u_final     = []
    u_direction = []
    for i in u_k.flatten():
        if (i == 1):
            u_final.append('N')
            u_direction.append(u'N \u2191')
        if (i == 2):
            u_final.append('S')
            u_direction.append(u'S \u2193')
        if (i == 3):
            u_final.append('E')
            u_direction.append(u'E \u2192')
        if (i == 4):
            u_final.append('W')
            u_direction.append(u'W \u2190')

    u_final = np.array(u_final).reshape(10,10)
    u_direction = np.array(u_direction).reshape(10,10)
    # print(u_final)
    # print(u_direction)
    return u_final,u_direction



def final_plot(J_pi_new, direction_new,i):
    #function to plot the final value function and the policy map
    heatmap = plt.pcolor(J_pi_new)

    for y in range(J_pi_new.shape[0]):
        for x in range(J_pi_new.shape[1]):
            plt.text(x + 0.5, y + 0.5, '%.4f' % J_pi_new[y, x],
                    horizontalalignment='center',
                    verticalalignment='center',
                    )
    plt.colorbar(heatmap)
    plt.title("Heat Map of Value Function of J_pi({})".format(i))  
    plt.show()
    #plt.savefig(os.path.join('heatmap_%02d.jpg'%(i)))

    heatmap2 = plt.pcolor(J_pi_new)

    for y in range(J_pi_new.shape[0]):
        for x in range(J_pi_new.shape[1]):
            plt.text(x + 0.5, y + 0.5, direction_new[y, x],
                    horizontalalignment='center',
                    verticalalignment='center',
                    )
    plt.colorbar(heatmap2)
    plt.title("Policy at({}) Iteration".format(i)) 
    plt.show()



def main():
    #Defining the four directions and gamma
    north                   = 1
    south                   = 2
    east                    = 3
    west                    = 4

    gamma                   = 0.9

    #defining the Environment
    q_init                  = Q_init()

    #defining the transition matrix
    transition_matrix       = TransitionMatrix()

    #defining the initial policy
    u_k                     = np.zeros((10,10,100))
    u_k[:,:,0]              = east

    #Calling for the first policy iteration
    J_pi = policy_eval(transition_matrix,q_init,gamma)
    
    #Plot of Initial J_pi and Policy
    policy,direction = policyplot(u_k[:,:,0])
    J_pi_new = np.flip(J_pi.reshape(10,10),0)
    direction_new = np.flip(direction,0)
    final_plot(J_pi_new, direction_new,0)


    for i in range(1,100):
        J_pi,u_k[:,:,i]    = policy_improvement_loop(J_pi, gamma, q_init, u_k[:,:,i-1],transition_matrix)
        
        #print(i+1); print(u_k[:,:,i]);print()
        #UNCOMMENT FOR 4 ITERATIONS
        # policy,direction = policyplot(u_k[:,:,i])
        # J_pi_new = np.flip(J_pi.reshape(10,10),0)
        # direction_new = np.flip(direction,0)
        # final_plot(J_pi_new, direction_new,i)
        
        #COMPLETE POLICY EVALUATION
        if (u_k[:,:,i] == u_k[:,:,i-1]).all():
            policy,direction = policyplot(u_k[:,:,i])
            J_pi_new = np.flip(J_pi.reshape(10,10),0)
            direction_new = np.flip(direction,0)
            final_plot(J_pi_new, direction_new,i)
            print(i,'in here')
            break

if __name__ == "__main__":
    main()





















