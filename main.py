from time import time
import numpy as np
from utils import visualize
from casadi import *
###########################################################
###########################################################
# GPI part 
# Control sets
# Select one of them
# U = [[0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]
U = [[v, omega] for v in [0, 0.25,0.5,0.75,1] for omega in [-1,-0.5,0,0.5,1]]

def mean_next(e, u, ref, ref_next, dt = 0.5):
    return [
        e[0] + ref[0]-ref_next[0] + dt*np.cos(e[2] + ref[2])*u[0],
        e[1] + ref[1]-ref_next[1] + dt*np.sin(e[2] + ref[2])*u[0],
        e[2] + ref[2]-ref_next[2] + dt*u[1]
        ]
def disc(s):
    return np.array([\
        int((s[0]+3)/0.1),\
        int((s[1]+3)/0.1),\
        int((s[2]+np.pi)/(np.pi/10)),\
    ])
# Discrete to continous
def cont(s):
    return np.array([\
        s[0]/10-3, s[1]/10-3, s[2]/10*np.pi - np.pi\
        ])
def key(e, u):
    return e[0]*4*8*1000**2 + e[1]*4*8*1000 + e[2]*4*8 + 2*u[0]*8 + 2*u[1] + 2 
# Free space constraint
def obstacle_cost(e, ref):
    C1 = [-2,-2]
    C2 = [1,2]
    r = 0.5
    obs_1 = ((e.T[0]+ref[0]-C1[0])**2 + (e.T[1]+ref[1]-C1[1])**2 - 0.5**2 < 0)
    obs_2 = ((e.T[0]+ref[0]-C2[0])**2 + (e.T[1]+ref[1]-C2[1])**2 - 0.5**2 < 0)
    return e*0 + e*1000*(obs_1.T or obs_2.T)
    
# Value function conditioned on e and u
V_e_u_table = dict()
def V_e_u(e, u, ref):
    if key(e,u) in V_e_u_table:
        return V_e_u_table[key(e,u)]
    a = 5
    b = 2
    c = 0.2
    d = 0.2
    P = np.array([[0.00163118, 0.03712553, 0.00163118],\
            [0.03712553, 0.84497315, 0.03712553],\
            [0.00163118, 0.03712553, 0.00163118]])
    E = np.array([[[-1,-1,0], [-1,0,0],[-1,1,0]], \
            [[0,-1,0], [0,0,0],[0,1,0]], \
            [[1,-1,0], [1,0,0],[1,1,0]]])
    e_ = E + np.array(e)
    q = cont(e_.T).T
    theta = q.T[-1].T[0][0]
    q = q.T[:2].T.reshape(-1,2)
    V_e_u_table[key(e,u)] = a*sum((np.einsum('Bi,Bi ->B', q, q).reshape(3,3)).reshape(-1)) + \
       b*(1-np.cos(theta))**2 + c*u[0]**2 + d*u[1]**2 + sum((obstacle_cost(e, ref)*P).reshape(-1))
    return  V_e_u_table[key(e,u)]


def V_e_u_2(e, u, ref_traj, T,T_fin, gamma=0.7):
    if T == T_fin+1:
        return V_e_u(e, [0,0], ref_traj[T])
    P = np.array([[0.00163118, 0.03712553, 0.00163118],\
            [0.03712553, 0.84497315, 0.03712553],\
            [0.00163118, 0.03712553, 0.00163118]])
    E = np.array([[[-1,-1,0], [-1,0,0],[-1,1,0]], \
            [[0,-1,0], [0,0,0],[0,1,0]], \
            [[1,-1,0], [1,0,0],[1,1,0]]])
    e = E + np.array(e)
    V_prev=  np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            e_next = disc(mean_next(cont(list(e[i][j])), u[T-1], ref_traj[T-1], ref_traj[T]))
            V_prev[i][j] = V_e_u(e[i][j], u[T-1], ref_traj[T-1]) + gamma*V_e_u_2(e_next, u, ref_traj, T+1, T_fin)
    return sum((V_prev*P).reshape(-1)) 
###########################################################
############################################################
# Casadi functions
# Terminal cost
def q(e):
    q11 = 1
    q22 = 1
    q12 = 0
    q_theta = 0.5
    return q11*e[0]**2 + q22*e[1]**2 + q12*e[0]*e[1] + \
        q_theta*(1-cos(e[2]))**2
# Stage cost
# Here e is a constant rather than a 
# symbolic variable

def l(e, u):
    q11 = 1
    q22 = 1
    q12 = 0
    q_theta = 1
    u11 = 1
    u22 = 2
    u12 = 0
    return q11*e[0]**2 + q22*e[1]**2 + q12*e[0]*e[1] + \
        q_theta*(1-cos(e[2]))**2 + u11*u[0]**2 + u22*u[1]**2 + \
        u12*u[0]*u[1]

# Dynamics constraint
def g1(e, u, e_prev, ref, ref_prev, delta):
    return [e[0] -e_prev[0] - \
        delta*cos(e_prev[2] + ref_prev[2])*u[0] - \
            (ref_prev[0] - ref[0]), \
            e[1] -e_prev[1] - \
        delta*sin(e_prev[2] + ref_prev[2])*u[0] - \
            (ref_prev[1] - ref[1]), \
            e[2] -e_prev[2] - \
        delta*u[1] - \
            (ref_prev[2] - ref[2])]

# Free space constraint
def F(e, ref):
    C1 = [-2,-2]
    C2 = [1,2]
    r = 0.5
    return [ \
        (e[0]+ref[0]-C1[0])**2 + (e[1]+ref[1]-C1[1])**2 - 0.5**2 - 0.1,\
        (e[0]+ref[0]-C2[0])**2 + (e[1]+ref[1]-C2[1])**2 - 0.5**2 - 0.1 \
    ]
###########################################################
###########################################################
# Simulation params
np.random.seed(10)
time_step = 0.5 # time between steps in seconds
sim_time = 120    # simulation time

# Car params
x_init = 1.5
y_init = 0.0
theta_init = np.pi/2
v_max = 1
v_min = 0
w_max = 1
w_min = -1

# This function returns the reference point at time step k
def lissajous(k):
    xref_start = 0
    yref_start = 0
    A = 2
    B = 2
    a = 2*np.pi/50
    b = 3*a
    T = np.round(2*np.pi/(a*time_step))
    k = k % T
    delta = np.pi/2
    xref = xref_start + A*np.sin(a*k*time_step + delta)
    yref = yref_start + B*np.sin(b*k*time_step)
    v = [A*a*np.cos(a*k*time_step + delta), B*b*np.cos(b*k*time_step)]
    thetaref = np.arctan2(v[1], v[0])
    return [xref, yref, thetaref]

# This function implements a simple P controller
def simple_controller(cur_state, ref_state):
    k_v = 0.55
    k_w = 1.0
    v = k_v*np.sqrt((cur_state[0] - ref_state[0])**2 + (cur_state[1] - ref_state[1])**2)
    v = np.clip(v, v_min, v_max)
    angle_diff = ref_state[2] - cur_state[2]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi ) - np.pi
    w = k_w*angle_diff
    w = np.clip(w, w_min, w_max)
    return [v,w]

# CEC controller
# T = 1
def cec_controller_1(cur_state, ref_traj):
    e_prev = cur_state - ref_traj[0]
    # State
    e_1 = SX.sym('e_1',3)

    # Control
    u_1 = SX.sym('u_1',2)

    gamma = 0.8
    nlp = {'x' : vertcat(e_1,u_1), \
       'f' : q(e_1) + gamma**0*0.2*l(e_prev,u_1),
       'g' : vertcat(*g1(e_1, u_1, e_prev, ref_traj[1], ref_traj[0], time_step), \
             u_1[0], u_1[1], \
             *F(e_1, ref_traj[1])
             )}
    S = nlpsol('S', 'ipopt', nlp)
    
    r = S(x0=[0, 0, 0, 0, 0],\
      lbg=vertcat(0,0,0,\
          0,-1,\
          0,0), \
      ubg=vertcat(0,0,0,\
          1,1,\
          36,36))
    return list(np.array(r['x'][3:5]).reshape(-1)) 
# T = 3
def cec_controller_3(cur_state, ref_traj):
    e_prev = cur_state - ref_traj[0]
    # State
    e_1 = SX.sym('e_1',3)
    e_2 = SX.sym('e_2',3)
    e_3 = SX.sym('e_3',3)

    # Control
    u_1 = SX.sym('u_1',2)
    u_2 = SX.sym('u_2',2)
    u_3 = SX.sym('u_3',2)

    gamma = 0.8
    nlp = {'x' : vertcat(e_1,u_1, e_2,u_2, \
                         e_3,u_3), \
       'f' : q(e_3) + gamma**0*0.2*l(e_prev,u_1) + gamma**1*0.2*l(e_1, u_2) \
            + gamma**2*0.2*l(e_2, u_3),
       'g' : vertcat(*g1(e_1, u_1, e_prev, ref_traj[1], ref_traj[0], time_step), \
             *g1(e_2, u_2, e_1, ref_traj[2], ref_traj[1], time_step), \
             *g1(e_3, u_3, e_2, ref_traj[3], ref_traj[2], time_step), \
             u_1[0], u_1[1], u_2[0], u_2[1], \
             u_3[0], u_3[1], \
             *F(e_1, ref_traj[1]), \
             *F(e_2, ref_traj[2]), \
             *F(e_3, ref_traj[3]))
             }
    S = nlpsol('S', 'ipopt', nlp)
    
    r = S(x0=[0, 0, 0, 0, 0, 
              0, 0, 0, 0, 0,\
              0, 0, 0, 0, 0 ],\
      lbg=vertcat(0,0,0,0,0,0,0,0,0,\
          0,-1,0,-1,0,-1,\
          0,0,0,0,0,0), \
      ubg=vertcat(0,0,0,0,0,0,0,0,0,\
          1,1,1,1,1,1,\
          36,36,36,36,36,36))
    return list(np.array(r['x'][3:5]).reshape(-1)) 

# T = 5
def cec_controller_5(cur_state, ref_traj):
    e_prev = cur_state - ref_traj[0]
    # State
    e_1 = SX.sym('e_1',3)
    e_2 = SX.sym('e_2',3)
    e_3 = SX.sym('e_3',3)
    e_4 = SX.sym('e_4',3)
    e_5 = SX.sym('e_5',3)
    # Control
    u_1 = SX.sym('u_1',2)
    u_2 = SX.sym('u_2',2)
    u_3 = SX.sym('u_3',2)
    u_4 = SX.sym('u_4',2)
    u_5 = SX.sym('u_5',2)

    gamma = 0.8
    nlp = {'x' : vertcat(e_1,u_1, e_2,u_2, \
                         e_3,u_3, e_4,u_4, \
                         e_5,u_5), \
       'f' : q(e_5) + gamma**0*0.2*l(e_prev,u_1) + gamma**1*0.2*l(e_1, u_2) \
            + gamma**2*0.2*l(e_2, u_3) + gamma**3*0.2*l(e_3, u_4) \
            + gamma**3*0.2*l(e_4, u_5),
       'g' : vertcat(*g1(e_1, u_1, e_prev, ref_traj[1], ref_traj[0], time_step), \
             *g1(e_2, u_2, e_1, ref_traj[2], ref_traj[1], time_step), \
             *g1(e_3, u_3, e_2, ref_traj[3], ref_traj[2], time_step), \
             *g1(e_4, u_4, e_3, ref_traj[4], ref_traj[3], time_step), \
             *g1(e_5, u_5, e_4, ref_traj[5], ref_traj[4], time_step), \
             u_1[0], u_1[1], u_2[0], u_2[1], \
             u_3[0], u_3[1], u_4[0], u_4[1],  \
             u_5[0], u_5[1], \
             *F(e_1, ref_traj[1]), \
             *F(e_2, ref_traj[2]), \
             *F(e_3, ref_traj[3]), \
             *F(e_4, ref_traj[4]), \
             *F(e_5, ref_traj[5])  )
             }
    S = nlpsol('S', 'ipopt', nlp)
    
    r = S(x0=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0 \
              , 0, 0, 0, 0, 0],\
      lbg=vertcat(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
          0,-1,0,-1,0,-1,0,-1,0,-1,\
          0,0,0,0,0,0,0,0,0,0), \
      ubg=vertcat(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\
          ,1,1,1,1,1,1,1,1,1,1,\
          36,36,36,36,36,36,36,36,36,36))
    return list(np.array(r['x'][3:5]).reshape(-1)) 
    
# GPI controller 
def GPI_controller(curr_state, ref_traj):
    e = disc(np.array(cur_state) - np.array(ref_traj[0]))

    V_min = np.infty
    u_min = [0,0]

    V = [V_e_u_2(e, [u_1,u_2], ref_trajj,1,2) for u_1 in U for u_2 in U]
    pi = [[u_1,u_2] for u_1 in U for u_2 in U]
    return pi[np.argmin(np.array(V))][0]


# This function implement the car dynamics
def car_next_state(time_step, cur_state, control, noise = True):
    theta = cur_state[2]
    rot_3d_z = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    f = rot_3d_z @ control
    mu, sigma = 0, 0.04 # mean and standard deviation for (x,y)
    w_xy = np.random.normal(mu, sigma, 2)
    mu, sigma = 0, 0.004  # mean and standard deviation for theta
    w_theta = np.random.normal(mu, sigma, 1)
    w = np.concatenate((w_xy, w_theta))
    if noise:
        return cur_state + time_step*f.flatten() + w
    else:
        return cur_state + time_step*f.flatten()

if __name__ == '__main__':
    # Obstacles in the environment
    obstacles = np.array([[-2,-2,0.5], [1,2,0.5]])
    # Params
    traj = lissajous
    ref_traj = []
    error = 0.0
    error_mean = 0.0
    car_states = []
    times = []
    # Start main loop
    main_loop = time()  # return time in sec
    # Initialize state
    cur_state = np.array([x_init, y_init, theta_init])
    cur_iter = 0
    curr_ref = cur_state

    # ref_traj.append(cur_state)
    # Main loop
    while (cur_iter * time_step < sim_time):
        t1 = time()
        # Get reference state
        cur_time = cur_iter*time_step
        cur_ref = traj(cur_iter)
        # Save current state and reference state for visualization
        ref_traj.append(cur_ref)
        car_states.append(cur_state)
        # Look five steps ahead
        ref_trajj = [traj(cur_iter-1), traj(cur_iter), \
                     traj(cur_iter+1), traj(cur_iter+2)\
                     , traj(cur_iter+3), traj(cur_iter+4)]
        ################################################################
        # Generate control input
        # TODO: Replace this simple controller with your own controller
        # control = simple_controller(cur_state, cur_ref)
        # CEC Controller
        # control = cec_controller_1(cur_state, ref_trajj)
        # control = cec_controller_3(cur_state, ref_trajj)
        # control = cec_controller_5(cur_state, ref_trajj)
        
        # GPI controller
        control =  GPI_controller(cur_state, ref_trajj)
        print("[v,w]", control)

        ################################################################

        # Apply control input
        next_state = car_next_state(time_step, cur_state, control, noise=True)
        # Update current state
        cur_state = next_state
        # Loop time
        t2 = time()
        print(cur_iter)
        print(t2-t1)
        times.append(t2-t1)
        error = error + np.linalg.norm(cur_state[:2] - cur_ref[:2])
        error_mean = np.exp(-4*cur_iter * time_step / sim_time)*error_mean + \
                        (1-np.exp(-4*cur_iter * time_step / sim_time))* \
                        np.linalg.norm(cur_state[:2] - cur_ref[:2])
        cur_iter = cur_iter + 1

    main_loop_time = time()
    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('Average iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('Final error: ', error)
    print('Mean error: ', error_mean)

    # Visualization
    ref_traj = np.array(ref_traj)
    car_states = np.array(car_states)
    times = np.array(times)
    visualize(car_states, ref_traj, obstacles, times, time_step, save=True)

