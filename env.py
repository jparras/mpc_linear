import gym
import numpy as np
from kalman import KalmanFilter
from agents import Agent
from estimators import TransitionEstimator


def perturbation_swirl(x, y, params):  # Swirl
    px = params[0] * (y - params[2]) / np.sqrt(np.square(x - params[1]) + np.square(y - params[2]) + 1e-3)
    py = - params[0] * (x - params[1]) / np.sqrt(np.square(x - params[1]) + np.square(y - params[2]) + 1e-3)
    return (px, py)


def perturbation_current_h(x, y, params):  # Horizontal current
    px = params[0] * np.exp(-np.square(y - params[1]) / (params[2] ** 2 + 1e-3))
    py = np.zeros_like(y)
    return (px, py)


def perturbation_current_v(x, y, params):  # Vertical current
    px = np.zeros_like(x)
    py = params[0] * np.exp(-np.square(x - params[1]) / (params[2] ** 2 + 1e-3))
    return (px, py)


def perturbation_const(x, y, params):  # Constant perturbation
    px = params[0] * np.cos(params[1]) * np.ones_like(x)
    py = params[0] * np.sin(params[1]) * np.ones_like(y)
    return (px, py)


class PerturbationModel:  # Class to implement the perturbation
    def __init__(self, mode, params=None, n_params=3):
        self.mode = mode  # swirl, current_h, current_v or const
        self.data = None
        self.n_params = n_params
        if params is None:
            if self.mode == 'const':
                self.params = np.random.rand(n_params) * np.array([1, 2 * np.pi])  # amplitude and angle!! angle goes from 0 to 2 * pi
            else:
                self.params = np.random.rand(n_params)  # c, x0, y0 / a, mu, sigma (swirl / current models)
        else:
            self.params = params

    def perturbation(self, x, y):
        if self.mode == 'swirl':
            return perturbation_swirl(x, y, params=self.params)
        elif self.mode == 'current_h':
            return perturbation_current_h(x, y, params=self.params)
        elif self.mode == 'current_v':
            return perturbation_current_v(x, y, params=self.params)
        elif self.mode == 'const':
            return perturbation_const(x, y, params=self.params)
        else:
            raise RuntimeError('Perturbation mode not recognized')


class UuvPertSingle(gym.Env):
    def __init__(self, max_force=2, max_pos=10, max_vel=2, pert_mode=None, pert_params=np.array([0.5, 5, 1]),
                 partial_obs=False, type='linear'):

        self.state_size = 4
        self.action_size = 2

        self.obs_low = np.array([-max_pos, -max_pos, -max_vel, -max_vel])
        self.obs_high = np.array([max_pos, max_pos, max_vel, max_vel])

        self.observation_space = gym.spaces.Box(low=np.float32(self.obs_low), high=np.float32(self.obs_high))

        self.ac_low = np.array([-max_force, -max_force])
        self.ac_high = np.array([max_force, max_force])

        self.action_space = gym.spaces.Box(low=np.float32(self.ac_low), high=np.float32(self.ac_high))

        # Simulation parameters
        self.n_max = 500  # Max number of time steps

        self.time_step = 1e-1  # To obtain actual trajectories values
        self.distance_th = 1 # Distance threshold to consider convergence: ball centered at position!
        self.distance_max = 2e1  # Distance threshold to consider divergence: ball centered at position!
        self.k = 5e-1  # Friction coefficient
        self.partial_obs = partial_obs
        self.type = type
        # Internal simulation parameters
        self.state = None
        self.state_initial = None
        self.n = None
        self.actual_pert_model = None  # Perturbation model of the environment
        self.pert_mode = pert_mode
        self.pert_params = pert_params
        self.par_state = None
        self.success = False
        self.timeout = False
        self.failure = False
        self.px = 0
        self.py = 0

    def get_obs(self):  # Returns the observation (note that it need not be the state if there is partial observation)
        if self.partial_obs:
            return self.par_obs()
        else:
            return self.state

    def get_state(self):  # Returns the actual state
        return self.state

    def par_obs(self):
        '''
        x = np.random.normal(self.state[0, 0], 0.1)
        y = np.random.normal(self.state[0, 1], 0.1)
        vx = np.random.normal(self.state[0, 2], 0.1)
        vy = np.random.normal(self.state[0, 3], 0.1)
        self.par_state = np.array([[x,y,vx,vy]])
        '''
        self.par_state = np.random.normal(self.state, 0.1)
        return self.par_state

    def clip(self, x, low, high):  # Clip the x vector in each of its dimensions!
        for i in range(x.size):  # Clip state components!
            x[i] = np.clip(x[i], low[i], high[i])
        return x

    def reset(self, state=None):

        if state is None:
            self.state = self.clip(np.random.rand(self.state_size) * (self.obs_high - self.obs_low)
                                   + self.obs_low, self.obs_low, self.obs_high)
        else:
            self.state = self.clip(np.squeeze(state), self.obs_low, self.obs_high)

        self.state = np.reshape(self.state, [1, len(self.obs_low)])
        self.state_initial = self.state
        self.n = 0
        self.done = False
        self.success = False
        self.timeout = False
        self.failure = False

        if self.pert_mode is not None:
            #self.pert_params = np.array([np.random.normal(0, 0.5), np.random.normal(0, 5), np.random.normal(0, 5)])
            self.actual_pert_model = PerturbationModel(mode=self.pert_mode, params=self.pert_params)  # Actual perturbation model, unknown to the agent!

        return self.get_obs()
    '''
    def action_adaptation(self, action):
        action = self.clip(action, self.ac_low, self.ac_high)  # Limit action values
        return self.ac_high * action / np.sqrt(np.sum(np.square(action)))  # Actions must be sine / cosine!
    '''

    def step(self, action):
        # Clip the action
        action[0] = np.clip(action[0], a_min=self.ac_low, a_max=self.ac_high)

        # Get state components
        x = self.state[0, 0]
        y = self.state[0, 1]
        u = self.state[0, 2]
        v = self.state[0, 3]

        # Get perturbation
        if self.actual_pert_model is not None:
            px, py = self.actual_pert_model.perturbation(x, y)  # Actual perturbation
            self.px = px
            self.py = py
        else:
            px, py = 0, 0

        # State transition for position: it is always the same
        x_next = x + u * self.time_step
        y_next = y + v * self.time_step
        # State transition for velocities: depends on the model!
        if self.type == 'linear':
            u_next = (action[0, 0] + px - self.k * u) * self.time_step + u
            v_next = (action[0, 1] + py - self.k * v) * self.time_step + v
        else: # Non linear transition function
            u_next = (action[0, 0] + px - self.k * u**2 * np.sign(u)) * self.time_step + u
            v_next = (action[0, 1] + py - self.k * v**2 * np.sign(v)) * self.time_step + v

        next_state = np.vstack([x_next, y_next, u_next, v_next]).T


        distance_to_target = np.sqrt(np.sum(np.square(next_state[0, 0:2])))
        done = False

        if distance_to_target <= self.distance_th or self.n >= self.n_max or distance_to_target >= self.distance_max:
            done = True

        if distance_to_target <= self.distance_th: # Success flag
            self.success = True

        if self.n >= self.n_max: # Timeout flag
            self.timeout = True

        if distance_to_target >= self.distance_max: # Failure flag
            self.failure = True

        reward = -1
        self.n += 1

        self.state = next_state

        return self.get_obs(), reward, done, (px, py)

    def get_transition_matrices_linear(self):
        A = np.array([[1, 0, self.time_step, 0], [0, 1, 0, self.time_step],
                      [0, 0, 1 - self.k * self.time_step, 0], [0, 0, 0, 1 - self.k * self.time_step]])
        B = np.array([[.0, .0], [.0, .0], [self.time_step, .0], [.0, self.time_step]])
        return A, B

    def get_kalman(self, kalman_type):
        F, B = self.get_transition_matrices_linear()
        P = np.array([[9, .0, .0, .0], [.0, 9, .0, .0], [.0, .0, 4, .0], [.0, .0, .0, 4]])
        H = np.eye(4)
        R = 0.5 * np.eye(4)
        Q = np.array([[.1, .0, .05, .0], [.0, .1, .0, .05], [.05, .0, .1, .0], [.0, .05, .0, .1]])

        if kalman_type == 'linear':
            return KalmanFilter(kalman_type, self.state_size, self.action_size, F, H, P, Q, R, B)
        elif kalman_type == 'ekf':
            if self.type == 'linear':  # Linear transition
                def Freal(state, ac):
                    return F @ state + B @ ac

                def Hreal(state):  # Simple measurement function
                    return H @ state

                return KalmanFilter(kalman_type, self.state_size, self.action_size, Freal, Hreal, P, Q, R, B, Fjac=F, Hjac=H)
            else:
                raise NotImplementedError
        else:
            return None  # No Kalman filter used!

    def get_agent(self, agent_type, N=20):

        if agent_type == 'mpc':
            #Q = np.array([[100, .0, .0, .0], [.0, 100, .0, .0], [.0, .0, 1, .0], [.0, .0, .0, 1]])
            #R = 0.1 * np.eye(2)
            #gx = np.array([10, 10, 3, 3, 10, 10, 3, 3])
            #gu = np.array([2, 2, 2, 2]) # Note: ensure that gu and ac_max below are consistent!
            gx = np.concatenate((self.obs_high, self.obs_high))
            gu = np.concatenate((self.ac_high, self.ac_high))
        else:  # Use Ricatti's costs for the other cases!
            #Q = np.array([[0.175, .0, .0, .0], [.0, 0.175, .0, .0], [.0, .0, 0.325, .0], [.0, .0, .0, 0.325]])
            #R = np.eye(2)
            gu = gx = None
        #Q = np.array([[1, .0, .0, .0], [.0, 1, .0, .0], [.0, .0, 0.01, .0], [.0, .0, .0, 0.01]])
        #R = 0.01 * np.eye(2)
        #Q = np.array([[1, .0, .0, .0], [.0, 1, .0, .0], [.0, .0, 0.01, .0], [.0, .0, .0, 0.01]])
        Q = np.array([[100, .0, .0, .0], [.0, 100, .0, .0], [.0, .0, 1, .0], [.0, .0, .0, 1]])
        R = 0.1 * np.eye(2)
        A, B = self.get_transition_matrices_linear()

        return Agent(agent_type, obs_dim=self.state_size, ac_dim=self.action_size, ac_max=self.ac_high[0],
                     ac_min=self.ac_low[0], A=A, B=B, Q=Q, R=R, N=N, gx=gx, gu=gu)

    def get_transition_estimator(self, warmup, estimate_transition, estimate_perturbation):
        return TransitionEstimator(self.state_size, self.action_size,
                                   warmup=warmup, estimate_transition=estimate_transition,
                                   estimate_perturbation=estimate_perturbation, delta=self.time_step, k=self.k,
                                   est_method='bfgs')


