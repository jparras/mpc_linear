import numpy as np
import time


class experiment(object):
    def __init__(self,
                 env,  # Environment to test
                 kalman_filter_type, # Options are 'none', 'linear' and 'ekf'
                 agent_type, # Options are 'random', 'arctan', 'ricatti', 'mpc'
                 observation_type,  # Options are 'total', 'partial'
                 transition_estimator=False, # Whether to estimate transitions or no
                 estimate_pert='none',  # Disturbance estimator mode
                 burnin_its=0,  # Burnin iterations for agent
                 burnin_type=None,  # Burnin type for agent
                 N=20,  # MPC horizon (20 by default)
                 ):
        self.env = env
        self.kalman_filter_type = kalman_filter_type
        self.agent_type = agent_type
        self.observation_type = observation_type
        self.transition_estimator = transition_estimator
        self.estimate_pert = estimate_pert

        # Now, define all classes needed for the experiment
        self.kalman_filter = env.get_kalman(self.kalman_filter_type)
        self.agent = env.get_agent(agent_type, N=N)
        self.burnin_its = burnin_its
        self.burnin_type = burnin_type

        if self.burnin_its > 0:
            assert self.burnin_type == 'random' or self.burnin_type == 'arctan'
            self.burning_agent = env.get_agent(self.burnin_type, N=N)

        if self.transition_estimator:
            self.tr_est = env.get_transition_estimator(warmup=burnin_its, estimate_transition=transition_estimator,
                                                       estimate_perturbation=estimate_pert)

        # Simulation parameters needed
        self.it = None
        self.results = None


    def run_experiment(self, init_state=None, save_trajs=True):

        t0 = time.time()

        self.agent.reset()

        if self.transition_estimator:
            self.tr_est.reset()

        A = B = C = None  # Initialized as None due to the burnin period

        # If there is a burnin period: use it to update the predictors (but do not account this as time consumed by the algorithm!)
        if self.burnin_its > 0:
            # Reset things
            self.burning_agent.reset()
            _ = self.env.reset(state=init_state)
            burnin_results = {'obs': None, 'state': None}
            burnin_results['obs'] = self.env.get_obs()
            burnin_results['state'] = self.env.get_state()

            for i in range(self.burnin_its):
                agent_input = burnin_results['obs']
                action, converged = self.burning_agent.get_action(agent_input)
                _ = self.env.step(action)

                burnin_results['obs'] = self.env.get_obs()
                burnin_results['state'] = self.env.get_state()

                if self.transition_estimator:  # Estimate A, B and C
                    next_agent_input = burnin_results['obs']
                    A, B, C = self.tr_est.estimate(np.hstack((agent_input, action)), next_agent_input)

        # After the burnin period, start with the proper simulation: "reset" the agent to the initial position for comparison (after the burnin period!)
        _ = self.env.reset(state=init_state)
        self.it = 0
        self.results = {'states': [], 'obs': [], 'actions': [], 'kalman_est': [], 'cost': [], 'converged': [],
                        'success': False, 'timeout': False, 'failure': False,
                        't_action': [], 't_kalman': [], 't_env': [], 't_trans': [], 'total_time': None,
                        'kalman': self.kalman_filter_type, 'agent': self.agent_type, 'observation': self.observation_type,
                        'transition_estimator': self.transition_estimator, 'estimate_pert': self.estimate_pert,
                        'est_norm': [], 'est_pert': [], 'est_pert_params': [], 'real_pert': None, 'real_pert_params': None}

        self.results['obs'].append(self.env.get_obs())
        self.results['states'].append(self.env.get_state())

        if self.estimate_pert != 'none':
            self.results['real_pert'] = self.env.pert_mode
            self.results['real_pert_params'] = self.env.pert_params

        if self.kalman_filter is not None:
            self.kalman_filter.reset(self.results['obs'][-1])

        done = False

        while not done:
            if self.it == 0 or self.kalman_filter is None:  # Use Kalman if available, except on initial state (it has no prediction yet!)
                agent_input = self.results['obs'][-1]
            else:
                agent_input = self.results['kalman_est'][-1]

            t1 = time.time()
            action, converged = self.agent.get_action(agent_input, recompute=True, init_t=10, threshold=self.env.distance_th, A=A, B=B, C=C)  #self.env.n_max+2
            t2 = time.time()
            self.results['t_action'].append(t2 - t1)

            t1 = time.time()
            _, _, done, _ = self.env.step(action)
            t2 = time.time()
            self.results['t_env'].append(t2 - t1)

            self.results['cost'].append(self.agent.get_cost(self.results['states'][-1], action))  # Cost of curent state - action pair
            self.results['actions'].append(action)
            self.results['obs'].append(self.env.get_obs())
            self.results['states'].append(self.env.get_state())
            self.results['converged'].append(converged)

            if self.kalman_filter is not None:
                t1 = time.time()
                self.results['kalman_est'].append(self.kalman_filter.estimate(self.results['obs'][-1], action))
                t2 = time.time()
                self.results['t_kalman'].append(t2 - t1)

            if self.transition_estimator:  # Estimate A and B
                t1 = time.time()
                if self.kalman_filter is None:
                    next_agent_input = self.results['obs'][-1]
                else:
                    next_agent_input = self.results['kalman_est'][-1]
                A, B, C = self.tr_est.estimate(np.hstack((agent_input, action)), next_agent_input)
                if self.kalman_filter is not None:
                    self.kalman_filter.set_linear_matrices(A, B)  # Sets the Kalman Filter with the most recent estimated matrices
                t2 = time.time()
                self.results['t_trans'].append(t2 - t1)
                if A is not None:
                    Ar, Br = self.env.get_transition_matrices_linear()
                    self.results['est_norm'].append(np.linalg.norm(A - Ar) + np.linalg.norm(B - Br) + np.linalg.norm(C))
                if self.estimate_pert != 'none':
                    self.results['est_pert'].append(self.tr_est.estimated_pert['est_pert'])
                    self.results['est_pert_params'].append(self.tr_est.estimated_pert['est_params'])

            self.it += 1

        self.results['success'] = self.env.success
        self.results['timeout'] = self.env.timeout
        self.results['failure'] = self.env.failure
        self.results['total_time'] = time.time() - t0

        if not save_trajs:  # Drop values that we do not want to save for later
            self.results['states'] = []
            self.results['obs'] = []
            self.results['actions'] = []
            self.results['kalman_est'] = []

        return self.results
