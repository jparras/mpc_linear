import numpy as np
from env import UuvPertSingle
import matplotlib.pyplot as plt
from experiment import experiment
from joblib import Parallel, delayed
import pickle
from tabulate import tabulate
from itertools import product


def run_experiment(observation_type, agent_type, kalman_filter_type, transition_estimator, estimate_pert, pert_val):
    np.random.seed(1234)  # For repeatability
    if observation_type == 'total':
        env = UuvPertSingle(partial_obs=False, pert_mode=pert_val['pert_mode'], pert_params=pert_val['pert_params'])
    elif observation_type == 'partial':
        env = UuvPertSingle(partial_obs=True, pert_mode=pert_val['pert_mode'], pert_params=pert_val['pert_params'])
    else:
        raise RuntimeError
    if transition_estimator and not (agent_type == 'mpc' or agent_type == 'ricatti'):  # Use transition estimator with MPC and Ricatti only
        return None
    if not transition_estimator and estimate_pert != 'none':  # Doesn't make sense to estimate the perturbation and not the transition
        return None
    if observation_type == 'total' and kalman_filter_type != 'none':  # For total observation, do not use any Kalman filter (it doesn't make any sense)
        return None
    if agent_type == 'mpc' or agent_type == 'ricatti':
        burnin_its = 500
        burnin_type = 'random'
    else:
        burnin_its = 0
        burnin_type = None
    exp = experiment(env, kalman_filter_type, agent_type, observation_type, transition_estimator=transition_estimator,
                     estimate_pert=estimate_pert, burnin_its=burnin_its, burnin_type=burnin_type, N=20)
    res = exp.run_experiment(init_state=pert_val['init_state'], save_trajs=pert_val['save_trajs'])
    res['pert'] = pert_val

    return res


# Train and test flags
train = False  # Use this to generate trajectories and save them
test = True  # Use this to show results obtained

# Generate init states, and for each, initialize also different perturbations
np.random.seed(1234)  # For repeatability
nTest = 100   # number of trajectories to test
nsave = 3  # Number of trajectories to save per case (to prevent a too large data to save!)
pert_modes = [None, 'swirl', 'current_h', 'current_v', 'const']
pert_values = []
env = UuvPertSingle()
for i in range(nTest):
    _ = env.reset()
    init_state = env.get_state()
    for pert in pert_modes:
        # Randomly sample params for each perturbation and initial condition!
        if pert == 'swirl':
            params = np.array((np.random.uniform(low=-1.5, high=1.5, size=1),  # This parameter is the strength of the perturbation: if it is small (compared to max acceleration), perturbations do not affect, if it is too large, then the agent cannot opose them!
                               np.random.normal(loc=0.0, scale=3, size=1),
                               np.random.normal(loc=0.0, scale=3, size=1)))
        elif pert == 'current_v' or pert == 'current_h':
            params = np.array((np.random.uniform(low=-1.5, high=1.5, size=1),  # This parameter is the strength of the perturbation: if it is small (compared to max acceleration), perturbations do not affect, if it is too large, then the agent cannot opose them!
                               np.random.normal(loc=0.0, scale=3, size=1),
                               np.random.normal(loc=5.0, scale=1, size=1)))
        else:
            params = np.array((np.random.uniform(low=-1.5, high=1.5, size=1),  # This parameter is the strength of the perturbation: if it is small (compared to max acceleration), perturbations do not affect, if it is too large, then the agent cannot opose them!
                               np.random.uniform(low=0, high=2 * np.pi, size=1), # Angle
                               np.random.normal(loc=0.0, scale=3, size=1)))  # The last one is never used indeed


        pert_values.append({'pert_mode': pert, 'pert_params': params, 'init_state': init_state, 'save_trajs': i < nsave, 'i': i})  # For None, params are not needed, but we generate them for easier code


n_threads = 10  # Number of threads: be careful to set this according to your machine!

agents = ['mpc']  # Agent to use
observations = ['total', 'partial']  # Observation model to use
kalmans = ['none', 'linear']  # Kalman filters to use
tran_ests = [False, True]  # Whether to use transition estimator or not
pert_ests = ['none', 'const', 'full']  # None: no disturbance estimator, const: constant estimator, full: full estimator

if train:
    print('Computing ', len(agents) * len(observations) * len(kalmans) * len(tran_ests) * len(pert_ests) * len(pert_values),' tasks')
    res = Parallel(n_jobs=n_threads, verbose=10, batch_size=1)(delayed(run_experiment)
                                                               (observation_type, agent_type, kalman_filter_type,
                                                                transition_estimator, estimate_pert, pert_val)
                                                               for agent_type in agents
                                                               for observation_type in observations
                                                               for kalman_filter_type in kalmans
                                                               for transition_estimator in tran_ests
                                                               for estimate_pert in pert_ests
                                                               for pert_val in pert_values)
    # Save results
    with open("results.pickle", "wb") as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

if test:

    with open("results.pickle", "rb") as handle:
        res = pickle.load(handle)

    # Show results in table format
    print('\n\n[RESULTS] Global results \n\n')
    timeout_prop = []
    table = []
    table.append(['Agent', 'Obs', 'Kalman', 'Trans est', 'Pert est', 'Pert', 'T total', 't action', 't env', 't trans', 't kalman', 'cost / 1e3', 'success', 'timeout', 'failure', 'cost success', 'cost timeout', 'cost failure'])
    for agent_type, observation_type, kalman_filter_type, transition_estimator, estimate_pert, pert_mode in product(agents, observations, kalmans, tran_ests, pert_ests, pert_modes):
        c = []
        t = []
        cs = []
        ct = []
        cf = []
        ta = []
        tk = []
        te = []
        tt = []
        success = 0
        timeout = 0
        failure = 0
        is_present = False  # Flag to test for present values only in the simulation results
        for r in res:
            if r is not None:
                if r['kalman'] == kalman_filter_type and r['agent'] == agent_type and r['observation'] == observation_type and r['transition_estimator'] == transition_estimator and r['estimate_pert'] == estimate_pert and r['pert']['pert_mode'] == pert_mode:
                    is_present = True  #Set to True only if the combination of parameters has been tested!
                    c.append(np.sum(r['cost']))
                    t.append(r['total_time'])
                    ta.append(np.sum(r['t_action']))
                    tk.append(np.sum(r['t_kalman']))
                    tt.append(np.sum(r['t_trans']))
                    te.append(np.sum(r['t_env']))
                    if r['success']:
                        success += 1
                        cs.append(np.sum(r['cost']))
                    if r['timeout']:
                        timeout += 1
                        ct.append(np.sum(r['cost']))
                    if r['failure']:
                        failure += 1
                        cf.append(np.sum(r['cost']))

        if is_present:  # Show results only for combinations that were actually tested!
            timeout_prop.append(timeout / (timeout + success + failure))
            table.append([agent_type, observation_type, kalman_filter_type, transition_estimator,
                          estimate_pert, str(pert_mode), np.mean(t), np.mean(ta) / np.mean(t), np.mean(te) / np.mean(t),
                          np.mean(tt) / np.mean(t), np.mean(tk) / np.mean(t), np.mean(c) / 1e3, success, timeout, failure,
                          np.mean(cs) / 1e3, np.mean(ct) / 1e3, np.mean(cf) / 1e3])
        c = 0
    print(tabulate(table,headers='firstrow'))
    print('Timeout average prop: ', np.mean(timeout_prop))

    print('\n\n[RESULTS] Disturbance estimator performance (full) \n \n')

    table2 = []
    table2.append(['Obs', 'State', 'Disturbance', 'Disturbance estimated', 'p error', 'Error gain', 'Time gain', 'Steps Gain'])

    for agent_type, observation_type, kalman_filter_type, pert_mode in product(agents, observations, kalmans, pert_modes):
        pert_estimated = np.zeros((len(pert_modes), ))  # Counter for each estimated perturbation
        param_error = []
        model_error_no_dist_init = []
        model_error_dist_init = []
        model_error_no_dist_end = []
        model_error_dist_end = []
        time_no_dist = []
        time_dist = []
        l_dist = []
        l_no_dist = []
        for r in res:
            if r is not None:
                if r['agent'] == agent_type and r['observation'] == observation_type and r['kalman'] == kalman_filter_type and r['pert']['pert_mode'] == pert_mode:
                    if r['estimate_pert'] == 'full':  # Take into account only good estimators!
                        if r['est_pert'][-1] is not None:  # Observe the last estimation and extract results from here
                            est_pert_index = pert_modes.index(r['est_pert'][-1])
                            pert_estimated[est_pert_index] += 1
                            if pert_modes[est_pert_index] == r['real_pert']:
                                real_params = np.squeeze(r['real_pert_params'])
                                if r['real_pert'] == 'const':
                                    real_params = real_params[0: 2]  # This perturbation has only two parameters!
                                    alt_params = real_params.copy()
                                    alt_params[0] *= -1
                                    alt_params[1] = (alt_params[1] + np.pi) % (2 * np.pi)  # Alt_params: because there are two solutions here with different phase
                                    error_real = 100 * np.mean(np.abs(r['est_pert_params'][-1] - real_params)) / np.mean(np.abs(real_params))
                                    error_alt = 100 * np.mean(np.abs(r['est_pert_params'][-1] - alt_params)) / np.mean(np.abs(alt_params))
                                    param_error.append(min(error_real, error_alt))
                                else:
                                    param_error.append(100 * np.mean(np.abs(r['est_pert_params'][-1] - real_params)) / np.mean(np.abs(real_params)))
                            if r['real_pert'] is None:
                                param_error.append(np.abs(r['est_pert_params'][-1][0]))  # Keep only the "strenght" parameter!
                        model_error_dist_init.append(r['est_norm'][0])
                        model_error_dist_end.append(r['est_norm'][-1])
                        time_dist.append(r['total_time'])
                        l_dist.append(len(r['cost']))
                    elif r['estimate_pert'] == 'none' and r['transition_estimator']:
                        model_error_no_dist_init.append(r['est_norm'][0])
                        model_error_no_dist_end.append(r['est_norm'][-1])
                        time_no_dist.append(r['total_time'])
                        l_no_dist.append(len(r['cost']))
        if np.sum(pert_estimated) > 0:
            table2.append(['T' if observation_type == 'total' else 'P',
                           'N' if kalman_filter_type == 'none' else 'KF',
                           str(pert_mode),
                           np.round(pert_estimated[1:] / np.sum(pert_estimated[1:]), 2), np.mean(param_error),
                           100 * (np.mean(model_error_no_dist_end) - np.mean(model_error_dist_end)) / np.mean(model_error_no_dist_end),
                           100 * (np.mean(time_no_dist) - np.mean(time_dist)) / np.mean(time_no_dist),
                           100 * (np.mean(l_no_dist) - np.mean(l_dist)) / np.mean(l_no_dist)])

    print(tabulate(table2, headers='firstrow', tablefmt='latex_raw', floatfmt=".2f"))

    print('\n\n[RESULTS] Disturbance estimator performance (const) \n \n')

    table2 = []
    table2.append(['Obs', 'State', 'Disturbance', 'Error gain', 'Time gain', 'Steps Gain'])

    for agent_type, observation_type, kalman_filter_type, pert_mode in product(agents, observations, kalmans, pert_modes):
        pert_estimated = np.zeros((len(pert_modes), ))  # Counter for each estimated perturbation
        param_error = []
        model_error_no_dist_init = []
        model_error_dist_init = []
        model_error_no_dist_end = []
        model_error_dist_end = []
        time_no_dist = []
        time_dist = []
        l_dist = []
        l_no_dist = []
        for r in res:
            if r is not None:
                if r['agent'] == agent_type and r['observation'] == observation_type and r['kalman'] == kalman_filter_type and r['pert']['pert_mode'] == pert_mode:
                    if r['estimate_pert'] == 'const':  # Take into account only good estimators!
                        if r['est_pert'][-1] is not None:  # Observe the last estimation and extract results from here
                            est_pert_index = pert_modes.index(r['est_pert'][-1])
                            pert_estimated[est_pert_index] += 1
                            if pert_modes[est_pert_index] == r['real_pert']:
                                real_params = np.squeeze(r['real_pert_params'])
                                if r['real_pert'] == 'const':
                                    real_params = real_params[0: 2]  # This perturbation has only two parameters!
                                    alt_params = real_params.copy()
                                    alt_params[0] *= -1
                                    alt_params[1] = (alt_params[1] + np.pi) % (2 * np.pi)  # Alt_params: because there are two solutions here with different phase
                                    error_real = 100 * np.mean(np.abs(r['est_pert_params'][-1] - real_params)) / np.mean(np.abs(real_params))
                                    error_alt = 100 * np.mean(np.abs(r['est_pert_params'][-1] - alt_params)) / np.mean(np.abs(alt_params))
                                    param_error.append(min(error_real, error_alt))
                                else:
                                    param_error.append(100 * np.mean(np.abs(r['est_pert_params'][-1] - real_params)) / np.mean(np.abs(real_params)))
                            if r['real_pert'] is None:
                                param_error.append(np.abs(r['est_pert_params'][-1][0]))  # Keep only the "strenght" parameter!
                        model_error_dist_init.append(r['est_norm'][0])
                        model_error_dist_end.append(r['est_norm'][-1])
                        time_dist.append(r['total_time'])
                        l_dist.append(len(r['cost']))
                    elif r['estimate_pert'] == 'none' and r['transition_estimator']:
                        model_error_no_dist_init.append(r['est_norm'][0])
                        model_error_no_dist_end.append(r['est_norm'][-1])
                        time_no_dist.append(r['total_time'])
                        l_no_dist.append(len(r['cost']))
        if np.sum(pert_estimated) > 0:
            table2.append(['T' if observation_type == 'total' else 'P',
                           'N' if kalman_filter_type == 'none' else 'KF',
                           str(pert_mode),
                           100 * (np.mean(model_error_no_dist_end) - np.mean(model_error_dist_end)) / np.mean(model_error_no_dist_end),
                           100 * (np.mean(time_no_dist) - np.mean(time_dist)) / np.mean(time_no_dist),
                           100 * (np.mean(l_no_dist) - np.mean(l_dist)) / np.mean(l_no_dist)])

    print(tabulate(table2, headers='firstrow', tablefmt='latex_raw', floatfmt=".2f"))


    print('\n\n[RESULTS] Advantage of knowing the model vs using a transition estimator / transition + pert estimator in successful trajs\n \n')

    table3 = []
    #table3.append(['Agent', 'Obs', 'Kalman', 'Pert', 'Cost known', 'Cost TE', 'Cost TE+PC',
    #               'Cost TE+PF', 'Gain TE / Known', 'Gain TE+PC / Known', 'Gain TE + PF / Known', 'Gain TE+PC / TE', 'Gain TE + PF / TE'])
    table3.append(['Obs', 'State', 'Disturbance', 'Gain TE / Known', 'Gain TEC / Known', 'Gain TEF / Known', 'Gain TEC / TE', 'Gain TEF / TE'])


    for agent_type, observation_type, kalman_filter_type, pert_mode in product(agents, observations, kalmans, pert_modes):
        cost_model_known = []
        cost_tran_est = []
        cost_tran_pert_est_const = []
        cost_tran_pert_est_full = []
        for r in res:
            if r is not None:
                if r['agent'] == agent_type and r['observation'] == observation_type and r['kalman'] == kalman_filter_type and r['pert']['pert_mode'] == pert_mode: #and r['success']:
                    if not r['transition_estimator']:
                        cost_model_known.append(np.sum(r['cost']))
                    elif r['estimate_pert'] == 'const':
                        cost_tran_pert_est_const.append(np.sum(r['cost']))
                    elif r['estimate_pert'] == 'full':
                        cost_tran_pert_est_full.append(np.sum(r['cost']))
                    else:
                        cost_tran_est.append(np.sum(r['cost']))
        if not np.isnan(np.mean(cost_model_known)):
            cost_model_known = np.array(cost_model_known)
            cost_tran_est = np.array(cost_tran_est)
            cost_tran_pert_est_const = np.array(cost_tran_pert_est_const)
            cost_tran_pert_est_full = np.array(cost_tran_pert_est_full)
            '''
            table3.append([agent_type, observation_type, kalman_filter_type, str(pert_mode),
                           np.mean(cost_model_known) / 1e3, np.mean(cost_tran_est) / 1e3,
                           np.mean(cost_tran_pert_est_const) / 1e3, np.mean(cost_tran_pert_est_full) / 1e3,
                           100 * (np.mean(cost_model_known) - np.mean(cost_tran_est)) / np.mean(cost_model_known),
                           100 * (np.mean(cost_model_known) - np.mean(cost_tran_pert_est_const)) / np.mean(cost_model_known),
                           100 * (np.mean(cost_model_known) - np.mean(cost_tran_pert_est_full)) / np.mean(cost_model_known),
                           100 * (np.mean(cost_tran_est) - np.mean(cost_tran_pert_est_const)) / np.mean(cost_tran_est),
                           100 * (np.mean(cost_tran_est) - np.mean(cost_tran_pert_est_full)) / np.mean(cost_tran_est)])'''
            table3.append(['T' if observation_type == 'total' else 'P',
                           'N' if kalman_filter_type == 'none' else 'KF',
                           str(pert_mode),
                           100 * (np.mean(cost_model_known) - np.mean(cost_tran_est)) / np.mean(cost_model_known),
                           100 * (np.mean(cost_model_known) - np.mean(cost_tran_pert_est_const)) / np.mean(
                               cost_model_known),
                           100 * (np.mean(cost_model_known) - np.mean(cost_tran_pert_est_full)) / np.mean(
                               cost_model_known),
                           100 * (np.mean(cost_tran_est) - np.mean(cost_tran_pert_est_const)) / np.mean(cost_tran_est),
                           100 * (np.mean(cost_tran_est) - np.mean(cost_tran_pert_est_full)) / np.mean(cost_tran_est)])

    print(tabulate(table3, headers='firstrow', tablefmt='latex_raw', floatfmt=".2f"))

    #Plot sample trajectories
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.family'] = 'Calibri'
    agent_type = 'mpc'
    observation_type = 'partial'
    kalman_filter_type = 'linear'
    for i, pert_mode in enumerate(['swirl', 'current_h', 'current_v', 'const']):
        for transition_estimator in [True]:
            for estimate_pert in ['const', 'full']:
                for r in res:
                    if r is not None:
                        if r['kalman'] == kalman_filter_type and r['agent'] == agent_type and r[
                            'observation'] == observation_type and r['transition_estimator'] == transition_estimator and \
                                r[
                                    'estimate_pert'] == estimate_pert and r['pert']['pert_mode'] == pert_mode and \
                                r['pert']['i'] == 0:
                            tm = min((1000, len(r['actions'])))
                            if estimate_pert == 'full':
                                plt.plot(np.squeeze(np.array(r['states']))[:tm, 0],
                                         np.squeeze(np.array(r['states']))[:tm, 1], c='b', linewidth=4, linestyle='--',
                                         label=str(transition_estimator) + str(estimate_pert))
                            else:
                                plt.plot(np.squeeze(np.array(r['states']))[:tm, 0],
                                         np.squeeze(np.array(r['states']))[:tm, 1], c='r', linewidth=4,
                                         label=str(transition_estimator) + str(estimate_pert))
                            plt.plot(np.squeeze(np.array(r['states']))[0, 0], np.squeeze(np.array(r['states']))[0, 1],
                                     'og')  # Initial point
                            env = UuvPertSingle(partial_obs=observation_type == 'partial',
                                                pert_mode=r['pert']['pert_mode'],
                                                pert_params=r['pert'][
                                                    'pert_params'])  # Initialize the right en, but call to quiver only once!
        plt.plot(0, 0, '*g')  # Target
        _ = env.reset()
        x = y = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
        X, Y = np.meshgrid(x, y)
        px, py = env.actual_pert_model.perturbation(np.ravel(X), np.ravel(Y))
        plt.quiver(np.ravel(X), np.ravel(Y), px.reshape(X.shape) + np.finfo(float).eps,
                   py.reshape(Y.shape) + np.finfo(float).eps)
        #plt.legend(loc='best')
        #plt.title(str(pert_mode))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(str(pert_mode) + '.pdf', dpi=300)
        plt.show()

    print('DONE')

    '''
    # This code can be used for plotting a trajectory
    
    def plot_res(res, tm=1000, env=None, npp=20):
        tm = min((tm, len(res['actions'])))
        plt.plot(np.squeeze(np.array(res['states']))[:tm, 0], np.squeeze(np.array(res['states']))[:tm, 1], 'b',
                 label='states')
        plt.plot(np.squeeze(np.array(res['obs']))[:tm, 0], np.squeeze(np.array(res['obs']))[:tm, 1], 'r', label='obs')
        plt.plot(np.squeeze(np.array(res['states']))[0, 0], np.squeeze(np.array(res['states']))[0, 1],
                 'ob')  # Initial point
        plt.quiver(np.squeeze(np.array(res['states']))[:tm, 0], np.squeeze(np.array(res['states']))[:tm, 1],
                   np.squeeze(np.array(res['actions']))[:tm, 0], np.squeeze(np.array(res['actions']))[:tm, 1])
        plt.plot(0, 0, 'ok')  # Target
        if env is not None:
            x = y = np.linspace(env.observation_space.low[0], env.observation_space.high[0], npp)
            X, Y = np.meshgrid(x, y)
            px, py = env.actual_pert_model.perturbation(np.ravel(X), np.ravel(Y))
            plt.quiver(np.ravel(X), np.ravel(Y), px.reshape(X.shape) + np.finfo(float).eps,
                       py.reshape(Y.shape) + np.finfo(float).eps, color='g')
        plt.legend(loc='best')
        plt.show()

    
    for r in res:
        if r is not None and len(r['states']) > 0:
            if r['kalman'] == 'linear' and r['agent'] == 'mpc' and r['observation'] == 'partial' and r['transition_estimator'] == True and r['pert']['pert_mode'] == None:
                if np.linalg.norm(r['states'][0] - pert_values[5]['init_state']) < 1e-3:
                    tm = min((1000, len(r['actions'])))
                    if r['estimate_pert']:
                        plt.plot(np.squeeze(np.array(r['states']))[:tm, 0], np.squeeze(np.array(r['states']))[:tm, 1], 'b:', label='states')
                        plt.plot(np.squeeze(np.array(r['obs']))[:tm, 0], np.squeeze(np.array(r['obs']))[:tm, 1], 'r:', label='obs')
                        plt.plot(np.squeeze(np.array(r['states']))[0, 0], np.squeeze(np.array(r['states']))[0, 1], 'ob')  # Initial point
                        #plt.quiver(np.squeeze(np.array(r['states']))[:tm, 0], np.squeeze(np.array(r['states']))[:tm, 1], np.squeeze(np.array(r['actions']))[:tm, 0], np.squeeze(np.array(r['actions']))[:tm, 1])
                        plt.plot(0, 0, 'ok')  # Target
                    else:
                        plt.plot(np.squeeze(np.array(r['states']))[:tm, 0], np.squeeze(np.array(r['states']))[:tm, 1], 'b', label='states')
                        plt.plot(np.squeeze(np.array(r['obs']))[:tm, 0], np.squeeze(np.array(r['obs']))[:tm, 1], 'r', label='obs')
                        plt.plot(np.squeeze(np.array(r['states']))[0, 0], np.squeeze(np.array(r['states']))[0, 1], 'ob')  # Initial point
                        #plt.quiver(np.squeeze(np.array(r['states']))[:tm, 0], np.squeeze(np.array(r['states']))[:tm, 1], np.squeeze(np.array(r['actions']))[:tm, 0], np.squeeze(np.array(r['actions']))[:tm, 1])
                        plt.plot(0, 0, 'ok')  # Target
                        env = UuvPertSingle(partial_obs=False, pert_mode=r['pert']['pert_mode'],
                                            pert_params=r['pert']['pert_params'])

    if env.actual_pert_model is not None:
        x = y = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
        X, Y = np.meshgrid(x, y)
        px, py = env.actual_pert_model.perturbation(np.ravel(X), np.ravel(Y))
        plt.quiver(np.ravel(X), np.ravel(Y), px.reshape(X.shape) + np.finfo(float).eps, py.reshape(Y.shape) + np.finfo(float).eps, color='g')
    plt.show()
    '''
    print('Test ended')






