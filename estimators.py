import numpy as np
from scipy.optimize import minimize


class LS_Estimator(object):
    def __init__(self, state_dim, ac_dim, intercept=False, clip=0.01):
        self.state_dim = state_dim
        self.ac_dim = ac_dim
        self.intercept = intercept
        self.clip = clip

    def estimate(self, Xtrain, ytrain, n=1000):
        n = min([Xtrain.shape[0], n])  # Number of elements to compute the estimation
        n_reg = ytrain.shape[1]  # Number of LS problems to solve
        P = np.zeros((n_reg, Xtrain.shape[1] + int(self.intercept)))  # Solution matrix
        for i in range(n_reg):  # Solve the LS problems
            y = ytrain[-n:, i].reshape([n, 1])
            if self.intercept:
                A = np.hstack((Xtrain[-n:, :], np.ones((n, 1))))
            else:
                A = Xtrain[-n:, :]

            min_eig = np.amin(np.linalg.eig(A.T @ A)[0])  # Minimal eigenvalue
            if min_eig <= 1e-3:  # Check if matrix A.T @ A is singular
                val = min_eig * np.sign(min_eig) + 2e-3  # Small value to make the matrix non-singular
                #print('LS matrix singular (eig : ', min_eig, ' ): adding ', val, ' to main diagonal')
                x = np.linalg.inv(A.T @ A + np.eye(A.shape[1]) * val) @ A.T @ y
            else:
                x = np.linalg.inv(A.T @ A) @ A.T @ y  # LS solution (note that the complexity of matrix inversion depends on n!)
            P[i] = np.squeeze(x)
        A = P[:, 0: self.state_dim]
        B = P[:, self.state_dim: self.state_dim + self.ac_dim]
        if self.intercept:
            C = P[:, -1].reshape([1, self.state_dim])
        else:
            C = np.zeros((1, self.state_dim))

        if self.clip is not None:  # Set very small values to 0.0 to alleviate noise (if desired)
            A[np.abs(A) < self.clip] = 0.0
            B[np.abs(B) < self.clip] = 0.0
            C[np.abs(C) < self.clip] = 0.0

        if np.amax(np.abs(A)) > 50:
            print('A large ', np.amax(np.abs(A)))
        if np.amax(np.abs(B)) > 50:
            print('B large ', np.amax(np.abs(B)))
        A = np.clip(A, a_max=50, a_min=-50)
        B = np.clip(B, a_max=50, a_min=-50)
        C = np.clip(C, a_max=50, a_min=-50)
        return A, B, C


# Perturbation definitions
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


class PerturbationRegressor:
    def __init__(self, mode, bound_val=20):
        self.mode = mode
        assert self.mode == 'const' or self.mode == 'full'
        self.bounds = [(-bound_val, bound_val), (-bound_val, bound_val), (-bound_val, bound_val)]  # Used in the optimizer to bound the solution!

    def estimate_params_and_pert(self, x, y, d1, d2, init_params, method='bfgs'):
        if self.mode == 'full':  # Estimate using all models
            perts = ['const', 'current_v', 'current_h', 'swirl']
        elif self.mode == 'const':  # Use only the const model
            perts = ['const']
        mse = [None for _ in perts]
        params = [None for _ in perts]
        for i, pert in enumerate(perts):
            params[i] = self.estimate_params(x, y, d1, d2, pert, init_params, method)
            mse[i] = self.get_mse(params[i], x, y, d1, d2, pert)
        #best_model = np.argmin(np.array(mse))
        if (np.amax(np.array(mse)) - np.amin(np.array(mse))) / np.amin(np.array(mse)) > 1e-3:  # The difference between MSE is large enough
            best_model = np.argmin(np.array(mse))
        else:
            #best_model = np.argmin(np.array([p[0] for p in params]))  # Choose the model with lower perturbation module
            best_model = 0  # Select the constant perturbation model, as it is the simplest
        return perts[best_model], params[best_model]

    def estimate_params(self, x, y, d1, d2, pert, init_params, method):
        if pert == 'const':
            return self.values_const(x, y, d1, d2)
        elif pert == 'current_h' or pert == 'current_v' or pert == 'swirl':
            if method == 'bfgs':
                return self.bfgs_estimate(x, y, d1, d2, pert, init_params)
            else:  # Use Nesterov in any other case
                return self.gradient_estimate(x, y, d1, d2, pert, init_params)
        else:
            raise RuntimeError('Perturbation mode not recognized: ' + str(pert))

    def get_mse(self, params, x, y, d1, d2, pert):
        if pert == 'swirl':
            px, py = perturbation_swirl(x, y, params)
        elif pert == 'current_h':
            px, py = perturbation_current_h(x, y, params)
        elif pert == 'current_v':
            px, py = perturbation_current_v(x, y, params)
        elif pert == 'const':
            px, py = perturbation_const(x, y, params)
        return np.mean(np.square(d1 - px) + np.square(d2 - py))

    def values_const(self, x, y, d1, d2):
        theta = np.arctan2(np.sum(d2), np.sum(d1)) % (2 * np.pi)  # Wrap the angle to be in the [0, 2\pi) range
        a = (np.cos(theta) * np.sum(d1) + np.sin(theta) * np.sum(d2)) / len(d1)
        return a, theta

    def bfgs_estimate(self, x, y, d1, d2, pert, init_params):
        #sol = fmin_bfgs(self.get_mse, init_params, fprime=self.grad, args=(x, y, d1, d2, pert), gtol=1e-8, disp=False)  # Set disp=True if the convergence summary wants to be seen (eliminated to avoid output noise)
        # Use L-BFGS-B to be able to bound the solution to valid ranges
        sol2 = minimize(self.get_mse, x0=init_params, jac=self.grad, args=(x, y, d1, d2, pert), method='L-BFGS-B', options={'gtol': 1e-10}, bounds=self.bounds)
        return sol2.x

    def gradient_estimate(self, x, y, d1, d2, pert, init_params):  # NOTE: DEPRECATED METHOD, USE NOT TESTED!!
        params_y = init_params.copy()
        params_x = params_y.copy()
        step = 1e-4
        threshold = 1e-6
        max_iter = int(10000)  # Number of iterations before considering divergence
        for i in range(max_iter):
            prev_vals = params_y.copy()
            params_x_prev = params_x.copy()
            params_x = params_y - step * self.grad(params_y, x, y, d1, d2, pert)
            params_y = params_x + (i - 2) / (i + 1) * (params_x - params_x_prev)  # Add momentum for faster convergence
            if np.sqrt(np.sum(np.square(prev_vals - params_y))) < threshold:
                #print('Convergence achieved with ', i, ' iterations')
                break
        #if i == max_iter - 1:
            #print('Divergence')
        return params_y


    def grad(self, params, x, y, d1, d2, pert):  # Gradient for each perturbation
        if pert == 'current_v':
            c = params[0]
            x0 = params[1]
            w = params[2]
            k = np.exp(-np.square(x - x0) / (w ** 2))
            grad_c = np.sum(2 * (c * k - d2) * k)
            grad_x0 = np.sum(4 * (c * k - d2) * c * k * (x - x0) / (w ** 2))
            grad_w = np.sum(4 * (c * k - d2) * c * k * np.square(x - x0) / (w ** 3))
            return np.array([grad_c, grad_x0, grad_w]) # Gradient vector!
        elif pert == 'current_h':
            c = params[0]
            y0 = params[1]
            w = params[2]
            k = np.exp(-np.square(y - y0) / (w ** 2))
            grad_c = np.sum(2 * (c * k - d1) * k)
            grad_y0 = np.sum(4 * (c * k - d1) * c * k * (y - y0) / (w ** 2))
            grad_w = np.sum(4 * (c * k - d1) * c * k * np.square(y - y0) / (w ** 3))
            return np.array([grad_c, grad_y0, grad_w]) # Gradient vector!
        elif pert == 'swirl':
            b = params[0]
            x0 = params[1]
            y0 = params[2]
            r = np.sqrt(np.square(x - x0) + np.square(y - y0))
            grad_b = np.sum(2 * (b * (y - y0) / r - d1) * (y - y0) / r +
                            2 * (b * (x0 - x) / r - d2) * (x0 - x) / r)
            grad_x0 = np.sum(2 * (b * (y - y0) / r - d1) * b * (y - y0) * (x - x0) / np.power(r, 3) +
                             2 * (b * (x0 - x) / r - d2) * b * (r - (x0 - x) * (x - x0) / r) / np.square(r))
            grad_y0 = np.sum(2 * (b * (y - y0) / r - d1) * b * (- r - (y - y0) * (y - y0) / r) / np.square(r) +
                             2 * (b * (x0 - x) / r - d2) * (b * (x0 - x) * (y - y0) / np.power(r, 3)))
            return np.array([grad_b, grad_x0, grad_y0]) # Gradient vector!
        else:
            raise RuntimeError('Perturbation gradient not implemented for perturbation ' +  str(pert))


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
            raise RuntimeError('Perturbation mode not recognized: ' + str(self.mode))


class TransitionEstimator(object):
    def __init__(self, state_size, action_size, warmup=10, estimate_transition=True, estimate_perturbation=False,
                 delta=None, k=None, est_method='bfgs', train_every=5):
        self.X = None
        self.y = None
        self.it = None
        self.train_every = train_every
        self.state_size = state_size
        self.action_size = action_size
        self.estimator = LS_Estimator(state_dim=state_size, ac_dim=action_size, intercept=False)
        self.warmup = warmup  # Number of iters of warmup (i.e., min number of samples before starting to estimate)
        self.estimate_transition = estimate_transition  # Set to True to estimate the transition model
        self.estimate_perturbation = estimate_perturbation  # Set to True to estimate the Perturbation model
        if self.estimate_perturbation != 'none':
            self.pert_reg = PerturbationRegressor(self.estimate_perturbation)
            self.delta = delta  # Time interval used
            self.k = k  # Friction parameter
            self.est_method = est_method  # Method for estimation

        self.A = None
        self.B = None
        self.C = None
        self.estimated_pert = None
        self.estimated_pert_model = None
        self.prev_pert_par = None

    def reset(self):
        self.X = []
        self.y = []
        self.it = 0
        self.A = None
        self.B = None
        self.C = None
        self.estimated_pert = {'est_pert': None, 'est_params': None}
        self.estimated_pert_model = None
        self.prev_pert_par = None

    def get_x_y(self):
        return np.squeeze(np.array(self.X)), np.squeeze(np.array(self.y))

    def get_perts(self):
        X, y = self.get_x_y()  # Obtain X and y stored as numpy arrays

        positions = X[:-1, 0: 2]  # x and y coordinates only!
        perts = (X[1:, 2: 4] - X[:-1, 2: 4]) / self.delta - X[: -1, 4: 6] + self.k * X[:-1, 2: 4]  # Perturbances estimated using the model!!
        # perts = ((X[1:, 0:4] - (self.A @ X[:-1, 0:4].T).T - (self.B @ X[:-1, 4:6].T).T) / self.delta)[:, 2:4]
        if self.warmup > 0:  # Careful here, as we reset the agent and we must eliminate a "false" perturbation that arises due to that
            perts = np.delete(perts, (self.warmup - 1), axis=0)
            positions = np.delete(positions, (self.warmup - 1), axis=0)
        return perts, positions

    def estimate(self, newX, newY):

        self.X.append(newX)
        self.y.append(newY)

        A = B = C = None

        if self.it >= self.warmup:  # Train only after warmup
            if (self.it - self.warmup) % self.train_every == 0:  # Train every train_every iteration

                X, y = self.get_x_y()  # Obtain X and y stored as numpy arrays

                if self.estimate_perturbation != 'none':

                    if self.A is not None:
                        perts, positions = self.get_perts()

                        if self.prev_pert_par is None:
                            init_params = np.ones((3, ))  # Fixed param start bor algorithm in case no previous params have been computed
                        else:
                            init_params = self.prev_pert_par  # This reuses previous computations to speed up the transition estimator

                        est_pert, est_params = self.pert_reg.estimate_params_and_pert(positions[:, 0], positions[:, 1],
                                                                                      perts[:, 0], perts[:, 1],
                                                                                      init_params=init_params,
                                                                                      method=self.est_method)
                        self.prev_pert_par = est_params  # For reusing them in other iterations!
                        if len(self.prev_pert_par) < 3:
                            self.prev_pert_par = np.append(self.prev_pert_par, 1)  # In case a constant disturbance is computed, it has a parameter less!
                        self.estimated_pert['est_pert'] = est_pert
                        self.estimated_pert['est_params'] = est_params
                        self.estimated_pert_model = PerturbationModel(est_pert, est_params)
                        px, py = self.estimated_pert_model.perturbation(X[:, 0], X[:, 1])  # Obtain perturbations using the estimated model
                        correction_term = np.zeros((px.shape[0], 4))
                        correction_term[:, 2] = px * self.delta
                        correction_term[:, 3] = py * self.delta
                        y = y - correction_term  # y corrected using the perturbation information!

                if self.estimate_transition:  # EStimate the transition model
                    A, B, C = self.estimator.estimate(X, y)

                self.A = A
                self.B = B
                self.C = C

        self.it += 1
        return self.A, self.B, self.C