import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints


class KalmanFilter(object):
    def __init__(self, type, state_size, action_size, F, H, P, Q, R, B=None, Hjac=None, Fjac=None, dt=None):

        self.type = type  # Either "linear", "ekf" or "ukf"

        self.state_size = state_size  # Assumes obs and state has same dim!!
        self.action_size = action_size

        if self.type == 'linear':
            assert isinstance(F, np.ndarray) and isinstance(B, np.ndarray)
        if self.type == 'ekf':
            assert isinstance(Fjac, np.ndarray) and isinstance(Hjac, np.ndarray)
        if self.type == 'ukf':
            raise RuntimeError('Current UKF implementation does not allow incorpore the action effects!')

        if type == 'linear':  # Transition and measurement functions are linear!
            self.Fjac = F
            self.Hjac = H

            # Prepare matrices for linear functions
            self.A_linear = F
            self.B_linear = B
            self.H = H

            self.Freal = self.F_linear
            self.Hreal = self.H_linear
        else:
            self.Freal = F
            self.Hreal = H
            self.Fjac = Fjac  # Jacobian of the transition function
            self.Hjac = Hjac  # Jacobian of the measurement function
        self.P = P  # Covariance matrix of the state
        self.Q = Q  # Process noise matrix
        self.R = R  # Covariance matrix of measurements

        # UKF initialization
        self.ukf = None
        self.ukf_init = None  # Flag to initialize ukf filter
        if type == 'ukf':
            sigmas = MerweScaledSigmaPoints(self.state_size + self.action_size, alpha=.3, beta=2., kappa=1.)
            self.ukf = UKF(dim_x=self.state_size + self.action_size, dim_z=self.state_size, fx=self.Freal, hx=self.Hreal, dt=dt, points=sigmas)
            self.ukf.Q = self.Q
            self.ukf.R = self.R

        self.preds = []  # To store the predictions used

    def set_linear_matrices(self, A, B):
        self.A_linear = A
        self.B_linear = B

    def F_linear(self, state, action):
        return self.A_linear @ state + self.B_linear @ action

    def H_linear(self, state):  # Simple measurement function
        return self.H @ state

    def reset(self, obs):
        obs = np.reshape(obs, (self.state_size, 1))
        self.preds.append(obs)  # Initial value
        self.ukf_init = False


    def estimate(self, obs, ac, Fjac=None):
        obs = np.reshape(obs, (self.state_size, 1))
        ac = np.reshape(ac, (self.action_size, 1))

        if Fjac is not None:
            self.Fjac = Fjac  # Update the Fjac using the value provided as input

        if self.type == 'linear' or self.type == 'ekf':
            x = self.Freal(self.preds[-1], ac)  # Predicted next state
            x = np.reshape(x, (self.state_size, 1))
            self.P = self.Fjac @ self.P @ self.Fjac.T + self.Q
            # Update
            S = self.Hjac @ self.P @ self.Hjac.T + self.R
            K = self.P @ self.Hjac.T @ np.linalg.inv(S)
            y = obs - self.Hreal(x)
            self.P = self.P - K @ self.Hjac @ self.P
            self.preds.append(x + (K @ y))  # Output value

        elif self.type == 'ukf':  # Use the UKF
            if not self.ukf_init:
                self.ukf.x = np.squeeze(np.vstack((obs, ac))) # Initial state-action pair
            self.ukf.predict()
            self.ukf.update(obs.reshape(self.state_size))
            self.preds.append((self.ukf.x).reshape(1, self.state_size))
        else:
            raise RuntimeError('Kalman Type ' + str(self.type) + ' not recognized')

        return self.preds[-1].reshape(1, self.state_size)
