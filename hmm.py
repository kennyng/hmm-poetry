import numpy as np


class HiddenMarkovModel:

    def __init__(self, num_states):
        self.num_obs = 0
        self.num_states = num_states
        self.token_dict = {}
        self.P_init = None  # Initial state distribution.
        self.P = None  # Transiton matrix (row: FROM, col: TO).
        self.E = None  # Emission matrix (row: STATE, col: OBSERVATION).

    def train(self, data, epsilon=0.001, scaling=True):
        X = self._transform_observations(data)
        # Initialize matrices
        self.P_init = self._normalize(np.random.rand(num_states))
        self.P = self._normalize(np.random.rand(num_states, num_states))
        self.E = self._normalize(np.random.rand(num_states, num_obs))

        norms = []
        iterations = 0
        while True:
            iterations += 1
            # E Step
            gammas, xis = self._compute_marginals(X, scaling)

            # M Step
            updated_norm = self._update(X, gammas, xis)
            print(updated_norm)
            norms.append(updated_norm)

            # Stopping condition.
            if len(norms) > 1 and norms[-1] / norms[0] < epsilon:
                print('Iterations: {}'.format(iterations))
                break

        print(self.P_init)
        print(self.P)
        print(self.E)
        return (self.token_dict, self.P_init, self.P, self.E)

    def _transform_observations(self, data):
        pass

    @staticmethod
    def _normalize(matrix):
        """ Constrain all rows of matrix to sum to 1."""
        if len(matrix.shape) == 1:
            return matrix / matrix.sum()
        sums = matrix.sum(axis=1)
        return matrix / sums.reshape(sums.shape[0], 1)

    def _compute_marginals(self, X, scaling=True):
        pass

    def _forward_backward_algorithm(self, sequence, scaling):
       pass

    def _update(self, X, gammas, xis):
       pass 
