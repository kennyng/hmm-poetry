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
        """Transform observation data to integers corresponding to tokens.
        
        """
        self.num_obs = 0
        token_to_index = {}
        X = []

        for sequence in data:
            X_i = []
            for token in sequence:
                if token not in token_to_index:
                    token_to_index[token] = self.num_obs
                    self.num_obs += 1
                X_i.append(token_to_index[token])
            X.append(X_i)

        self.token_dict = {v: k for k, v in token_to_index.items()}
        return X

    @staticmethod
    def _normalize(matrix):
        """ Constrain all rows of matrix to sum to 1."""
        if len(matrix.shape) == 1:
            return matrix / matrix.sum()
        sums = matrix.sum(axis=1)
        return matrix / sums.reshape(sums.shape[0], 1)

    def _compute_marginals(self, X, scaling=True):
        # Calculate alphas and betas for all sequences.
        alphas = []
        betas = []
        for sequence in X:
            alpha, beta = self._forward_backward_algorithm(sequence, scaling)
            alphas.append(alpha)
            betas.append(beta)

        # Compute gammas for all sequences.
        # P(Y_i = z)
        # Indexed by: sequence index, position, state
        gammas = []
        for i in range(len(X)):
            seq_len = len(X[i])
            alpha = alphas[i]
            beta = betas[i]

            # Gamma for this sequence.
            gamma = np.zeros((seq_len, self.num_states))
            for j in range(seq_len):
                for state in range(self.num_states):
                    gamma[j, state] = alpha[j, state] + beta[j, state]
                gamma[j] = gamma[j] / gamma[j].sum()
            gammas.append(gamma)

        # Compute xis for all sequences.
        # P(Y_i = prev, Y_i+1 = next)
        # Indexed by: sequence index, prev position, prev state, next state
        xis = []
        for i in range(len(X)):
            seq = X[i]
            seq_len = len(seq)
            alpha = alphas[i]
            beta = betas[j]

            # Xi for this sequence.
            xi = np.zeros((seq_len-1, self.num_states, self.num_states))
            for j in range(seq_len-1):
                for prev_state in range(self.num_states):
                    for next_state in range(self.num_states):
                        xi[j, prev_state, next_state] = (
                            alpha[j, prev_state] *
                            self.E[next_state, seq[j+1]] *
                            self.P[prev_state, next_state] *
                            beta[j+1, next_state]
                        )
                xi[j] = xi[j] / xi[j].sum()
            xis.append(xi)

    def _forward_backward_algorithm(self, sequence, scaling):
        seq_len = len(sequence)
        alphas =    np.zeros((seq_len, self.num_states))
        betas = np.zeros((seq_len, self.num_states))

        # FORWARD
        for i in range(seq_len):
            for state in range(self.num_states):
                # Base Case
                if i == 0:
                        alphas[i, state] = (self.E[state, sequence[0]] *
                                            self.P_init[state])
                else:
                    prob_sum = 0
                    for prev_state in range(self.num_states):
                        prob_sum += (alphas[i-1, prev_state] *
                                     self.P[prev_state, state])
                    alphas[i, state] = prob_sum * self.E[state, sequence[i]]

            # Scaling
            if scaling:
                scale = np.sum(alphas[i])
                alphas[i] = alphas[i] / scale

        # BACKWARD
        for i in reversed(range(seq_len)):
            for state in range(self.num_states):
                # Base Case
                if  i == (seq_len - 1):
                    betas[i, state] = 1
                else:
                    for next_state in range(self.num_states):
                        betas[i, state] += (betas[i+1, next_state] *
                                            self.P[state, next_state] *
                                            self.E[next_state, sequence[i+1]])

            # Scaling
            if scaling:
                scale = np.sum(betas[i])
                betas[i] = betas[i] / scale

        return (alphas, betas)
 

    def _update(self, X, gammas, xis):
        pass
