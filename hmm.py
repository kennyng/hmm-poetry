##############################################################################
# hmm.py
# ------
# General Hidden Markov Model implementation.
# Trains HMM using the standard Baum-Welch (EM) algorithm.
#
##############################################################################

import numpy as np


class HiddenMarkovModel:
    """Hidden Markov Model implementation.

    Attributes
    ----------
    num_obs : int
        Count of unique observations.
    num_states: int
        Number of states in the model.
    token_dict : dict
        Mapping from tokens to observation indices (int).
    PI: numpy.array
        Initial state distribution vector.
    A: numpy.array
        Transition probabilities matrix [row: FROM, col: TO].
    B: numpy.array
        Emission probabilities (observation likelihoods) matrix
        [row: STATE, col: OBSERVATION].

    """

    def __init__(self, num_states):
        self.num_obs = 0
        self.num_states = num_states
        self.token_dict = {}
        self.PI = None
        self.A = None
        self.B = None

    def train(self, data, epsilon=0.01, max_iter=100, scale=True):
        """Train HMM using the standard Baum-Welch (EM) algorithm.

        The EM (expectation-maximization) algorithm is an iterative process
        that updates a set of HMM parameter estimates (A, B) until convergence.

        Parameters
        ----------
        data : list
            List of training sequences (observation data).
        epsilon : float
            Value for calculating the stopping condition for the algorithm
            to converge (when ratio between updated norm and initial norm
            is less than epsilon).
        max_iter : int
            Maximum number of iterations that the algorithm should run.
            (Set max_iter=0 for indefinite number of iterations.)
        scale : bool
            Indicates whether to normalize probability vectors.

        Returns
        -------
        A : numpy.array
            Transition probabilities matrix of trained model.
        B : numpy.array
            Emission probabilities matrix of trained model.
        PI : numpy.array
            Initial state distribution vector of trained model.
        token_dict : dict
            Mapping from tokens to observation indices (int).

        """
        X = self.transform_observations(data)

        # Initialize matrices randomly.
        self.PI = self.normalize(np.random.rand(self.num_states))
        self.A = self.normalize(np.random.rand(self.num_states, self.num_states))
        self.B = self.normalize(np.random.rand(self.num_states, self.num_obs))

        norms = []
        iterations = 0
        while True:
            iterations += 1
            # Expectation Step
            gammas, xis = self.compute_probabilities(X, scale)

            # Maximization Step
            updated_norm = self.update(X, gammas, xis)
            print(updated_norm)
            norms.append(updated_norm)

            # Stopping Conditions
            if (len(norms) > 1 and norms[-1] / norms[0] < epsilon or
                    iterations > max_iter):
                print('Number of Iterations: {}'.format(iterations))
                break

        return self.A, self.B, self.PI, self.token_dict

    @staticmethod
    def normalize(matrix):
        """ Constrain all rows of matrix to sum to 1."""
        if len(matrix.shape) == 1:
            return matrix / matrix.sum()
        sums = matrix.sum(axis=1)
        return matrix / sums.reshape(sums.shape[0], 1)

    def transform_observations(self, data):
        """Transform observation sequences to integers corresponding to tokens.

        Parameters
        ----------
        data : list
            List of training sequences (tokenized observation data).

        Returns
        =======
        X : list
            List of observation sequences (lists) where observations (tokens)
            are mapped to indices (int).

        """
        self.num_obs = 0
        self.token_dict = {}

        X = []
        for sequence in data:
            X_i = []
            for token in sequence:
                if token not in self.token_dict:
                    self.token_dict[token] = self.num_obs
                    self.num_obs += 1
                X_i.append(self.token_dict[token])
            X.append(X_i)

        return X

    def compute_probabilities(self, X, scale=True):
        """Estimate set of HMM parameters for observations aligned to each state.

        To do so, four main variables are calculated: Alpha, Beta, Gamma, Xi.

        Parameters
        ----------
        X : list
            List of observation sequences where observations (tokens) are
            mapped to integers.
        scale : bool
            Indicates whether to normalize probability vectors.

        Returns
        -------
        gammas : list
            Probability estimates of hidden state at each observation.
        xis : list
            State transition probabilty estimates at each observation.

        """
        alphas = []
        betas = []
        for sequence in X:
            alpha, beta = self._forward_backward_algorithm(sequence, scale)
            alphas.append(alpha)
            betas.append(beta)

        gammas = self._compute_gammas(X, alphas, betas)
        xis = self._compute_xis(X, alphas, betas)

        return gammas, xis

    def _forward_backward_algorithm(self, sequence, scale):
        """Computes posterior marginals (Alpha and Beta) of all hidden states.

        Given a sequence of observations (emissions) X, this inference task
        ("smoothing") computes for all hidden state variables Q, the
        distribution Pr(Q | X) using dynamic programming in two passes.
        Forward Procedure: forward probabilities (Alpha) are calculated with
            an initial estimate of the hidden state starting from the first
            data observation.
        Backward Procedure: backward probabilities (Beta) are calculated as
            conditional probabilities starting from the last data observation.

        Parameters
        ----------
        sequence : list
            List of observations (a single sequence).
        scale : bool
            Indicates whether to normalize probability vectors.

        Returns
        -------
        alpha : numpy.array
            Forward probabilities.
        beta : numpy.array
            Backward probabilities.

        """
        num_obs = len(sequence)
        alpha = np.zeros((num_obs, self.num_states))
        beta = np.zeros((num_obs, self.num_states))

        # FORWARD Procedure
        for i in range(num_obs):
            for state in range(self.num_states):
                if i == 0:
                    alpha[i, state] = (self.B[state, sequence[0]] *
                                       self.PI[state])
                else:
                    prob_sum = 0
                    for prev_state in range(self.num_states):
                        prob_sum += (alpha[i - 1, prev_state] *
                                     self.A[prev_state, state])
                    alpha[i, state] = prob_sum * self.B[state, sequence[i]]
            if scale:
                factor = np.sum(alpha[i])
                alpha[i] = alpha[i] / factor

        # BACKWARD Procedure
        for i in reversed(range(num_obs)):
            for state in range(self.num_states):
                if i == (num_obs - 1):
                    beta[i, state] = 1
                else:
                    for next_state in range(self.num_states):
                        beta[i, state] += (beta[i + 1, next_state] *
                                           self.A[state, next_state] *
                                           self.B[next_state, sequence[i + 1]])
            if scale:
                factor = np.sum(beta[i])
                beta[i] = beta[i] / factor

        return alpha, beta

    def _compute_gammas(self, X, alphas, betas):
        """Compute probability estimates of hidden state at each observation.

        Gamma(i) = Pr(Q_t = i | X) and is calculated using results
        (Alpha and Beta) from Forward-Backward algorithm.

        Parameters
        ----------
        X : list
            List of observation sequences where observations (tokens)
            are mapped to integers.
        alphas : list
            List of numpy.array containing forward probabilities for
            each observation sequence.
        betas: list
            List of numpy.array containing backward probabilities for
            each observation sequence.

        Returns
        -------
        gammas : list
            List of numpy.array containing the probability estimates of
            the hidden state at each observation.
            Gammas indexed by: sequence index, position, state.

        """
        gammas = []
        for j in range(len(X)):
            num_obs = len(X[j])
            alpha = alphas[j]
            beta = betas[j]
            gamma = np.zeros((num_obs, self.num_states))

            for i in range(num_obs):
                for state in range(self.num_states):
                    gamma[i, state] = alpha[i, state] + beta[i, state]
                gamma[i] = gamma[i] / gamma[i].sum()
            gammas.append(gamma)

        return gammas

    def _compute_xis(self, X, alphas, betas):
        """Compute state transition probabilty estimates at each observation.

        Xi(i, j) = Pr(Q_t = i, Q_t+1 = j | X) and is calculated using
        the results (Alpha and Beta) from the Forward-Backward algorithm.

        Parameters
        ----------
        X : list
            List of observation sequences where observations (tokens)
            are mapped to integers.
        alphas : list
            List of numpy.array containing forward probabilities for
            each observation sequence.
        betas: list
            List of numpy.array containing backward probabilities for
            each observation sequence.

        Returns
        -------
        xis : list
            List of numpy.array containing the state transition
            probability estimates at each observation.
            Xis indexed by: sequence index, previous position,
                            previous state, next state.

        """
        xis = []
        for j in range(len(X)):
            sequence = X[j]
            num_obs = len(sequence)
            alpha = alphas[j]
            beta = betas[j]
            xi = np.zeros((num_obs - 1, self.num_states, self.num_states))

            for i in range(num_obs - 1):
                for prev_state in range(self.num_states):
                    for next_state in range(self.num_states):
                        xi[i, prev_state, next_state] = (
                            alpha[i, prev_state] *
                            self.B[next_state, sequence[i + 1]] *
                            self.A[prev_state, next_state] *
                            beta[i + 1, next_state]
                        )
                xi[i] = xi[i] / xi[i].sum()
            xis.append(xi)

        return xis

    def update(self, X, gammas, xis):
        """Updates HMM parameters (A, B, PI) to maximize Pr(X | A,B,PI).

        Parameters
        ----------
        X : list
            List of observation sequences where observations (tokens)
            are mapped to integers.
        gammas : list
            List of numpy.array containting the probability estimates of
            the hidden state at each observation.
        xis : list
            List of numpy.array containting the state transition probability
            estimates at each observation.

        Returns
        -------
        norm : float
            Frobenius norm of the change between initial and updated matrices.

        """
        PI = self._update_init_probs(X, gammas)
        A = self._update_transition_probs(X, gammas, xis)
        B = self._update_emission_probs(X, gammas)

        # Calculate Frobenius norm (for A and B) and L2 norm (for PI).
        norm = (np.linalg.norm(self.A - A) + np.linalg.norm(self.B - B) +
                np.linalg.norm(self.PI - PI))

        # Update matrices.
        self.PI = PI
        self.A = A
        self.B = B

        return norm

    def _update_init_probs(self, X, gammas):
        """Update initial state distribution vector.

        Parameters
        ----------
        X : list
            List of observation sequences where observations (tokens)
            are mapped to integers.
        gammas : list
            List of numpy.array containting the probability estimates of
            the hidden state at each observation.

        Returns
        -------
        PI : numpy.array
            Updated initial state distribution vector.

        """
        PI = np.zeros(self.PI.shape)
        for state in range(self.num_states):
            prob_sum = 0
            for i in range(len(X)):
                prob_sum += gammas[i][0, state]
            # State distribution probability is the average across all sequences.
            PI[state] = prob_sum / len(X)

        # Check that probabilities sum to 1.
        np.testing.assert_allclose(PI.sum(), 1)
        return PI

    def _update_transition_probs(self, X, gammas, xis):
        """Update transition probability matrix.

        Parameters
        ----------
        X : list
            List of observation sequences where observations (tokens)
            are mapped to integers.
        gammas : list
            List of numpy.array containting the probability estimates of
            the hidden state at each observation.
        xis : list
            List of numpy.array containting the state transition probability
            estimates at each observation.

        Returns
        -------
        A : numpy.array
            Updated transition probability matrix.

        """
        A = np.zeros(self.A.shape)
        for prev_state in range(self.num_states):
            for next_state in range(self.num_states):
                numerator, denominator = 0, 0
                for j in range(len(X)):
                    # Skip last index in sequence (because no next state).
                    for i in range(len(X[j]) - 1):
                        numerator += xis[j][i, prev_state, next_state]
                        denominator += gammas[j][i, prev_state]
                A[prev_state, next_state] = numerator / denominator

            # Check that probabilities sum to 1.
            # np.testing.assert_allclose(A[prev_state].sum(), 1)

        return A

    def _update_emission_probs(self, X, gammas):
        """Update emission probability (observation likelihoods) matrix.

        Args:
            X: A list of lists (observation sequences) where observation
               (tokens) are mapped to integers.
            gammas: A list of matrices containting the probability estimates of
                    the hidden state at each observation.

        Returns
        -------
        B : numpy.array
            Updated emission probability matrix.

        """
        B = np.zeros(self.B.shape)
        for state in range(self.num_states):
            for token in range(self.num_obs):
                numerator, denominator = 0, 0
                for j in range(len(X)):
                    for i in range(len(X[j])):
                        probability = gammas[j][i, state]
                        if X[j][i] == token:  # Indicator function.
                            numerator += probability
                        denominator += probability
                B[state, token] = numerator / denominator

            # Check that probabilities sum to 1.
            # np.testing.assert_allclose(B[state].sum(), 1)

        return B
