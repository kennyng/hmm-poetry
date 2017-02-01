##############################################################################
# hmmhelper.py
# ------------
# Helper methods for training, saving, and loading Hidden Markov Models.
#
##############################################################################

import os
import pickle

import tokenizer
from hmm import HiddenMarkovModel


MODEL_DIRNAME = 'models'
SYLLABLE_DICT = 'data/syllable_dict.p'
STRESS_DICT = 'data/stress_dict.p'
PUNCTUATION = [',', ':', '.', ';', '?', '!', '(', ')', "'", '"']


def train_hmm(model_name, data, num_states,
              epsilon=0.01, max_iter=100, scale=True):
    """Wrapper function to train HMM and then save resulting model parameters.

    Parameters
    ----------
    model_name : str
        Model name for saving the resulting HMM to file.
    data : list
        Training sequences (observation data).
    num_states : int
        Number of states to have in the model.
    epsilon : float
        Value to calculate the stopping condition for algorithm
        to converge (when ratio between updated norm and initial
        norm is less than epsilon).
    max_iter : int
        Maximum number of iterations that the algorithm should run.
        (Set max_iter=0 for indefinite number of iterations.)
    scale : bool
        Indicates whether to normalize probability vectors.

    """
    hmm = HiddenMarkovModel(num_states)
    A, B, PI, token_dict = hmm.train(data)
    save_hmm(model_name, A, B, PI, token_dict)


def save_hmm(model_name, A, B, PI, token_dict):
    """Save HMM parameters as specified model name.

    Parameters
    ----------
    model_name : str
        Model name for saving the resulting HMM to file.
    A : numpy.array
        Transition proability matrix of HMM.
    B : numpy.array
        Emission probability matrix of HMM.
    PI : numpy.array
        Initial state distribution vector of HMM.
    token_dict : dict
        Mapping from tokens to observation indices (int).

    """
    try:
        model_dirpath = os.path.join(MODEL_DIRNAME, model_name)
        os.mkdir(model_dirpath)
    except FileExistsError:
        print("Directory for model already exists; model will be overwritten.")
    finally:
        model_filepath = os.path.join(model_dirpath, model_name)
        with open(model_filepath + '-TRANSITION.p', 'wb') as fp:
            pickle.dump(A, fp)

        with open(model_filepath + '-EMISSION.p', 'wb') as fp:
            pickle.dump(B, fp)

        with open(model_filepath + '-INIT.p', 'wb') as fp:
            pickle.dump(PI, fp)

        with open(model_filepath + '-TOKENS.p', 'wb') as fp:
            pickle.dump(token_dict, fp)


def load_hmm(model_name):
    """Load HMM with given model name.

    Parameters
    ----------
    model_name : str
        Name of model to load from file.

    Returns
    -------
    A : numpy.array
        Transition proability matrix of HMM.
    B : numpy.array
        Emission probability matrix of HMM.
    PI : numpy.array
        Initial state distribution vector of HMM.
    token_dict : dict
        Mapping from tokens to observation indices (int).

    """
    A, B, PI, token_dict = None, None, None, None
    model_filepath = os.path.join(MODEL_DIRNAME, model_name, model_name)

    with open(model_filepath + '-TRANSITION.p', 'rb') as fp:
        A = pickle.load(fp)

    with open(model_filepath + '-EMISSION.p', 'rb') as fp:
        B = pickle.load(fp)

    with open(model_filepath + '-INIT.p', 'rb') as fp:
        PI = pickle.load(fp)

    with open(model_filepath + '-TOKENS.p', 'rb') as fp:
        token_dict = pickle.load(fp)

    return A, B, PI, token_dict


def fix_sentence_mechanics(line):
    """Parses line and fixes sentence mechanics (mostly capitalization)."""
    # Capitalize the first word.
    words = line.split(' ')
    words[0] = words[0].capitalize()

    # Capitalize the word 'I'.
    for i in range(len(words)):
        if words[i] == 'i':
            words[i] = 'I'

    # Capitalize the first word of sentences.
    for i in range(len(words) - 1, 0, -1):
        prev = words[i - 1]
        if prev[-1] in ['!', '.', '?']:
            words[i] = words[i].capitalize()

    return ' '.join(words)


def syllable_count(line, syllable_dict):
    """Returns the number of syllables in a given line."""
    count = 0
    words = tokenizer.tokenize_nopunc(line)

    for word in words:
        if word != '':
            count += syllable_dict[word]

    return count


def has_stress(line, stress_dict, syllable_dict):
    """Checks whether a given line ends in a stress or unstressed syllable."""
    stress = True
    current_stress = 0
    words = tokenizer.tokenize_nopunc(line)

    for word in words:
        if word != '':
            if stress_dict[word] != current_stress:
                stress = False
            current_stress = (current_stress + syllable_dict[word]) % 2

    return stress


def load_syllable_dict():
    return _load_lingual_dict(SYLLABLE_DICT)


def load_stress_dict():
    return _load_lingual_dict(STRESS_DICT)


def _load_lingual_dict(filename):
    with open(filename, 'rb') as fp:
        d = pickle.load(fp)
    return d
