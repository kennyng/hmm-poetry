import pickle
import numpy as np


# Sonnets 99, 126, 145 removed due to deviations.
NUM_SHAKESPEARE_SONNETS = 151
# SONNET FORMAT:
SONNET_LINES = 14
NUM_QUATRAINS = 3
QUATRAIN_LINES = 4
COUPLET_LINES = 2

PUNCTUATION = [',', ':', '.', ';', '?', '!', '(', ')', "'", '"']
SYLLABLE_DICT = 'syllable_dict.pickle'
STRESS_DICT = 'stress_dict.pickle'


def load_raw_text(filename='shakespeare.txt'):
    """Read in raw text of Shakespeare sonnets.

    Args:
        filename: text file of sonnets.

    Returns:
        A numpy array of sonnet data (lines) read from file.

    """
    sequences = np.loadtxt(filename, delimiter='\n', dtype='str')
    return sequences


# TOKENIZERS
def tokenize_sonnet(line):
    """Tokenize sonnet lines into sequence of words."""
    # Attach punctuation to word on left and remove newline characters.
    line = line.lower().lstrip().rstrip()
    line = line.split(' ')
    return line


# TRAINING SEQUENCES
def sequence_sonnet_lines(tokenizer, filename='shakespeare.txt'):
    """
        Sequence sonnets by splitting text on a per-line basis.

    """
    data = load_raw_text(filename)
    sequences = []
    cursor = 0

    for sonnet in range(NUM_SHAKESPEARE_SONNETS):
        # Skip first line (which is a number).
        cursor += 1
        for i in range(SONNET_LINES):
            sequences.append(tokenizer(data[cursor]))
            cursor += 1

    return sequences


def sequence_quatrain_couplet_lines(tokenizer, filename='shakespeare.txt'):
    """
    Sequence sonnets by splitting text into a set of quatrain lines
    and a set of couplet lines.

    """
    data = load_raw_text(filename)
    quatrains = []
    couplets = []
    cursor = 0

    for sonnet in range(NUM_SHAKESPEARE_SONNETS):
        cursor += 1
        couplet = []
        for quatrain in range(NUM_QUATRANS):
            for line in range(QUATRAIN_LINES):
                quatrains.append(tokenizer(data[cursor]))
                cursor += 1
        for line in range(COUPLET_LINES):
            couplets.append(tokenizer(data[cursor]))
            cursor += 1

    return quatrains, couplets

