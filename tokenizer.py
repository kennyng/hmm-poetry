import numpy as np


# Sonnets 99, 126, 145 removed due to format deviations.
NUM_SHAKESPEARE_SONNETS = 151
# SONNET FORMAT:
SONNET_LINES = 14
NUM_QUATRAINS = 3
QUATRAIN_LINES = 4
COUPLET_LINES = 2


def tokenize(line):
    """Tokenize sonnet lines into sequence of words.

    Args:
        line: single sonnet line as string.

    Returns:
        Formatted sonnet line with punctuation attached to word on the left
        and newline characters removed.

    """
    line = line.lower().lstrip().rstrip()
    line = line.split(' ')
    return line


# TRAINING SEQUENCES
def process_sonnet_lines(tokenizer, filename='data/shakespeare.txt'):
    """Process sonnets by splitting text on a per-line basis.

    Args:
        tokenizer: function for tokenizing each sonnet line.
        filename: text file of sonnets.

    Returns:
        List of tokenized sonnet lines.

    """
    data = load_raw_text(filename)
    lines = []
    cursor = 0

    for sonnet in range(NUM_SHAKESPEARE_SONNETS):
        # Skip first line (which is a number).
        cursor += 1
        for i in range(SONNET_LINES):
            lines.append(tokenizer(data[cursor]))
            cursor += 1

    return lines


def process_quatrains_couplets(tokenizer, filename='data/shakespeare.txt'):
    """Process sonnets by splitting text into quatrains and couplets.

    Each sonnet line from a quatrain is added to a set of quatrain lines
    and each sonnet line from a couplet is added to a set of couplet lines.

    Args:
        tokenizer: function for tokenizing each sonnet line.
        filename: text file of sonnets.

    Returns:
        Two lists - a set of quatrain lines and a set of couplet lines.
    """
    data = load_raw_text(filename)
    quatrains = []
    couplets = []
    cursor = 0

    for sonnet in range(NUM_SHAKESPEARE_SONNETS):
        cursor += 1
        for quatrain in range(NUM_QUATRAINS):
            for line in range(QUATRAIN_LINES):
                quatrains.append(tokenizer(data[cursor]))
                cursor += 1
        for line in range(COUPLET_LINES):
            couplets.append(tokenizer(data[cursor]))
            cursor += 1

    return quatrains, couplets


def _load_raw_text(filename='data/shakespeare.txt'):
    """Read in raw text of Shakespeare sonnets.

    Args:
        filename: text file of sonnets.

    Returns:
        Numpy array of sonnet data (lines) read from file.

    """
    data = np.loadtxt(filename, delimiter='\n', dtype='str')
    return data
