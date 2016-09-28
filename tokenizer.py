import numpy as np


# Sonnets 99, 126, 145 removed due to format deviations.
NUM_SHAKESPEARE_SONNETS = 151
# SONNET FORMAT:
SONNET_LINES = 14
NUM_QUATRAINS = 3
QUATRAIN_LINES = 4
COUPLET_LINES = 2

SONNET_FILEPATH = 'data/shakespeare.txt'


def tokenize(line):
    """Parse sonnet lines and tokenize on words.

    Tokenization rules:
        - For standardization, all lines are in lowercase.
        - Newline characters are removed.
        - Hyphenated words are kept as single words
            (to prevent random words to be hyphenated together).
        - Ending punctuation is attached to the word to its left
            (this can help the model learn the positioning
             of words with punctuation).

    Args:
        line: A string representing a single sonnet line.

    Returns:
        Formatted sonnet line with punctuation attached to word on the left
        and newline characters removed.

    """
    line = line.lower().lstrip().rstrip()
    line = line.split(' ')
    return line


# TRAINING SEQUENCES
def sequence_each_line(filename=SONNET_FILEPATH):
    """Split sonnets into training sequences on a per-line basis.

    Args:
        tokenizer: A function for tokenizing each sonnet line.
        filename: A string representing the path to the sonnets text file.

    Returns:
        List of training sequences, where each sequence is
        a tokenized sonnet line.

    """
    data = _load_raw_text(filename)
    lines = []
    cursor = 0

    for sonnet in range(NUM_SHAKESPEARE_SONNETS):
        # Skip first line (which is a number).
        cursor += 1
        for i in range(SONNET_LINES):
            lines.append(tokenize(data[cursor]))
            cursor += 1

    return lines


def sequence_quatrains_couplets(filename=SONNET_FILEPATH):
    """Split sonnets into training sequences as sets of quatrains and couplets.

    Sonnets are split into quatrains and couplets, and then quatrains and
    couplets are split on a per-line basis.

    Args:
        tokenizer: A function for tokenizing each sonnet line.
        filename: A string representing the path to the sonnets text file.

    Returns:
        Two lists of training sequences: a set of all lines of all
        the quatrainss and a set of all lines of all the couplets.
    """
    data = _load_raw_text(filename)
    quatrains = []
    couplets = []
    cursor = 0

    for sonnet in range(NUM_SHAKESPEARE_SONNETS):
        cursor += 1
        for quatrain in range(NUM_QUATRAINS):
            for line in range(QUATRAIN_LINES):
                quatrains.append(tokenize(data[cursor]))
                cursor += 1
        for line in range(COUPLET_LINES):
            couplets.append(tokenize(data[cursor]))
            cursor += 1

    return quatrains, couplets


def process_rhymes(filename=SONNET_FILEPATH):
    """Compile lists of rhyming pairs from the sonnets text.

    Args:
        filename: A string representing the path to the sonnets text file.

    Returns:
        A list of tuples containing rhyming pairs from all the quatrains
        ans a list of tuples containing rhyming pairs from all the couplets.

    """
    data = _load_raw_text(filename)
    quatrain_rhymes = []
    couplet_rhymes = []
    cursor = 0

    for sonnet in range(NUM_SHAKESPEARE_SONNETS):
        cursor += 1
        for quatrain in range(NUM_QUATRAINS):
            line0 = tokenize(data[cursor])
            line1 = tokenize(data[cursor + 1])
            line2 = tokenize(data[cursor + 2])
            line3 = tokenize(data[cursor + 3])
            quatrain_rhymes.append((line0[-1], line2[-1]))
            quatrain_rhymes.append((line1[-1], line3[-1]))
            cursor += 4

        line0 = tokenize(data[cursor])
        line1 = tokenize(data[cursor + 1])
        couplet_rhymes.append((line0[-1], line1[-1]))
        cursor += 2

    return quatrain_rhymes, couplet_rhymes


def process_word_frequency(filename=SONNET_FILEPATH):
    data = _load_raw_text(filename)
    word_count = {}
    cursor = 0
    
    for sonnet in range(NUM_SHAKESPEARE_SONNETS):
        cursor += 1
        for i in range(SONNET_LINES):
            words = tokenize(data[cursor])
            for word in words:
                if word_count.get(word) is not None:
                    word_count[word] += 1
                else:
                    word_count[word] = 1
            cursor += 1
    
    return word_count

    
def _load_raw_text(filename=SONNET_FILEPATH):
    """Read in raw text of Shakespeare sonnets.

    Args:
        filename: A string representing the path to the sonnets text file.

    Returns:
        Numpy array of sonnet data (lines) read from file.

    """
    data = np.loadtxt(filename, delimiter='\n', dtype='str')
    return data
