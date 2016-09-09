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


def syllable_count_and_has_stress(line, syllable_dict, stress_dict):
    count = 0
    stress = True
    current_stress = 0
    words = tokenize_sonnet(line)

    for word in words:
        for punc in PUNCTUATION:
            word = word.replace(punc, '')
        if word != '':
            count += syllable_dict[word]
            if stress_dict[word] != current_stress:
                stress = False
            current_stress = (current_stress + syllable_dict[word]) % 2

    return count, stress


def fix_sentence_mechanics(line):
    # Capitalize the first word.
    words = line.split(' ')
    words[0] = words[0].capitalize()

    # Capitalize the word 'I'.
    for i in range(len(words)):
        if words[i] == 'i':
            words[i] = 'I'

    # Capitalize the first word of sentences.
    for i in range(len(words)-1, 0, -1):
        prev = words[i-1]
        if prev[-1] in ['!', '.', '?']:
            words[i] = words[i].capitalize()

    return ' '.join(words)


def load_syllable_stress_dicts():
    syllable_file = open(SYLLABLE_DICT, 'rb')
    syllable_dict = pickle.load(syllable_file)
    syllable_file.close()

    stress_file = open(STRESS_DICT, 'rb')
    stress_dict = pickle.load(stress_file)
    stress_file.close()

    return syllable_dict, stress_dict
