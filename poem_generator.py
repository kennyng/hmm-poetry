import numpy as np
import shakespeare_tokenizer as tokenizer


def generate_line(transition, emission, init_vec, wordmap, syllable_dict,
                  stress_dict, length=10, delim=' ', current_state=None):
    """Generates an un-rhymed sonnet line.
    
    The start state is chosen using an initial vector and the length
    specified by the number of syllables.

    """
    # Verify model is functional.
    num_states = len(transition)
    num_words = len(emission[0])
    assert(num_states == len(transition[0])), "Transition matrix is not square."
    assert(num_states == len(emission)), "Emission matrix has incorrect dimensions."

    if current_state is None:
        current_state = np.random.choice(num_states, p=init_vec)
    current_length = 0
    current_stress = 0
    line = ''

    while current_length < length:
        # Select random word according to emission matrix.
        token = int(np.random.choice(num_words, p=emission[int(current_state)]))
        word = wordmap[token].rstrip('.,?!;:()').lstrip('(')

        # print(word)

        # Check that word is not too long and is stressed correctly.
        while (syllable_dict[word] + current_stress > length) or (stress_dict[word] != current_stress):
            token = int(np.random.choice(num_words, p=emission[int(current_state)]))
            word = wordmap[token].rstrip('.,?!;:()').lstrip('(')

        line += wordmap[token] + delim
        # Keep track of line length and number of syllables
        current_length += syllable_dict[word]
        current_stress += (current_stress + syllable_dict[word]) % 2

        # Go to next state.
        current_state = np.random.choice(num_states, p=transition[int(current_state)])

    return line, current_state
