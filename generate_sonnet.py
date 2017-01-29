#!/usr/bin/env python

import sys
import poemgen


MODELS_DICT = {"rhyme":
               {'10': "10-states-rhyme",
                '20': "20-states-rhyme",
                '30': "30-states-rhyme",
                '50': "50-states-rhyme",
                '75': "75-states-rhyme",
                '100': "100-states-rhyme"},
               "meter":
               {'20': "20-states",
                '30': "30-states"}}


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: ./generate_sonnet.py [rhyme|meter] [num_states]")
        sys.exit()

    poem = None
    model = None
    if sys.argv[1] == "rhyme":
        model = MODELS_DICT["rhyme"].get(sys.argv[2])
        if model:
            poem = poemgen.generate_poem_rhyme(model)
    elif sys.argv[1] == "meter":
        model = MODELS_DICT["meter"].get(sys.argv[2])
        if model:
            poem = poemgen.generate_poem(model)

    if poem and model:
        print("{:~^30}".format(model.upper()))
        print(poem)
