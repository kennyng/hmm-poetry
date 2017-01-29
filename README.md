## hmm-poetry


### Dependencies:
- Python 3+ (developed and tested using Python 3.5.1)
- [numpy](https://github.com/numpy/numpy)
	- 'pip install numpy'
- (optional) [wordnik API](http://developer.wordnik.com/): for rhyming


### Files:
- 'tokenizer.py': Parses sonnet data (text file) into tokens for model training.
- 'hmm.py': Hidden Markov Model implementation.
- 'hmmhelper.py': Helper methods for training, saving, and loading HMM.
- 'poemgen.py': Generates sonnets from a given model.


### How to Run:
- To train your own models: see `test-hmm.ipynb` for an example.
	- Each HMM must include (in Python pickle format):
		- Initial State Distribution Vector (*-INIT.p)
		- Transition Probability Matrix (*-TRANSITION.p)
		- Emission Probability Matrix (*-EMISSION.p)
		- Mapping of word tokens to observation indicies (*-TOKENS.p)
	- See `models/test` for example trained HMM.

- To generate poems from a pre-existing model:
	<pre>
	import poemgen

	poemgen.generate_poem('path-to-model')
	poemgen.generate_poem_rhyme('path-to-model')
	</pre>


## Sample Sonnet (non-rhyming):
1 When bright thy sober thee all fortune

2 To carry love, those 'greeing, grace divide

3 The days my nothing frame the interest

4 Of mine invention assailed, taught have show,

5 Not outward sea, with habit lovely place

6 Thee, dates lover, my captain wrought having

7 Authorizing thy change, doth beauty were

8 To ill, in ocean be one, others wear

9 Thee that and never were dispatch and will

10 With gentle herald rich of bastard all

11 Not a knowing my habitation so

12 Delight art, rolling haste, strange pursuit side

13 Or world unused remains, such heart's thee show

14 The be glass leave thee lose in verse next that
