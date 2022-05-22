<!-- ABOUT THE PROJECT -->
## About The Project

This is the final project from CS 451 Computational Intelligence offered at Habib University in Spring 2022.

### Abstract

This research presents an evolutionary algorithm approach for evolving a music composer. The algorithm runs over a corpus of MIDI musical files and compares the end result to well known musical compositions in order to judge similarity. Different fitness functions are implemented, and their results compared to determine the feasibility of different fitness functions for evolving musical compositions. The output is a set of musical notes generated entirely by the algorithm which exhibit vague musical tendencies. Our results demonstrate that musical composition through an evolutionary algorithm is not a trivial problem. Small changes in chromosome representation and fitness functions are vital to ensuring an aesthetic output.

### Built With

* [Python](https://www.python.org/)
* [NumPy](https://numpy.org/)
* [matplotlib](https://matplotlib.org/)
* [pretty_midi](https://craffel.github.io/pretty-midi/)
* [music21](https://web.mit.edu/music21/)
* [textdistance](https://pypi.org/project/textdistance/)

### Instructions:

Run main.py in the code folder to run the midi preprocessing on the corpus, and run the evolutionary algorithm for the specified parameters, to produce a fitness graph and generate an output midi file. The report is available in the report folder. 
