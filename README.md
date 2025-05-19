# SEL3-2025-Groep-6 (Learning to Crawl: How Evolution Teaches Brittle Stars to Move)

Project Software Engineering Lab 3 @ UGent


Robots operating in complex environments need movement that is efficient, coordinated, and minimizes
wear and tear over time. Biological systems, like the Brittle star, offer inspiration through their
flexible locomotion. In this project, we use Evolution Strategies (ES) combined with Central Pattern
Generators (CPGs) to evolve natural movement in a simulated brittle star. By mimicking biology, we aim
to develop simple yet robust control strategies that could inspire safer and more adaptable robotic
systems.

This project can be used both as a CLI tool or as a library for in your own code.

## Code layout

All code is in the `groep6` package.
Shared code is at the top level of this package.
This package contains two sub packages for the two different learning methods. They contain the training and simulation code for the respective methods.

We have set some default values in the `defaults.py` files. These can be overwritten both by calling a function
with different parameters or by passing different parameters in the command line.


## Setup/Env
> This guide is made for a machine running on Linux.

Create and activate a venv with Python3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Make sure you have Poetry installed otherwise follow the instructions [here](https://python-poetry.org/docs/#installation).
Install Poetry dependencies.

```bash
poetry install
```
> If having problems with poetry install, try `poetry lock` and then try again.


## Train / Run
### CLI
Export the Weights and Biases api key to log training results:

```bash
export WANDB_API_KEY=<your_api_key>
```

To train the ES (similarly for PPO) model, run: 

```bash
python groep6/es/train_es.py # use "--help" for options
```

To view the results of a trained model.

```bash
python groep6/es/run_simulation.py # use "--help" for options
```

### Python

Example for ES, but similar for PPO:

```python
from groep6.es.train_es import train_es
from groep6.es.run_simulation import create_video

# to train a new model
train_es() # add / change any desired parameters here

# to run a simulation with a trained model stored at MODEL_FILE (default saving location of train_es)
create_video() # add / change any desired parameters here
```

Developed by:
- [Lukas Barragan Torres](https://github.com/lbarraga)
- [Emma Vandewalle](https://github.com/EmmaVandewalle)
- [Matthias Seghers](https://github.com/matt01y)

