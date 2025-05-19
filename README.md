# SEL3-2025-Groep-6 (Learning to Crawl: Evaluating CPG Inference frequency in Brittle Star Robots)

Project Software Engineering Lab 3 @ UGent

We investigate how the Central Pattern Generators (CPGs) modulation interval affects the locomotion performance of a simulated brittle star.
Inspired by the animal’s flexible and resilient movement, we use these CPGs to generate rhythmic motion and Evolution Strategies (ES) to optimize control.
By adjusting the rate of decision-making, we explore the effect on correctness. This biologically inspired approach aims to create adaptable, low-complexity controllers for soft robotic systems.

This project can be used both as a CLI tool or as a library for in your own code.

## Code layout

All code is in the `groep6` package.
Shared code is at the top level of this package.
This package contains two sub packages for the two different learning methods (ES and PPO). 
They contain the training and simulation code for the respective methods.

The code can be used in two different ways:
1. **Command Line Interface (CLI)**: The code can be run from the command line.
2. **Python Library**: The code can be imported and used in your own Python code.

## Setup/Env
> This guide is made for a machine running on Linux.

Create and activate a venv with Python3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Make sure you have Poetry installed. Otherwise, follow the instructions [here](https://python-poetry.org/docs/#installation).
Install Poetry dependencies.

```bash
poetry install
```
> When having trouble with `poetry install`, try `poetry lock` and then try again.


## Train / Run
### CLI
Export the Weights and Biases api key to log training results:

```bash
export WANDB_API_KEY=<your_api_key>
```

To train an ES model, run: 

```bash
python groep6/es/train_es.py # use "--help" for options
```

To view the results of a trained model as a video.

```bash
python groep6/es/run_simulation.py trained_model.msgpack  # use "--help" for options 
```

Similarly, to train a PPO model, run:

```bash
python groep6/ppo/train_ppo.py  # use "--help" for options
```

to view the video:

```bash
python groep6/ppo/run_simulation.py trained_model.zip # use "--help" for options
```

Developed by:
- [Lukas Barragan Torres](https://github.com/lbarraga)
- [Emma Vandewalle](https://github.com/EmmaVandewalle)
- [Matthias Seghers](https://github.com/matt01y)

