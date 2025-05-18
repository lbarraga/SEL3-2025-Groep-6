# SEL3-2025-Groep-6 (Learning to Crawl: How Evolution Teaches Brittle Stars to Move)

Project Software Engineering Lab 3 @ UGent


Robots operating in complex environments need movement that is efficient, coordinated, and minimizes
wear and tear over time. Biological systems, like the Brittle star, offer inspiration through their
flexible locomotion. In this project, we use Evolution Strategies (ES) combined with Central Pattern
Generators (CPGs) to evolve natural movement in a simulated brittle star. By mimicking biology, we aim
to develop simple yet robust control strategies that could inspire safer and more adaptable robotic
systems.


## Code layout

Shared code is at the top level, and ES / PPO specific code are in their own sub directories.

The shared parameters can be adjusted in `config.py`, while method specific parameters can be
adjusted in their respective files. (E.g. to adjust training `POPULATION_SIZE` for ES it would be in `train_es.py`)


## Setup/Env
> This guide is made for a machine running Linux.

Create and activate a venv with Python3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install Poetry dependencies.

```bash
poetry install --no-root
```
> If having problems with poetry install, try `poetry lock` and then try again.


## Train / Run

Export the weights and biases api key, to log training results:

```bash
export WANDB_API_KEY=<your_api_key>
```

To train the ES model run:

```bash
python es/train_es.py
```
Similarly for PPO.


To view the end results of an es trained model,
specify the model in `MODEL_FILENAME` variable and run:

```bash
python es/run_simulation.py
```

While the ppo training will always create a video when done training.
To generate a video of a previously trained PPO model, change the training
code for the loading code that is currently commented out.


Developed by:
- [Lukas Barragan Torres](https://github.com/lbarraga)
- [Emma Vandewalle](https://github.com/EmmaVandewalle)
- [Matthias Seghers](https://github.com/matt01y)

