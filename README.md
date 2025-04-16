# Bio-Robot-CPG
Project Software Engineering Lab 3

## setup/env

create and activate a venv with python3.11

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

install poetry dependencies

```bash
poetry install --no-root
```

## run a script

export the weights and biases api key

```bash
export WANDB_API_KEY=<your_api_key>
```

run a script

```bash
python train_es.py
```