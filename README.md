# Federated Learning
Federated learning with `flower`, `pytorch` and `tensorflow`.

## Dependencies
You can install the required dependencies running

```bash
pip install -r requirements.txt
```

## Usage
Just play around with the `hydra` [configuration file](config/config.yaml) to set the hyperparameters
and run `server.py`

```bash
cd fed_torch
python server.py
```

In case you have any problem with imports, try setting the `PYTHONPATH` environment variable in your
virtual environment

```bash
export PYTHONPATH=$PYTHONPATH:`pwd`
```

## Development Notes
Install development requirements running

```bash
pip install -r requirements-dev.txt
```

and run the tests with
```bash
python -m pytest
```

Make sure to install the required hooks for `pre-commit` before committing
, running
```bash
pre-commit install
```
which will ensure `pre-commit` will run at `git commit`.
