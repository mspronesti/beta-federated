# Federated Learning
Federated learning with `flower`, `pytorch` and `tensorflow`.

## Dependencies
You can install the required dependencies running

```bash
pip install -r requirements.txt
```

or
```bash
pip install -r requirements-dev.txt
```

in case you want the additional dependencies
required for the development.

Make sure to install the required hooks for `pre-commit` before committing
, running
```bash
pre-commit install
```
which will ensure `pre-commit` will run at `git commit`.

## Usage
Just play around with the `hydra` [configuration file](config/config.yaml) to set the hyperparameters
and run `server.py`
