# DECAF (DEbiasing CAusal Fairness)

This is example code for running DECAF for synthesizing fair data.  

## Requirements 

- Python 3.6+
- `PyTorch`
- `PyTorch Lightning`
- `numpy`
- `networkx`
- `scikit-learn`
- `pandas`

## Contents

- `DECAF.py` - Synthetic data generator class - DECAF.
- `main.py` - Runs a nonlinear toy DAG example.  The dag structure is stored in the `dag_seed` variable.  The edge removal is stored in the `bias_dict` variable.  See example usage in this file.
- `utils.py` 

## Examples

Base example on toy dag:
```bash
$ python main.py
```

An example to run with a dataset size of 2000 for 300 epochs:
```bash
$ python main.py --datasize 2000 --epochs 300
```

