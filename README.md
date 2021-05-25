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

Base example on toy dag.
```bash
$ python main.py
```

An example to run toy example with a dataset size of 2000 for 300 max_steps with a missingness of 30%
```bash
$ python3 run_example.py --dataset_sz 2000 --max_steps 300 --missingness 0.3
```

