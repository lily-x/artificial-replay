# Artifical Replay

This code implements and evaluates algorithms for the paper Artificial Replay: A Meta-Algorithm for Harnessing Historical Data in Bandits. In this paper we introduce Artificial Replay, a wrapper algorithm to most efficiently integrate historical data into arbitrary multi-armed bandit algorithms that exhibit independence of irrelevant data (IIData), overcoming spurious data and imbalanced data coverage.

We implement UCB-based algorithms using both fixed discretization and adaptive discretization for combinatorial bandits with continuous resource allocation (CMAB-CRA). We include a number of simulation environments including a quadratic function, piecewise-linear, and one based on real poaching data in Uganda. This library also implements and compares against a variety of baselines.

```
@inproceedings{banerjee2023artificial,
  title={Artificial Replay: A Meta-Algorithm for Harnessing Historical Data in Bandits},
  author={Banerjee, Siddhartha and Sinclair, Sean R. and Tambe, Milind and Xu, Lily and Yu, Christina Lee},
  booktitle={arXiv},
  year={2023},
}
```

This project is licensed under the terms of the MIT license.


## Usage

To execute the code and run experiments comparing Artificial Replay against the baselines for combinatorial continuous bandits, from the `./cmab_model/` directory, run:
```sh
python main.py
```
For finite K-armed bandits, from the `./k_armed_model/` directory, run
```sh
python main.py
```

## Files

In the `./cmab_model` directory:
- `main.py` - driver
- `environment.py`
- for learning and setting up wildlife environment from real-world data
- `preprocess.py` -
- `get_data.py` -
- `learn_reward.py` -
- `algorithms/` - directory with implementation of algorithms
- `algorithm.py`
- `fixed_artificial_replay.py`
- `fixed_historical.py`
- `fixed_discretization.py`
- `adaptive_artificial_replay.py`
- `adaptive_historical.py`
- `adaptive_discretization.py`
- `regression.py`
- `algorithms/utils/` - general helper functionality
- `bounds_utils.py`
- `common.py`
- `tree.py`

For `./finite_armed_model`
- `main.py` - driver
- `k_armed.py` - environment class and algorithms



## Requirements
- python==3.10.6
- seaborn==0.12.0
- pandas==1.5.0
- pytorch==1.12.1
- tqdm==4.64.1
- geopandas==0.11.1
- scikit-learn==1.1.1
- shapely==1.8.4
- pyproj==3.4.0
- richdem==2.3.0
- rasterio==1.3.2
- gdal==3.5.2
- gurobi==9.5.2
