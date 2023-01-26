# Artifical Replay

Siddhartha Banerjee, Sean R. Sinclair, Milind Tambe, Lily Xu, Christina Lee Yu

This code implements and evaluates algorithms for the paper [Artificial Replay: A Meta-Algorithm for Harnessing Historical Data in Bandits](https://arxiv.org/abs/2210.00025). In this paper we introduce Artificial Replay, a wrapper algorithm to most efficiently integrate historical data into arbitrary multi-armed bandit algorithms that exhibit independence of irrelevant data (IIData), overcoming spurious data and imbalanced data coverage.

For **K-armed bandits**, we include UCB, Thompson sampling, and IDS, and provide versions with the Artificial Replay wrapper for each approach. For **combinatorial bandits with continuous resource allocation** (CMAB-CRA), we implement UCB-based algorithms using both fixed discretization and adaptive discretization. We include a number of **simulation environments** including a quadratic function, piecewise-linear, and one based on real poaching data in Uganda. 


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

### CMAB-CRA
CMAB-CRA (continuous combinatoiral bandits), with both fixed and adaptive discretization.

In the `./cmab_model` directory:
- `main.py` - driver to run experiments. To change between fixed and adaptive discretization, change which items are selected in `algo_list`
- `environment.py` - define generic environment and PWL, Quadratic, Random, and Uganda settings
- `algorithms/` - implementation of algorithms
    - `./utils/` - general helper functionality. Contains `tree.py` tree implementation for adaptive discretization, `bounds_utils.py` for checking discretization region bounds, and `common.py` with optimization and confidence radius implementation
    - `./algorithm.py` - generic algorithm class
    - `./fixed_artificial_replay.py` - Artificial Replay wrapper class with fixed discretization
    - `./fixed_historical.py` - Full Start with fixed discretization
    - `./fixed_discretization.py` - fixed discretization with no historical data (Ignorant)
    - `./adaptive_artificial_replay.py` - Artificial Replay wrapper class with adaptive discretization
    - `./adaptive_historical.py` - Full Start with adaptive discretization
    - `./adaptive_discretization.py` - adaptive discretization with no historical data (Ignorant)
    - `./regression.py` - Regressor, which trains a neural network on the historical data

Poaching data is not included due to the sensitive nature of the data.

### K-armed bandits
Finite K-armed bandits.

In the `./finite_armed_model` directory:
- `main.py` - driver to run experiments for the finite K-armed bandit
- `k_armed.py` - environment class for K-armed bandits and algorithms
- `common.py` - helper functionality
- `algorithms/` - implementation of algorithms
    - `./algorithm.py` - generic class for algorithms
    - `./common.py` - helper functionality
    - `./online_wrapper.py` - generic wrapper class for the Artificial Replay meta-algorithm
    - `./ids.py` - information-directed sampling (IDS)
    - `./thompson_sampling.py` - Thompson sampling (TS)
    - `./ucb.py` - UCB
    - `./historical_ids.py` - information-directed sampling (IDS) with Full Start
    - `./historical_thompson_sampling.py` - Thompson sampling (TS) with Full Start
    - `./historical_ucb.py` - UCB with Full Start



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
