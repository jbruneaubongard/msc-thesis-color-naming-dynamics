# Color-naming dynamics

Code for the simulations of [1]. This is adapted from [2].

## Installation

Clone the repository 

```
git clone https://github.com/jbruneaubongard/msc-thesis-color-naming-dynamics
cd msc-thesis-color-naming-dynamics
```

Create a conda environment that will have the necessary dependencies:

```
conda env create --name color-naming-dynamics --file environment.yml
conda activate color-naming-dynamics
```

Install the package with `pip`:
```
pip install -e .
```
## Usage
### Compute the trahectories

To compute the trajectories starting from a language of a given `dataset`, using a given `algorithm`, run the following command:

```
cd scripts 
python trajectories.py --dataset=dataset --mode=algorithm
```
The results will be saved in separate `.pkl` files in the `results` folder.

### Visualize the results
You can visualize the languages' trajectories and plot the initial and final languages using the `viz.ipynb` file in the `scripts` folder. 
## References 

[1] Bruneau--Bongard, J. (2024). *Dynamic Characterizations of Cross-Linguistic Semantic Organization*, Master's Thesis.

[2] Carlsson, E., Dubhashi, D., and Regier, T. (2023). *Iterated learning and communication jointly explain efficient color naming systems.* Proceedings of the 45th Annual Meeting of the Cognitive Science Society.