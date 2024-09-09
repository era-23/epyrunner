# Epyrunner
A package for running EPOCH simulations automatically on Viking and training ML models

For this script to work correctly you need to have the following folder structure:
```
.
├── epyrunner
│ ├── random_sampling.py # The script file to run
│ ├── train.py # The script file to train the model
│ ├── template.deck # The template deck
│ ├── template.sh # The Slurm/Viking template jobscript
│ ├── README.md
│ ├── src
│ │   └── epyrunner # helper functions
│ │       └── __init__.py
│ └── ...
├── epoch
│ ├── epoch1d
│ ├── epoch2d
│ └── epoch3d
└── ...
```

# Installation 
Until this is on PyPI, please install directly from this repo:

```bash
pip install git+https://github.com/JoelLucaAdams/epyrunner.git@main
```

or from a local checkout:

```bash
git clone https://github.com/JoelLucaAdams/epyrunner
cd epyrunner
pip install .
```

# Running
## First time setup (only run once)
First run the `random_sampling.py` script to create random latin hypercube samples over a set of parameters described in the file in the `parameters` variable.

Example running command line input
```bash
python random_sampling.py --dir=$(pwd) --epochPath=~/scratch/epoch/epoch2d/ --numSimulations=1 --verbose
```

## Training
After completion run the `train.py` script to run the model, no input parameters are required. It creates a simple Gaussian Process Regression (GPR) model and displays the 5 points with the highest uncertainty (largest standard deviation).

NOTE: This does not currently save the model to a file
NOTE: This is only a basic working example, requires heavy rework to correct.
NOTE: Looping behaviour to recursively train the model is not implemented yet. See TODO section below

# Initial run - random_sampling.py
1. Run `epydeck` and generate the file paths for the input decks
2. Create a new jobscript within the top level parent folder for the campaign and do the following:
	1. Rename the `job_name` of the sbatch job
	2. Rename the `campaign_path` to the location where the campaign runs
	3. Rename the `array_range_max` to the total number of sims to run
	4. Rename the `epoch_dir` to the directory containing all EPOCH. e.g. `~/scratch/epoch`
	5. Rename the `epoch_version` to the epoch version. e.g. `epoch2d`
	6. Rename the `file_path` to location of the txt file containing the deck paths. e.g. `paths.txt`

# TODO
- Save GPR model after training
- Consider replacing existing GPR training with https://github.com/bayesian-optimization/BayesianOptimization
- Allow args to be passed into training to suggest most points
- Figure out how to cause the setup to trigger looping. Possibly by parent jobscript that runs setup.py, simulation and train.py. This can be accomplished using the `sbatch --parsable --dependency=afterok:JOBID` from https://hpc-unibe-ch.github.io/slurm/dependencies.html

# Looping run - train.py
1. Load the `paths.txt` file with links to all the simulations and their respective input decks and for each path do the following:
	1. `input.deck` - Select the desired features (e.g. "intensity" and "density")
	2. `*.sdf` - Load in one or several sdf files and acquire a output value from that simulation
2. Run a Gaussian Process Regression (GPR) model splitting data into train (75%) and test (25%)
3. Find the top 5 simulation parameters with the highest uncertainty
4. Create new input decks using the same template changing the features to those originally varying using `epydeck`
5. Create a new jobscript