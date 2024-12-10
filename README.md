# Epyrunner

A package for running EPOCH simulations automatically on Viking and training ML models

For this script to work correctly you need to have the following folder structure:

```bash
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

## Installation

### EPOCH Installation

Prior to installing this package you must first have EPOCH installed. If you do not have it currently installed I have attached a handy guide here at the [Installing_EPOCH.md](Installing_EPOCH.md) file.

### Virtual Environments

> [!TIP]
> We recommend using the `uv` package to speed up `pip`. It can be installed on any operating system that has python installed by running `pip install uv` OR `pip3 install uv`

Once this is installed please create a [virtual python environment](https://docs.python.org/3/library/venv.html) using the following steps.

> [!IMPORTANT]  
> When creating a virtual environment it will be located in the folder where you run the venv creation command using one the below options

#### Using `uv`

- Install uv to manage packages `pip install uv`
- Restart your shell `exec $SHELL`
- Create a new virtual environment (venv) using `uv venv`
- Activate your venv using `source .venv/bin/activate`

#### Using Regular `pip`

- Create a new virtual environment (venv) using `python -m venv .`
- Activate your venv using `source .venv/bin/activate`

### Project Installation

> [!IMPORTANT]  
> If running on Viking we recommend running all of the following commands in your `~/scratch` folder. Please note that data is only retained in this folder for a certain amount of time and is not backed up. See [Viking data retention policy](https://vikingdocs.york.ac.uk/getting_started/storage_on_viking.html)

To use this project on Viking please log in using `ssh` and run the following from within the `~/scratch` folder:

```bash
git clone https://github.com/JoelLucaAdams/epyrunner
cd epyrunner
pip install . # If using uv add uv to the front of this line
```

## Running

### First time setup (only run once)

First run the `random_sampling.py` script to create random latin hypercube samples over a set of parameters described in the file in the `parameters` variable.

Example command to get the base script working:

```bash
python random_sampling.py --dir=$(pwd) --epochPath=~/scratch/epoch/epoch2d/ --numSimulations=1
```

There are a few other properties that can be used with this file currently, they can all be printed out using:

```bash
python random_sampling.py -h
```

### Training

After completion run the `train.py` script to run the model, no input parameters are required. It creates a simple Gaussian Process Regression (GPR) model and displays the 5 points with the highest uncertainty (largest standard deviation).

NOTE: This does not currently save the model to a file
NOTE: This is only a basic working example, requires heavy rework to correct.
NOTE: Looping behaviour to recursively train the model is not implemented yet. See TODO section below

## Developers Section

This section describes current tasks that need to be completed and contains information on how the files work

### Initial run - random_sampling.py

1. Run `epydeck` and generate the file paths for the input decks
2. Create a new jobscript within the top level parent folder for the campaign and do the following:
    1. Rename the `job_name` of the sbatch job
    2. Rename the `campaign_path` to the location where the campaign runs
    3. Rename the `array_range_max` to the total number of sims to run
    4. Rename the `epoch_dir` to the directory containing all EPOCH. e.g. `~/scratch/epoch`
    5. Rename the `epoch_version` to the epoch version. e.g. `epoch2d`
    6. Rename the `file_path` to location of the txt file containing the deck paths. e.g. `paths.txt`

### Looping run - train.py

1. Load the `paths.txt` file with links to all the simulations and their respective input decks and for each path do the following:
    1. `input.deck` - Select the desired features (e.g. "intensity" and "density")
    2. `*.sdf` - Load in one or several sdf files and acquire a output value from that simulation
2. Run a Gaussian Process Regression (GPR) model splitting data into train (75%) and test (25%)
3. Find the top 5 simulation parameters with the highest uncertainty
4. Create new input decks using the same template changing the features to those originally varying using `epydeck`
5. Create a new jobscript

### TODO

- Save GPR model after training
- Consider replacing existing GPR training with <https://github.com/bayesian-optimization/BayesianOptimization>
- Eventually migrate to pyspark
- Allow args to be passed into training to suggest most points
- Figure out how to cause the setup to trigger looping. Possibly by parent jobscript that runs setup.py, simulation and train.py. This can be accomplished using the `sbatch --parsable --dependency=afterok:JOBID` from <https://hpc-unibe-ch.github.io/slurm/dependencies.html>
- QCG-PilotJob
