import argparse
import sys
from pathlib import Path

import epydeck
import epyscan

import epyrunner

# Some arguments that can be passed into this function via the terminal
# Run python setup.py -h for list of possible arguments
parser = argparse.ArgumentParser("parser")
parser.add_argument(
    "--verbose",
    action="store_true",
    help="Displays verbose output including directory locations.",
)
parser.add_argument(
    "--test",
    action="store_true",
    help="Simply tests that the script can run without executing the simulations.",
)
args = parser.parse_args()

# Paths setup
# TODO: Fix this hack
script_path = Path("/users/bmp535/scratch/epoch_runner")
template_deck_filename = "template.deck"
template_jobscript_filename = "template.sh"
campaign_dir_name = "example_campaign"
simulation_path_filename = "paths.txt"

epoch_version = "epoch2d"
epoch_path = Path(script_path.parent / "epoch")

if args.verbose:
    print(f"Script directory: {script_path}")
    print(f"Template deck: {template_deck_filename}")
    print(f"Template jobscript: {template_jobscript_filename}")
    print(f"Simulation paths: {simulation_path_filename}")
    print(f"Campaign folder: {campaign_dir_name}")
    print(f"Epoch version: {epoch_version}")
    print(f"Epoch location: {epoch_path}")


# INITIAL RANDOM SAMPLING
# -----------------------
with Path.open(script_path / template_deck_filename) as f:
    deck = epydeck.load(f)

parameters = {
    "constant:intens": {"min": 1.0e22, "max": 1.0e24, "log": True},
    "constant:nel": {"min": 1.0e20, "max": 1e24, "log": True},
}

# Sets up sampling of simulation and specifies number of times to run each simulation
hypercube_samples = epyscan.LatinHypercubeSampler(parameters).sample(40)

# Takes in the folder and template and starts a counter so each new simulation gets saved to a new folder
campaign = epyscan.Campaign(deck, (script_path / campaign_dir_name))

# Randomly samples the parameter space and creates folders for each simulation
paths = [campaign.setup_case(sample) for sample in hypercube_samples]

# Save the paths to a file on separate lines
with Path.open(simulation_path_filename, "w") as f:
    [f.write(str(path) + "\n") for path in paths]

if args.test:
    print("Successfully validated and created filepaths")
    sys.exit()

# EXECUTE THE JOB
# ---------------
job = epyrunner.SlurmJob(args.verbose)

job.enqueue_array_job(
    epoch_path=epoch_path,
    epoch_version=epoch_version,
    campaign_path=script_path / campaign_dir_name,
    file_path=script_path / simulation_path_filename,
    template_path=script_path / template_jobscript_filename,
    n_runs=len(paths),
    job_name="setup_run",
)

job.poll_jobs(interval=2)
_, failed_jobs = job.get_job_results()

if failed_jobs:
    print("The following jobs failed", failed_jobs)
    sys.exit("Initial/Setup simulation run failed. See job log files")
