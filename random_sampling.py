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
parser.add_argument(
    "--dir",
    action="store",
    type=str,
    required=True,
    help="The epyrunner directory",
)
parser.add_argument(
    "--epochPath",
    action="store",
    type=str,
    required=True,
    help="Directory of epoch installation and the version",
)
parser.add_argument(
    "--numSimulations",
    action="store",
    type=int,
    required=True,
    help="Number of simulations to run",
)
parser.add_argument(
    "--campaignName",
    action="store",
    type=str,
    required=True,
    help="The campagin directory name",
)
args = parser.parse_args()

# Paths setup
script_path = Path(args.dir)
template_deck_filename = script_path / "template.deck"
template_jobscript_filename = script_path / "template.sh"
campaign_dir_name = script_path / args.campaignName
simulation_dir_paths = script_path / "paths.txt"

epoch = Path(args.epochPath)
epoch_path = epoch.parent
epoch_version = epoch.name

if args.verbose:
    print(f"Script directory: {script_path}")
    print(f"Template deck: {template_deck_filename}")
    print(f"Template jobscript: {template_jobscript_filename}")
    print(f"Simulation paths: {simulation_dir_paths}")
    print(f"Campaign folder: {campaign_dir_name}")
    print(f"Epoch version: {epoch_version}")
    print(f"Epoch location: {epoch_path}")


# INITIAL RANDOM SAMPLING
# -----------------------
with open(template_deck_filename) as f:
    deck = epydeck.load(f)

parameters = {
    "constant:background_density": {"min": 1.0e18, "max": 1.0e20, "log": True},
    "constant:frac_beam": {"min": 1.0e-4, "max": 1.0e-2, "log": True},
    "constant:b0_strength": {"min": 0.5, "max": 5.0, "log": False},
    "constant:b0_angle": {"min": 80, "max": 100, "log": False}, # Angle of B relative to x
}

# Sets up sampling of simulation and specifies number of times to run each simulation
hypercube_samples = epyscan.LatinHypercubeSampler(parameters).sample(
    args.numSimulations
)

# Takes in the folder and template and starts a counter so each new simulation gets saved to a new folder
campaign = epyscan.Campaign(deck, (script_path / campaign_dir_name))

# Randomly samples the parameter space and creates folders for each simulation
paths = [campaign.setup_case(sample) for sample in hypercube_samples]

# Save the paths to a file on separate lines
with open(simulation_dir_paths, "w") as f:
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
    campaign_path=campaign_dir_name,
    file_paths=simulation_dir_paths,
    template_path=template_jobscript_filename,
    n_runs=len(paths),
    job_name=args.campaignName,
)

print(f"Job submitted. {job.job_id}")

# job.poll_jobs(interval=2)
# _, failed_jobs = job.get_job_results()

# if failed_jobs:
#     print("The following jobs failed", failed_jobs)
#     sys.exit("Initial/Setup simulation run failed. See job log files")
