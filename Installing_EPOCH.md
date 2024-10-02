1. `cd scratch`
2. `git clone --recursive https://github.com/Warwick-Plasma/epoch.git`
3. `cd epoch/epoch2d`
4. `nano Makefile`
5. [OPTIONAL] Remove the `#` at the start of the `#DEFINES += $(D)PHOTONS` for QED
6. `module load OpenMPI/3.1.3-GCC-8.2.0-2.31.1`
7. `make COMPILER=gfortran --debug -j4`
8. Create a new simulation folder for epoch `mkdir test1`
9. Copy an existing deck file over (e.g. `example_decks/qed_rese`) `cp example_decks/qed_rese test1/input.deck`. NOTE: deck must always be called `input.deck`
10. Navigate to the `~/scratch` directory and create a jobscript using `nano jobscript.sh` and copy in the bash script attached at the bottom of this file. NOTE: Please change the `user.email` to be your normal username
11. Finally run `sbatch jobscript.sh` and wait for completion
```bash
#!/usr/bin/env bash

#SBATCH --job-name=muon_test_1
#SBATCH --ntasks=96
#SBATCH --partition=nodes
#SBATCH --time=02-00:00:00 # Time limit (DD-HH:MM:SS)
#SBATCH --account=pet-pic-2022
#SBATCH --output=%x-%j.log
#SBATCH --mail-user=user.email@york.ac.uk
#SBATCH --mail-type=ALL # Mail events (NONE, BEGIN, END, FAIL, ALL)

# Abort if any command fails
set -e

# Purge any previously loaded modules
module purge

# Load modules
module load OpenMPI/3.1.3-GCC-8.2.0-2.31.1

# Prestart
cd epoch/epoch2d
echo Working directory: `pwd`
echo Running job on host:
echo -e '\t'`hostname` at `date`'\n'

# Script initialisation
mpiexec -n ${SLURM_NTASKS} ~/scratch/epoch/epoch2d/bin/epoch2d <<< ~/scratch/epoch/epoch2d/test1

# Job completed
echo '\n'Job completed at `date`
```