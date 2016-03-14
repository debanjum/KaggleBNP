#!/bin/bash
# The name of the job, can be anything, simply used when displaying the list of running jobs
#$ -N randomforest.py
# Combining output/error messages into one file
#$ -j y
# Set memory request:
#$ -l vf=4G
# Set walltime request:
#$ -l h_rt=02:59:00
# One needs to tell the queue system to use the current directory as the working directory
# Or else the script may fail as it will execute in your top level home directory /home/username
#$ -cwd
# then you tell it retain all environment variables (as the default is to scrub your environment)
#$ -V
# Now comes the command to be executed
python randomforest.py
exit 0
