#!/bin/bash

# Loop over opt_id from 0 to 4
for opt_id in {0..4}; do
    # Loop over run_id from 0 to 14
    for run_id in {0..14}; do
        # Execute the command with nohup and run it in the background
        nohup python exp_scripts/CAI/benchmark.py $opt_id $run_id > .log/output_${opt_id}_${run_id}.log 2>&1 &
        # Optional: Add a sleep command to avoid overwhelming the system
        sleep 0.5
    done
done

# Inform the user that all commands have been dispatched
echo "All commands dispatched."
