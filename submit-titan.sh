#!/bin/bash
# Identifies the best titan-machine to run a command on and executes that command remotely. Aliases must live in .bashrc in order to be respected. Best GPU should be identified by the python script.
results=$(python ~/titan_utils/titan_utils.py)
IFS=$'\n'; results=($results); unset IFS;
best_titan=${results[0]}
loc=$(pwd)
echo "Executing \`${@}\` on ${best_titan}"
ssh -t $USER@$best_titan "/bin/bash -ci \"cd ${loc}; ${@}\""
