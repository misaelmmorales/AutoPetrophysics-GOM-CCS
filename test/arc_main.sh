#!/bin/bash

# Run the main script and save outputs (.out) and errors (.err)
python3 auto_rock_classification.py --well_number 411 --n_classes 3 --method 'gmm' 2>&1 | tee arc.out