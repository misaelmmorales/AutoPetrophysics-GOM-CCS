#!/bin/bash

# Run the main script and save outputs (.out) and errors (.err)
python3 auto_baseline_correction.py 2>&1 | tee abc.out