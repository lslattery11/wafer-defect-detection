#!/bin/bash

parallel \
    --jobs 2 \
    """
    cd ../ ; wandb agent --count 10 lslattery/wafer-defect-detection/1myjarll
    """ ::: $(seq 0 1)
