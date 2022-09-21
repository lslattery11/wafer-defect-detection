#!/bin/bash

parallel \
    --jobs 20 \
    """
    cd ../ ; wandb agent --count 1 lslattery/wafer-defect-detection/1myjarll
    """ ::: $(seq 0 19)
