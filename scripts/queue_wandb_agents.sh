#!/bin/bash

parallel \
    --jobs 10 \
    """
    cd ../ ; wandb agent --count 10 lslattery/wafer-defect-detection/m89ah3n1
    """ ::: $(seq 0 9)
