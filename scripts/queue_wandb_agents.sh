#!/bin/bash

parallel \
    --jobs 4 \
    """
    cd ../ ; wandb agent --count 5 lslattery/wafer-defect-detection/kp0581v2
    """ ::: $(seq 0 3)
