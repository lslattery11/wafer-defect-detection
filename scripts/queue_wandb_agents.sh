#!/bin/bash

parallel \
    --jobs 10 \
    """
    cd ../ ; wandb agent --count 5 lslattery/wafer-defect-detection/kp0581v2
    """ ::: $(seq 0 3)
