#!/bin/bash

parallel \
    --jobs 10 \
    """
    cd ../ ; wandb agent --count 1 lslattery/wafer-defect-detection/srwmqhek
    """ ::: $(seq 0 9)
