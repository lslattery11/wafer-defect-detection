#!/bin/bash

parallel \
    --jobs 3 \
    """
    cd ../ ; wandb agent --count 10 lslattery/wafer-defect-detection/$1
    """ ::: $(seq 0 2)
