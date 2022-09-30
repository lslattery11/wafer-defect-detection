#!/bin/bash

parallel \
    --jobs 15 \
    """
    cd ../ ; wandb agent --count 10 lslattery/wafer-defect-detection/$1
    """ ::: $(seq 0 14)
