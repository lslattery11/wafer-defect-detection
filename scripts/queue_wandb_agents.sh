#!/bin/bash

parallel \
    --jobs 10 \
    """
    cd ../ ; wandb agent --count 10 lslattery/wafer-defect-detection/$1
    """ ::: $(seq 0 9)
