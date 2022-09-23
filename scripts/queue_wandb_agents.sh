#!/bin/bash

parallel \
    --jobs 10 \
    """
    cd ../ ; wandb agent --count 10 lslattery/wafer-defect-detection/w6gxwl7y
    """ ::: $(seq 0 9)
