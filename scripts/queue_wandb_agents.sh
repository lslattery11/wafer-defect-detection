#!/bin/bash

parallel \
    --jobs 20 \
    """
    cd ../ ; wandb agent --count 2 lslattery/wafer-defect-detection/vyss3un0
    """ ::: $(seq 0 19)
