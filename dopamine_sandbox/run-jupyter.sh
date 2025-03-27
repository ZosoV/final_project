#!/bin/bash
docker run -it --rm --gpus all \
  -p 8888:8888 \
  -v "$PWD":/workspace/ \
  dopamine-gpu
