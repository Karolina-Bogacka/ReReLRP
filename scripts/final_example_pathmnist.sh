#!/bin/bash

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && cd .. && pwd )"
SRC_DIR="$PROJECT_DIR/src"
echo "Project dir: $PROJECT_DIR"
echo "Sources dir: $SRC_DIR"

RESULTS_DIR="$PROJECT_DIR/results"
if [ "$4" != "" ]; then
    RESULTS_DIR=$4
else
    echo "No results dir is given. Default will be used."
fi
echo "Results dir: $RESULTS_DIR"

for LR in 0.1 0.05 0.01
  do
    for SEED in 0 1 2
    do
    PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name check_out_${SEED} \
    --datasets pathmnist --num-tasks 4 --network vgg11 --seed $SEED  \
    --nepochs 200 --batch-size 512 --results-path $RESULTS_DIR --lr $LR\
    --gridsearch-tasks -1 --approach lwm --gpu 0 --exp-name final_pathmnist --gradcam-layer "classifier" --num-exemplars-per-class 6 --beta 2.0 --exemplar-selection="herding"
    done
done 