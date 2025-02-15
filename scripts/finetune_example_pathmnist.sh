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

for ACC in 1 5 15
do 
    PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name check_out_${SEED} \
    --datasets pathmnist --num-tasks 4 --network vgg11 --seed 0  \
    --nepochs 200 --batch-size 512 --results-path $RESULTS_DIR --lr 0.1 --acc_drop $ACC\
    --gridsearch-tasks -1 --approach rere_lrp --gpu 0 --exp-name finetune_pathmnist  --num-exemplars-per-class 132 --exemplar-selection="herding"
done 