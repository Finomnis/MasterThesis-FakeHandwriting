#!/bin/bash

SCRIPTPATH=$( cd $(dirname $(readlink -f $0)) ; pwd -P )

if [ $# -ne 1 ]; then
    echo "Please provide two arguments."
    exit 1
fi

DATASET_PATH="$SCRIPTPATH/../../../../../../ext/datasets/cvl"

cp "$DATASET_PATH/original_png/$1.png" "$SCRIPTPATH/$1_original.png"
cp "$DATASET_PATH/skeletons_nn/$1.png" "$SCRIPTPATH/$1_skel.png"
cp "$DATASET_PATH/skeletons_nn_sharp/$1.png" "$SCRIPTPATH/$1_sharp.png"
