#!/bin/bash

SCRIPTPATH=$( cd $(dirname $(readlink -f $0)) ; pwd -P )

if [ $# -ne 3 ]; then
    echo "Please provide two arguments."
    exit 1
fi

FILENAME=$1
CROPDATA=$2
CROPID=$3

#convert 0002-1-4_sharp_nn.png -crop 344x132+722+0 0002-1-4_sharp_nn_crop.png

convert "$SCRIPTPATH/${FILENAME}_original.png" -crop $CROPDATA "$SCRIPTPATH/${FILENAME}_original_crop_$CROPID.png"
convert "$SCRIPTPATH/${FILENAME}_skel.png"     -crop $CROPDATA "$SCRIPTPATH/${FILENAME}_skel_crop_$CROPID.png"
convert "$SCRIPTPATH/${FILENAME}_sharp.png"    -crop $CROPDATA "$SCRIPTPATH/${FILENAME}_sharp_crop_$CROPID.png"

