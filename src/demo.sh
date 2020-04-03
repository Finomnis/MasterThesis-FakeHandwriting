#!/bin/bash

SCRIPTPATH=$( cd $(dirname $(readlink -f $0)) ; pwd -P )

export PYTHONPATH=$PYTHONPATH:"$SCRIPTPATH"

python "$SCRIPTPATH/tools/full_pipeline.py" \
       "$SCRIPTPATH/../latex/thesis/assets/showcase/input.png" \
       --text-in "above or sinking below" \
       --text-out "but rising again"
