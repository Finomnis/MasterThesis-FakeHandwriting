#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Please provide two arguments."
    exit 1
fi

convert $1 -crop 261x145+283+62 $2
