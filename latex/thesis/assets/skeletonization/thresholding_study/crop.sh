#!/bin/bash

mkdir -p cropped

for file in "$@"; do
    fName=$(basename "$file") 
    convert "$file" -crop 282x118+1811+7 "cropped/$fName"
done

