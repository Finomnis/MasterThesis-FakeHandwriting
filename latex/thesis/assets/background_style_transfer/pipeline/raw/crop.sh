#!/bin/bash

if [ "$#" -ne 1 ]
then
  echo "No argument provided."
  exit 1
fi

convert "$1" -crop 256x256+981+139 "cropped/$1"

