#!/bin/sh

# FILENAME="bone"
FILENAME="circle"
#FILENAME="rect"

EXTENSION=".dxf"
TREE=0123443333233232323 

INPUT="example_shapes_video/$FILENAME$EXTENSION"
DEBUG_OUT="examples_video/debug_"$FILENAME

mkdir -p $DEBUG_OUT
mkdir -p $DEBUG_OUT"_clear"

python3 generate.py $INPUT $TREE --output video_example.dxf --debug-directory $DEBUG_OUT --line-width=1.6