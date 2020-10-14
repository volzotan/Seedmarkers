#!/bin/bash

INPUT_DIR="failure_cases_shapes"
EXTENSION=".dxf"

files=( 
    "bone" 
    "concave_poly"
    "circle" 
    "rect_long" 
)
graphs=( 
    "0123443333233233" 
    "0123443333233233" 
    "012344444433333323" 
    "0123443333233233"
)

# exit on SIGINT by CTRL+C
trap "exit" INT

for ((i=0;i<${#files[@]};++i)); do
    FILENAME="${files[i]}"
    INPUT=$INPUT_DIR/$FILENAME$EXTENSION
    DEBUG_OUT="failure_cases/debug_"$FILENAME
    OUTPUT=$FILENAME$EXTENSION

    echo $INPUT

    mkdir -p $DEBUG_OUT
    mkdir -p $DEBUG_OUT"_clear"
    python3 generate.py $INPUT ${graphs[i]} --output $OUTPUT --debug-directory $DEBUG_OUT --line-width=3
done
