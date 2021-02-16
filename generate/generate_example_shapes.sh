#!/bin/sh

INPUT_DIR="example_shapes_video"
EXTENSION=".dxf"

TREE=01234433332332323

# exit on SIGINT by CTRL+C
trap "exit" INT

for f in $INPUT_DIR/*.dxf; do
    FILENAME=$(basename "$f" $EXTENSION)
    INPUT=$INPUT_DIR/$FILENAME$EXTENSION
    DEBUG_OUT="examples/debug_"$FILENAME
    OUTPUT=$FILENAME$EXTENSION

    echo $INPUT

    mkdir -p $DEBUG_OUT
    mkdir -p $DEBUG_OUT"_clear"
    python3 generate.py $INPUT $TREE --output $OUTPUT --debug-directory $DEBUG_OUT --line-width=1.6
done
