#!/bin/sh

CURDIR=`/bin/pwd`

INPUT=voronoi
SVG_FILE=$CURDIR/$INPUT.svg
EPS_FILE=$CURDIR/$INPUT.eps
DXF_FILE=$CURDIR/$INPUT.dxf

# /Applications/Inkscape.app/Contents/Resources/bin/inkscape --export-png $CURDIR"/output.png" $CURDIR"/output.svg"

/Applications/Inkscape.app/Contents/MacOS/inkscape $SVG_FILE --export-type="eps" # &> /dev/null
pstoedit -dt -f 'dxf:-polyaslines -mm' $EPS_FILE $DXF_FILE # &> /dev/null