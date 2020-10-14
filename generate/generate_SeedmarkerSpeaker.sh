#!/bin/sh

# python3 generate.py SeedmarkerSpeaker_no_cutout3.dxf 012323234233423334233334 --output SeedmarkerSpeaker_no_cutout3_marker_engrave.dxf
# python3 generate.py SeedmarkerSpeaker_no_cutout3.dxf 012323234233423334233334 --output SeedmarkerSpeaker_no_cutout3_marker_hex.dxf --hexagon-fill-radius=0.75 --line-width=2.4
python3 generate.py SeedmarkerSpeaker_no_cutout3.dxf 012323234233423334233334 --output SeedmarkerSpeaker_no_cutout3_marker_print.dxf --line-width=2.4