# SeedMarkers - Embeddable Markers for Physical Objects

## Installation

Install opencv depending on your operating system (for ArUco opencv-contrib is required)
Install python3 dependencies:  
`pip3 install -r requirements.txt`

## Usage

run `generate/generate_example_shapes.sh` for an example. PNG output can be found in `seedmarker_debug_clear`.

### DXF

The generator requires an DXF-file that either contains circles or Polylines. Arcs, splines or other primitives are not supported. Note that polyline-bulges are supported.
Note: there are a few tricks to force Polyline output. Fusion360 will use Polylines if fillets are used at the end of arc-segments.

`python3 generate.py example_shapes/circle.dxf 01232323423 --output circle_marker.dxf`

### ReacTIVision compatibility

If an orientation vector (0-360 deg) and a ReacTIVision amoeba graph is supplied (with or without the leading w/b), the generator will try to arrange the leaves in a way that results in a clear separation between white and black circles. Note that this is more error-prone on complex graphs and outlines.

See `generate/generate_reactivision.sh`

### Vertical flip

PNGs that are generated are vertically flipped (DXF coordinate system lower left corner, PIL coordinate system upper left corner). If the marker is imprinted using the DXF on the bottom of a 3D printed part the orientation is correct. If the PNG is using for inkjet-printing, vertical mirroring is required.

### 6 DoF pose estimation:

Use a graph that has at least 4 unique subtrees with only a single leaf on the last layer. See `generate/SeedmarkerSpeaker_no_cutout3_marker_print.txt` for an example. 