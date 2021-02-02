# SeedMarkers - Embeddable Markers for Physical Objects

![](media/example_shapes.png)

## Installation

Install opencv depending on your operating system (for ArUco opencv-contrib is required)
Install python3 dependencies:  
`pip3 install -r requirements.txt`

## Quickstart

Install the requirements as stated above and run `generate_example_shapes.sh` in the `generate` directory. 
PNG output can be found in `seedmarker_debug_clear`.

## Manufacturing

![](media/manufacturing.png)

Emboss with 1 layer height on object and orient this face towards the buildplate when printing. Configure "pause at Z" in your slicing software and let the printer pause and eject filament after the first layer. Change filament and continue.

For lasercutting post-process the DXF (conversion to SVG may be necessary) according to the needs of your lasercutters software package. Sometimes colouring the lines with a certain color is necessary, sometimes manual selection of polygons for engraving needs to be done.

## Applications

![](media/applications.png)
 
See the paper for more details.

## Detection

...

## Details

### DXF

The generator requires an DXF-file that either contains circles or Polylines. Arcs, splines or other primitives are not supported (polyline-bulges are supported).
Note: there are a few tricks to force Polyline output. Fusion360 will use Polylines if fillets are used at the end of arc-segments.

`python3 generate.py example_shapes/circle.dxf 01232323423 --output circle_marker.dxf`

### ReacTIVision compatibility

If an orientation vector (0-360 deg) and a ReacTIVision amoeba graph is supplied (with or without the leading w/b), the generator will try to arrange the leaves in a way that results in a clear separation between white and black circles. Note that this is more error-prone on complex graphs and outlines. Some geometries will work well, some will terminate early.

See `generate/generate_reactivision.sh`

### Vertical flip

PNGs that are generated are vertically flipped (DXF coordinate system lower left corner, PIL coordinate system upper left corner). If the marker is imprinted using the DXF on the bottom of a 3D printed part the orientation is correct. If the PNG is using for inkjet-printing, vertical mirroring is required.

### 6 DoF pose estimation:

Use a graph that has at least 4 unique subtrees with only a single leaf on the last layer. See `generate/SeedmarkerSpeaker_no_cutout3_marker_print.txt` for an example. The detector requires a OpenCV-style calibration matrix in a JSON-file (see `detect/iphonexs_calibration.json`).