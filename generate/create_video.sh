#!/bin/sh

# ffmpeg -framerate 24 -i debug_treemap_clear/%05d.png debug_treemap_clear_2.mp4

ffmpeg -y -framerate 24 -pattern_type glob -i 'examples_video/debug_rect_clear/*.png' examples_video/debug_rect.mp4
ffmpeg -y -framerate 24 -pattern_type glob -i 'examples_video/debug_circle_clear/*.png' examples_video/debug_circle.mp4
ffmpeg -y -framerate 24 -pattern_type glob -i 'examples_video/debug_bone_clear/*.png' examples_video/debug_bone.mp4

# ffmpeg                                                                  \
# -i debug_treemap/%05d.png                                               \
# -i treemap/%05d.png                                                     \
# -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]'    \
# -map [vid]                                                              \
# -framerate 24                                                           \
# debug_treemap.mp4