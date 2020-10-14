import ezdxf

class WriterSimpleDXF():

    def __init__(self, filename, dimensions=None, unit="px"):

        if not filename.endswith(".dxf"):
            filename += ".dxf"

        self.filename   = filename
        self.dimensions = dimensions
        self.unit       = unit    

        self.paths      = []

    def add_path(self, path_coords, strokewidth=1, stroke=[255, 0, 0], fill=[255, 0, 0], opacity=1.0, close_path=True):
        self.paths.append([path_coords, strokewidth, stroke, fill, opacity, close_path])

    def save(self):
        doc = ezdxf.new('R2010') 

        doc.header['$MEASUREMENT']  = 1 # metric
        doc.header['$LUNITS']       = 2 # decimal (default)
        doc.header['$INSUNITS']     = 4 # millimeters

        msp = doc.modelspace()  # add new entities to the modelspace

        num_lines = 0
        for p in self.paths:

            path_coords = p[0]
            strokewidth = p[1]
            stroke = p[2]
            fill = p[3]
            opacity = p[4]
            close_path = p[5]

            if close_path:
                path_coords.append(path_coords[0])

            msp.add_lwpolyline(path_coords) 

            num_lines += len(path_coords)

        print("written {} polylines to: {}".format(num_lines, self.filename))

        doc.saveas(self.filename)

class WriterSimpleSVG():

    def __init__(self, filename, dimensions=None, unit="px", image=None):

        if not filename.endswith(".svg"):
            filename += ".svg"

        self.filename   = filename
        self.dimensions = dimensions
        self.unit       = unit
        self.image      = image

        self.hexagons   = []
        self.circles    = []
        self.rectangles = []
        self.paths      = []

    def add_hexagons(self, hexagons, fills):
        for i in range(0, len(hexagons)):
            self.hexagons.append([
                Hexbin.create_svg_path(hexagons[i], absolute=True), 
                [fills[i][0]*255, fills[i][1]*255, fills[i][2]*255, fills[i][3]]
            ]) 

    def add_circles(self, circles, radius=3, fill=[255, 0, 0]):
        for item in circles:
            self.circles.append([item, radius, fill])


    def add_rectangles(self, coords, strokewidth=1, stroke=[255, 0, 0], opacity=1.0):
        for item in coords:
            self.rectangles.append([item, strokewidth, stroke, opacity])


    def add_path(self, path_coords, strokewidth=1, stroke=[255, 0, 0], fill=[255, 0, 0], opacity=1.0, close_path=True):
        self.paths.append([path_coords, strokewidth, stroke, fill, opacity, close_path])


    def save(self):
        with open(self.filename, "w") as out:

            out.write("<?xml version=\"1.0\" encoding=\"utf-8\" ?>")
            out.write("<?xml-stylesheet href=\"style.css\" type=\"text/css\" title=\"main_stylesheet\" alternate=\"no\" media=\"screen\" ?>")
            if self.dimensions is not None:
                out.write("<svg baseProfile=\"tiny\" version=\"1.2\" width=\"{}{}\" height=\"{}{}\" ".format(self.dimensions[0], self.unit, self.dimensions[1], self.unit))
            else:
                out.write("<svg baseProfile=\"tiny\" version=\"1.2\" ")
            out.write("xmlns=\"http://www.w3.org/2000/svg\" xmlns:ev=\"http://www.w3.org/2001/xml-events\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" >")
            out.write("<defs />")
            if self.image is not None:
                out.write("<image x=\"0\" y=\"0\" xlink:href=\"{}\" />".format(self.image))

            for h in self.hexagons:
                out.write("<path d=\"")
                for cmd in h[0]:
                    out.write(cmd[0])
                    if (len(cmd) > 1):
                        out.write(str(cmd[1]))
                        out.write(" ")
                        out.write(str(cmd[2]))
                        out.write(" ")
                # out.write("\" fill=\"rgba({},{},{},{})\" />".format(int(h[1][0]), int(h[1][1]), int(h[1][2]), int(h[1][3])))
                out.write("\" fill=\"rgb({},{},{})\" fill-opacity=\"{}\" />".format(int(h[1][0]), int(h[1][1]), int(h[1][2]), h[1][3]))
 
            for c in self.circles:
                out.write("<circle cx=\"{}\" cy=\"{}\" fill=\"rgb({},{},{})\" r=\"{}\" />".format(c[0][0], c[0][1], c[2][0], c[2][1], c[2][2], c[1]))

            for r in self.rectangles:
                out.write("<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" stroke-width=\"{}\" stroke=\"rgb({},{},{})\" fill-opacity=\"0.0\" stroke-opacity=\"{}\" />".format(*r[0], r[1], *r[2], r[3]))

            for p in self.paths:
                path_coords = p[0]
                strokewidth = p[1]
                stroke = p[2]
                fill = p[3]
                opacity = p[4]
                close_path = p[5]

                if fill is None:
                    fill = "none"

                out.write("<path d=\"")
                for i in range(0, len(path_coords)):
                    if i == 0:
                        out.write("M")
                    else:
                        out.write("L")
                    out.write("{} {} ".format(*path_coords[i]))
                    if close_path:
                        if i == len(path_coords)-1:
                            out.write(" Z")

                out.write("\" stroke-width=\"{}\" stroke=\"{}\" fill=\"{}\" opacity=\"{}\" />".format(
                    strokewidth, stroke, fill, opacity))

            out.write("</svg>")