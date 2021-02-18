package de.uniweimar.mm.seedmarkerdetector;

import android.graphics.Bitmap;
import android.graphics.Matrix;

import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;

import java.util.ArrayList;
import java.util.List;

public class Marker {

    Matrix transformationMatrix;
    List<MatOfPoint> contours;
    Bitmap bitmap;

    public Marker() {
        contours = new ArrayList<>();
    }

    public void addContour(MatOfPoint points) {
        contours.add(points);
    }

    public void setTransformationMatrix(Matrix transformationMatrix) {
        this.transformationMatrix = transformationMatrix;
    }


}
