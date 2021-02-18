package de.uniweimar.mm.seedmarkerdetector;

import android.graphics.Matrix;
import android.util.Log;
import android.util.Size;

import androidx.camera.core.ImageProxy;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.RotatedRect;
import org.opencv.imgproc.Imgproc;

import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import static org.opencv.core.CvType.CV_32F;
import static org.opencv.core.CvType.CV_64F;
import static org.opencv.core.CvType.CV_8UC1;

public class Detector {

    public static String TAG = Detector.class.getSimpleName();

    public static int LOFI = 2;
    public static int HIFI = 3;

    Mat cameraMatrix;
    MatOfDouble distCoeffs;

    Mat hierarchy;
    List<MatOfPoint> contours;

    Mat mYuv;
    Mat mGray;
    byte[] nv21;

    Mat kernel;

    HashMap<String, List<Double[]>> markerDescriptors = new HashMap<>();

    public Detector(Size size, List<String> descriptors) {
        System.loadLibrary("opencv_java3");

        mYuv = new Mat(size.getHeight() + size.getHeight() / 2, size.getWidth(), CV_8UC1);
        mGray = new Mat(size.getHeight(), size.getWidth(), CV_8UC1);
        nv21 = new byte[size.getHeight() * size.getWidth() * 2];

        hierarchy = new Mat();
        contours = new ArrayList();

        kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new org.opencv.core.Size(LOFI, LOFI));

        for (String desc : descriptors) {
            String markerName = desc.split("\\|")[0];

            if (desc.length() > markerName.length()) {
                markerDescriptors.put(markerName, parseDescriptor(desc));
            }
            else {
                markerDescriptors.put(markerName, null);
            }
        }

        float scaling = size.getWidth() / 1920f;

        cameraMatrix = new Mat(7, 7, CV_64F);

        cameraMatrix.put(0, 0, 1884.5618000288894 * scaling); // fx
        cameraMatrix.put(0, 1, 0.0);
        cameraMatrix.put(0, 2, 911.7922321737778 * scaling);  // cx

        cameraMatrix.put(1, 0, 0.0);
        cameraMatrix.put(1, 1, 1880.876139665315 * scaling);  // fy
        cameraMatrix.put(1, 2, 534.5826018179368 * scaling);  // cy

        cameraMatrix.put(2, 0, 0.0);
        cameraMatrix.put(2, 1, 0.0);
        cameraMatrix.put(2, 2, 1.0);

        double[] values = {0.3550507363758069, -2.2889560633704336, -0.002343699246072931, -0.006808603240379938, 5.479282743032857};
        distCoeffs = new MatOfDouble();
        distCoeffs.fromArray(values);

    }

    public void changeKernelSize(int size) {
        Log.d(TAG, String.format("changing kernel size to %d", size));
        kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new org.opencv.core.Size(size, size));
    }

    private void convertToGrayscale(ImageProxy image) {

        if (image == null){
            return;
        }

        ByteBuffer yBuffer = image.getPlanes()[0].getBuffer();
        ByteBuffer uBuffer = image.getPlanes()[1].getBuffer();
        ByteBuffer vBuffer = image.getPlanes()[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        //U and V are swapped
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        mYuv.put(0, 0, nv21);
        Imgproc.cvtColor(mYuv, mGray, Imgproc.COLOR_YUV2GRAY_NV21, 3);
    }

    private List<Node> traverseGraph(List<MatOfPoint> contours, Mat hierarchy, int index, Node parent, int depth) {

        List<Node> nodeList = new ArrayList<>();

        if (depth > 1000){
            return nodeList;
        }

        int next_e      = (int) hierarchy.get(0, index)[0];
        int prev_e      = (int) hierarchy.get(0, index)[1];
        int child_e     = (int) hierarchy.get(0, index)[2];
        int parent_e    = (int) hierarchy.get(0, index)[3];

        Node n = new Node(index, parent);
        nodeList.add(n);

        if (child_e > -1) {
            n.addChildren(traverseGraph(contours, hierarchy, child_e, n, depth++)); // children
        }

        if (next_e > -1) {
            nodeList.addAll(traverseGraph(contours, hierarchy, next_e, parent, depth++)); // siblings
        }

        // guarantee LHD order for all children

        Collections.sort(n.getChildren());

        return nodeList;
    }

    private List<Node> parseContourHierarchy(List<MatOfPoint> contours, Mat hierarchy) {

        // List<Node> topLevelNodes = new ArrayList<>();
        List<Integer> indicesVisitedGlobally = new ArrayList<>();

        if (contours == null || contours.size() == 0 || hierarchy == null || hierarchy.rows() == 0) {
            return new LinkedList<Node>();
        }

        List<Node> topLevelNodes = traverseGraph(contours, hierarchy, 0, null, 0);

//        for (int c = 0; c < hierarchy.cols(); c++) {
//
//            if (hierarchy.get(0, c)[3] >= 0) {
//                continue;
//            }
//
//            topLevelNodes.add(traverseGraph(contours, hierarchy, 0, null, 0).get(0));
//        }

        return topLevelNodes;
    }

    private List<Double[]> parseDescriptor(String descriptor) {

        List<Double[]> results = new ArrayList<>();

        String[] elements = descriptor.split("\\|");

        for (int i=1; i<elements.length; i++) {
            String[] coords = elements[i].split("\\:");

            Double[] c = new Double[3];
            c[0] = Double.parseDouble(coords[0]);
            c[1] = Double.parseDouble(coords[1]);
            c[2] = Double.parseDouble(coords[2]);

            results.add(c);
        }

        return results;
    }

    private void matchLeaves(List<Double[]> descriptorCircles, Node marker, MatOfPoint3f objectPoints, MatOfPoint2f imagePoints) {

        marker.computeUniques(true);
        List<Node> leaves = marker.getLeaves();

        List<Point3> objPointsList = new ArrayList<>();
        List<Point> imgPointsList = new ArrayList<>();

        for (int i=0; i<leaves.size(); i++) {
            Node detectedLeaf = leaves.get(i);

            if (!detectedLeaf.isUnique) {
                continue;
            }

            MatOfPoint leafContour = this.contours.get(detectedLeaf.contourIndex);

            if (leafContour.rows() > 5) { // min amount of points to fit ellipse

                Double[] circle = descriptorCircles.get(i);
                objPointsList.add(new Point3(circle[0], circle[1], 0));

                MatOfPoint2f leafContour2f = new MatOfPoint2f();
                leafContour.convertTo(leafContour2f, CvType.CV_32F);
                RotatedRect ellipse = Imgproc.fitEllipse(leafContour2f);

                imgPointsList.add(ellipse.center);
            }

        }

        objectPoints.fromList(objPointsList);
        imagePoints.fromList(imgPointsList);
    }


    public DetectorResult run(ImageProxy image) {

        convertToGrayscale(image);
        image.close();

        //Imgproc.threshold(mGray, mGray, 0, 255, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);
        //Imgproc.adaptiveThreshold(mGray, mGray, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 83, 2);
        Imgproc.adaptiveThreshold(mGray, mGray, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 79, 2); //127, 2);

        Imgproc.morphologyEx(mGray, mGray, Imgproc.MORPH_OPEN, kernel);
        Imgproc.morphologyEx(mGray, mGray, Imgproc.MORPH_CLOSE, kernel);

        //hierarchy = new Mat();
        contours = new ArrayList();

        Imgproc.findContours(mGray, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        List<Node> topLevelNodes = parseContourHierarchy(contours, hierarchy);
        // topLevelNodes = filterForCandidates(topLevelNodes);

        List<Marker> markers = new ArrayList<>();

        for (Node tln : topLevelNodes) {

            List<Node> foundMarkers = new ArrayList<>();

            for (String name : markerDescriptors.keySet()) {
                foundMarkers.addAll(tln.findSubtree(name));
            }

            for (Node foundMarker : foundMarkers) {

                Marker m = new Marker();

                for (Integer i : foundMarker.getAllContourIndices(new ArrayList<Integer>())) {
//                    MatOfPoint2f contour2f = new MatOfPoint2f();
//                    contours.get(i).convertTo(contour2f, CvType.CV_32F);
//                    Imgproc.approxPolyDP(contour2f, contour2f, 1, true);
//                    m.addContour(contour2f);

                    // deep copy wont work (the drawing function may scale a MatOfPoint twice (once because it's a marker and once because it is in contours)
                    m.addContour(contours.get(i));
                }

                if (markerDescriptors.get(foundMarker.getLHDSname(0)) == null) {
                    // no pose info

                    markers.add(m);
                    continue;
                }

                MatOfPoint3f objectPoints = new MatOfPoint3f();
                MatOfPoint2f imagePoints = new MatOfPoint2f();

                matchLeaves(markerDescriptors.get(foundMarker.getLHDSname(0)), foundMarker, objectPoints, imagePoints);

                // get 4x3 pose Matrix from solvePnP / ignore planar constraint

//                Mat rvec = new Mat(3, 3, CV_32F);
//                Mat tvec = new Mat(3, 3, CV_32F);
//
//                Calib3d.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
//
//                List<Point3> poseEstimationPoints = new ArrayList<Point3>();
//
//                poseEstimationPoints.add(new Point3(0, 110, 0));
//                poseEstimationPoints.add(new Point3(200, 110, 0));
//                poseEstimationPoints.add(new Point3(200, 110+60, 0));
//                poseEstimationPoints.add(new Point3(0, 110+60, 0));
//
//                MatOfPoint3f poseEstimationPointsMat = new MatOfPoint3f();
//                poseEstimationPointsMat.fromList(poseEstimationPoints);
//
//                MatOfPoint2f poseEstimationImagePoints = new MatOfPoint2f();
//
//                Calib3d.projectPoints(poseEstimationPointsMat, rvec, tvec, cameraMatrix, distCoeffs, poseEstimationImagePoints);

//                Mat rmat = new Mat(3, 3, CV_64F);
//                Mat rtmat = new Mat(4, 4, CV_64F);
//                Mat transMat = new Mat(3, 4, CV_64F);
//                Calib3d.Rodrigues(rvec, rmat);
//
//                List<Mat> lm = new ArrayList<>();
//                lm.add(rmat);
//                lm.add(tvec);
//
//                Core.hconcat(lm, rtmat);
//                rtmat.push_back(new Mat(1, 4, CV_64F));
//                rtmat.put(3, 3, 1);
//
//                Mat cameraMatrixE = new Mat(3, 4, CV_64F);
//                lm = new ArrayList<>();
//                lm.add(cameraMatrix);
//                lm.add(new Mat(3, 1, CV_64F));
//                Core.hconcat(lm, cameraMatrixE);
//
//                Matrix rt = new Matrix();
//                Core.gemm(cameraMatrixE, rtmat, 1, new Mat(), 0, transMat);
//
//                Log.d(TAG, transMat.dump());
//
//                transMat.convertTo(transMat, CV_32F);
//
//                float[] opencv_matrix_values = new float[transMat.cols() * transMat.rows()];
//                transMat.get(0, 0, opencv_matrix_values);
//                rt.setValues(opencv_matrix_values);
//                m.setTransformationMatrix(rt);

                // get 3x3 transformation Matrix from findHomography

                MatOfPoint2f objectPoints2f = new MatOfPoint2f();
                List<Point> objectPoints2fList = new ArrayList<>();

                for (Point3 p : objectPoints.toList()) {
                    objectPoints2fList.add(new Point(p.x, -p.y)); // DXF/raster coordinate system switch
                }

                objectPoints2f.fromList(objectPoints2fList);

                Mat h = Calib3d.findHomography(objectPoints2f, imagePoints, 0); //Calib3d.RANSAC);
                h.convertTo(h, CV_32F);
                Matrix android_h = new Matrix();

                float[] opencv_matrix_values = new float[h.cols() * h.rows()];
                h.get(0, 0, opencv_matrix_values);
                android_h.setValues(opencv_matrix_values);
                m.setTransformationMatrix(android_h);
                android_h.preTranslate(0, -170);

                markers.add(m);

                // markerContours.add(poseEstimationImagePoints);
            }

//            String lhds = tln.getLHDSname(0);
//            if (lhds.length() > 3) {
//                Log.i(TAG, lhds);
//            }
        }

        // Log.wtf(TAG, Integer.toString(topLevelNodes.size()));

        // return new DetectorResult(new ArrayList<MatOfPoint>(), markers);
        return new DetectorResult(contours, markers);
    }

//    private List<Node> filterForCandidates(List<Node> topLevelNodes) {
//        List<Node> candidates = new ArrayList<>();
//
//        for (Node tln : topLevelNodes) {
//            tln.getChildren().size()
//        }
//    }

    class Node implements Comparable<Node>{

        public int contourIndex;
        public Node parent;
        public List<Node> children;
        public boolean isUnique;

        public Node(int contourIndex, Node parent) {
            this.contourIndex = contourIndex;
            this.parent = parent;
            this.children = new ArrayList<>();
        }

        public void addChildren(List<Node> children) {
            this.children.addAll(children);
        }

        public List<Node> getChildren() {
            return children;
        }

        public int getContourIndex() {
            return contourIndex;
        }

        public Node getRoot() {
            if (this.parent == null) {
                return this;
            }

            return this.parent.getRoot();
        }

        public int getDepth(int depth) {
            if (this.parent == null) {
                return depth;
            }

            return this.parent.getDepth(depth+1);
        }

        public String toString() {
            return this.getLHDSname(this.getDepth(0));
        }

        public List<Integer> getAllContourIndices(List<Integer> indices) {
            indices.add(contourIndex);

            for (Node c : children) {
                c.getAllContourIndices(indices);
            }

            return indices;
        }

        public String getLHDSname(int depth) {
            StringBuilder sb = new StringBuilder();

            sb.append(depth);

            for (Node c : this.children) {
                sb.append(c.getLHDSname(depth+1));
            }

            return sb.toString();
        }

        public List<Node> findSubtree(String lhds) {
            List<Node> results = new ArrayList();

//            if (this.getChildren().size() < 2) {
//                return results;
//            }

            String thisName = this.getLHDSname(0);

            if (thisName.equals(lhds)) {
                results.add(this);
            } else {
                for (Node c : this.getChildren()) {
                    results.addAll(c.findSubtree(lhds));
                }
            }

            return results;
        }

        public void computeUniques(boolean unique) {
            this.isUnique = unique;

            if (!unique) {
                for (Node c : this.getChildren()) {
                    c.computeUniques(false);
                }
            } else {
                HashMap<String, List<Node>> childrenNamesHashed = new HashMap<>();

                for (Node c : this.getChildren()) {
                    String name = c.getLHDSname(0);

                    if (!childrenNamesHashed.containsKey(name)) {
                        List<Node> entries = new ArrayList<>();
                        entries.add(c);
                        childrenNamesHashed.put(name, entries);
                    } else {
                        childrenNamesHashed.get(name).add(c);
                    }
                }

                for (List<Node> entries : childrenNamesHashed.values()) {
                    boolean childrenUnique = false;
                    if (entries.size() == 1) {
                        childrenUnique = true;
                    }
                    for (Node c : entries) {
                        c.computeUniques(childrenUnique);
                    }
                }
            }
        }

        public List<Node> getLeaves() {
            List<Node> leaves = new ArrayList<>();

//            if (!this.isUnique) {
//                return leaves;
//            }

            if (this.getChildren().size() == 0) {
                leaves.add(this);
            } else {
                for (Node c : this.getChildren()) {
                    leaves.addAll(c.getLeaves());
                }
            }

            return leaves;
        }

        @Override
        public int compareTo(Node o) {

            String thisName = this.getLHDSname(0);
            String otherName = o.getLHDSname(0);

            if (thisName.length() > otherName.length()) {
                return -1;
            } else if (thisName.length() < otherName.length()) {
                return 1;
            }

            // noise. No comparisons of subtrees necessary
            if (thisName.length() > 10) {
                return 0;
            }

            if (Long.parseLong(thisName) > Long.parseLong(otherName)) {
                return -1;
            } else if (Long.parseLong(thisName) < Long.parseLong(otherName)) {
                return 1;
            }

            return 0;
        }
    }

    class DetectorResult {

        List<MatOfPoint> contours;
        List<Marker> markers;

        public DetectorResult(List<MatOfPoint> contours, List<Marker> markers) {
            this.contours = contours;
            this.markers = markers;
        }
    }

}
