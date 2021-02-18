package de.uniweimar.mm.seedmarkerdetector;

import android.annotation.SuppressLint;
import android.app.Application;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Rect;
import android.hardware.camera2.CaptureRequest;
import android.util.Log;
import android.util.Range;
import android.util.Size;

import androidx.annotation.NonNull;
import androidx.camera.camera2.interop.Camera2Interop;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class CameraControllerX {

    private String TAG = CameraControllerX.class.getSimpleName();
    private Context context;

    private Detector detector;
    private DrawSurface drawSurface;

    private PreviewView previewView;
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;

    public CameraControllerX(Context context, PreviewView previewView, DrawSurface drawSurface) {
        this.context = context;

        this.drawSurface = drawSurface;

        this.previewView = previewView;
        this.cameraProviderFuture = ProcessCameraProvider.getInstance(context);

        cameraProviderFuture.addListener(new Runnable() {
            @Override
            public void run() {
                try {
                    ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                    bind(cameraProvider);
                } catch (ExecutionException | InterruptedException e) {
                    // No errors need to be handled for this Future.
                    // This should never be reached.
                }
            }
            }, ContextCompat.getMainExecutor(context));
    }

    public void changeDetectionMode() {
        if (detector == null) {return;}

        if (detector.kernel.rows() == Detector.HIFI) {
            detector.changeKernelSize(Detector.LOFI);
        } else {
            detector.changeKernelSize(Detector.HIFI);
        }
    }

    void bind(ProcessCameraProvider cameraProvider) {

        // Camera Selector

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        // Preview

        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        // Analysis

        Size size = new Size(1280, 720);

        List<String> descriptors = new ArrayList<>();

        // hardcoded Seedmarker descriptors
        descriptors.add("012343332343323432342323|136.73:24.71:8.25|102.87:19.63:6.54|115.11:32.91:7.65|152.39:41.88:7.10|98.91:54.16:2.87|84.78:81.76:4.47|98.07:78.47:4.90|167.81:75.73:6.39|182.79:60.37:4.27|179.36:24.95:4.16|16.67:82.06:3.66|16.67:16.78:3.71");
        descriptors.add("0123423222");
        descriptors.add("0123232222");
        descriptors.add("0123423232");

        this.detector = new Detector(size, descriptors);

        ImageAnalysis.Builder builder = new ImageAnalysis.Builder();
        builder.setTargetResolution(size);
        builder.setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST);

        // hacky FPS enforcement
        Camera2Interop.Extender ext = new Camera2Interop.Extender<>(builder);
        //ext.setCaptureRequestOption(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF);
        ext.setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, new Range<Integer>(15, 30));

        ImageAnalysis imageAnalysis = builder.build();

        // Executor executor = ContextCompat.getMainExecutor(context);
        Executor executor = Executors.newSingleThreadExecutor();

        imageAnalysis.setAnalyzer(executor, new ImageAnalysis.Analyzer() {
            @Override
            public void analyze(@NonNull ImageProxy image) {
                long startTime = System.nanoTime();

                Detector.DetectorResult detectorResult = detector.run(image);
                drawSurface.addMarkers(detectorResult.markers);
                drawSurface.addLines(detectorResult.contours);

                long stopTime = System.nanoTime();
                //System.out.println(String.format("%.3f fps | %d", 1/((stopTime - startTime)/1000000000.0), detectorResult.markers.size()));

                image.close();
            }
        });

        cameraProvider.bindToLifecycle((LifecycleOwner) this.context, cameraSelector, imageAnalysis, preview);

    }
}
