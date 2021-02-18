package de.uniweimar.mm.seedmarkerdetector;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.ViewPort;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.Lifecycle;
import androidx.lifecycle.LifecycleOwner;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.SurfaceTexture;
import android.os.Bundle;
import android.util.Log;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.TextureView;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Toast;

public class MainActivity extends AppCompatActivity implements TextureView.SurfaceTextureListener, GestureDetector.OnGestureListener {

    private static final String TAG = MainActivity.class.getSimpleName();

    public static final int PERMISSION_REQUEST_CODE = 0x123;

    private CameraControllerX cameraController;
    // DrawSurface drawSurface;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // make the activity fullscreen (removes the top bar with clock and battery info)
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);

        setContentView(R.layout.activity_main);

        // tell the TextureView we want to know if the GPU has initialized it
        // TextureView tv = (TextureView) findViewById(R.id.textureView);
        // tv.setSurfaceTextureListener(this);
        // drawSurface = (DrawSurface) findViewById(R.id.drawSurface);

        // ask for permissions

        if (!checkPermissionsAreGiven()) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO}, PERMISSION_REQUEST_CODE);
        } else {
            Log.d(TAG, "permissions are already granted");
        }

        // do not do anything else in onCreate(). init() will be called once the permissions are granted.
    }

    public boolean checkPermissionsAreGiven() {
        return (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED &&
                ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String permissions[], int[] grantResults) {
        switch (requestCode) {
            case PERMISSION_REQUEST_CODE: {
                boolean success = true;

                for (int i=0; i<grantResults.length; i++) {

                    // even though only still image permissions are required, video/audio is part of the
                    // permission package. But since android never displays the audio request, it is denied.

                    if (permissions[i].equals(Manifest.permission.RECORD_AUDIO)) {
                        continue;
                    }

                    if (grantResults[i] != PackageManager.PERMISSION_GRANTED) {
                        success = false;
                        break;
                    }
                }

                if (success) {
                    Log.w(TAG, "permissions are granted by user");
                    init();
                } else {
                    Log.w(TAG, "permissions denied by user");
                }
            }
        }
    }

    // permissions are granted, setup everything
    private void init() {

        PreviewView pv = findViewById(R.id.previewView);
        this.cameraController = new CameraControllerX(this, pv, findViewById(R.id.drawSurface));

//        @SuppressLint("UnsafeExperimentalUsageError")
//        ViewPort vp = pv.getViewPort();
//        vp.getAspectRatio();
//        vp.getLayoutDirection();
//        vp.getRotation();
//        vp.getScaleType();




        // the textureView may not be instantly available
        // if we can use it we use it, otherwise we will call
        // this method again from the available-callback

//        TextureView tv = (TextureView) findViewById(R.id.textureView);
//        if (tv.isAvailable()) {
//
//            cameraController = new CameraControllerX(this, tv);
//
//            try {
//                Log.d(TAG, "opening camera...");
//                cameraController.openCamera();
//            } catch (Exception e) {
//                Log.e(TAG, "open camera failed", e);
//            }
//        } else {
//            Log.w(TAG, "textureView not (yet) available");
//        }
    }

    @Override
    public void onSurfaceTextureAvailable(SurfaceTexture surfaceTexture, int i, int i1) {

        Log.d(TAG, "textureView is available");

        if (checkPermissionsAreGiven()) {
            init();
        } else {
            Log.e(TAG, "camera inactive : permissions are missing");
            Toast.makeText(this, "camera inactive : permissions are missing", Toast.LENGTH_LONG).show();
        }
    }

    @Override
    public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {}

    @Override
    public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
        return false;
    }

    @Override
    public void onSurfaceTextureUpdated(SurfaceTexture surface) {}

    @Override
    public boolean onDown(MotionEvent e) {
        cameraController.changeDetectionMode();
        return true;
    }

    @Override
    public void onShowPress(MotionEvent e) {

    }

    @Override
    public boolean onSingleTapUp(MotionEvent e) {
        return false;
    }

    @Override
    public boolean onScroll(MotionEvent e1, MotionEvent e2, float distanceX, float distanceY) {
        Log.d(TAG, "scroll");
        return false;
    }

    @Override
    public void onLongPress(MotionEvent e) {

    }

    @Override
    public boolean onFling(MotionEvent e1, MotionEvent e2, float velocityX, float velocityY) {
        Log.d(TAG, "fling");
        return true;
    }
}
