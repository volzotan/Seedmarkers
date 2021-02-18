package de.uniweimar.mm.seedmarkerdetector;

import android.app.Application;

import androidx.annotation.NonNull;
import androidx.camera.camera2.Camera2Config;
import androidx.camera.core.CameraXConfig;

public class SeedmarkerDetector extends Application implements CameraXConfig.Provider {
    @NonNull
    @Override
    public CameraXConfig getCameraXConfig() {
        CameraXConfig config = Camera2Config.defaultConfig();
        return config;
    }
}
