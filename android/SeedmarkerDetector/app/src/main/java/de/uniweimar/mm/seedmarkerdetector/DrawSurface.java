package de.uniweimar.mm.seedmarkerdetector;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.PorterDuff;
import android.util.AttributeSet;
import android.util.Log;
import android.util.Size;
import android.view.GestureDetector;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;

import java.util.ArrayList;
import java.util.List;

public class DrawSurface extends SurfaceView implements SurfaceHolder.Callback {

    private static final String TAG = DrawSurface.class.getSimpleName();

    private DrawSurfaceCallback onReadyCallback;
    private boolean interactive = false;

    private SurfaceHolder holder = null;
    private final Context context;
    private GestureDetector gestureDetector;

    public Paint paintWhite = null;
    public Paint paintGreen = null;
    public Paint paintRed = null;
    public Paint paintShine = null;

    private List<MatOfPoint> contours;
    List<Marker> markers = new ArrayList<>();

    // 640 x 480
    //private Scalar scale = new Scalar(3.27, 3.27);
    //private float[] offset = {-3, -240};

    // 1280 x 720
    private Scalar scale = new Scalar(1.65, 1.65);
    private float[] offset = {-10, -49};

    // 1920 x 1080
    // private Scalar scale = new Scalar(1.095, 1.095);
    // private float[] offset = {-5, -49};

    public DrawSurface(Context context) {
        super(context);
        this.context = context;
        init();
    }

    public DrawSurface(Context context, AttributeSet attrs) {
        super(context, attrs);
        this.context = context;

        init();
    }

    public DrawSurface(Context context, AttributeSet attrs, int defStyle) {
        super(context, attrs, defStyle);
        this.context = context;
        init();
    }

    private void init() {

        if (this.gestureDetector == null) {
            this.gestureDetector = new GestureDetector(this.context, (MainActivity) this.context);
            this.setClickable(true);
        }

        holder = getHolder();
        holder.addCallback(this);
        holder.setFormat(PixelFormat.TRANSPARENT);

        this.setZOrderOnTop(true);
        this.setWillNotDraw(false);

        paintWhite = new Paint(Paint.ANTI_ALIAS_FLAG);
        paintWhite.setColor(Color.argb(5, 255, 255, 255));
        paintWhite.setStyle(Paint.Style.STROKE);
        paintWhite.setStrokeWidth(2.0f);

        paintGreen = new Paint(Paint.ANTI_ALIAS_FLAG);
        paintGreen.setColor(Color.argb(255, 0, 255, 0));
        paintGreen.setStyle(Paint.Style.STROKE);
        paintGreen.setStrokeWidth(2.0f);

        paintRed = new Paint(Paint.ANTI_ALIAS_FLAG);
        paintRed.setColor(Color.GREEN);
        paintRed.setStyle(Paint.Style.STROKE);
        paintRed.setStrokeWidth(paintGreen.getStrokeWidth());

        paintShine = new Paint(Paint.ANTI_ALIAS_FLAG);
        paintShine.setStyle(Paint.Style.STROKE);
        paintShine.setStrokeWidth(2.0f);
        paintShine.setColor(Color.argb(255, 0, 255, 0));
        paintShine.setStrokeCap(Paint.Cap.ROUND);
        paintShine.setStrokeJoin(Paint.Join.ROUND);
    }

    public void setCallback(DrawSurfaceCallback onReadyCallback) {
        this.onReadyCallback = onReadyCallback;

        if (holder.getSurface().isValid()) onReadyCallback.onSurfaceReady(this);
    }

    public void setInteractive(boolean interative) {
        this.interactive = interative;
    }

    public void clearCanvas() throws Exception {
        if (!holder.getSurface().isValid()) {
            throw new Exception("surface not valid");
        }

        Canvas canvas = holder.lockCanvas();

        if (canvas == null) {
            throw new Exception("canvas not valid");
        }

        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
        holder.unlockCanvasAndPost(canvas);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        drawLines(canvas, this.contours, paintWhite);

        for (Marker marker : markers) {

            if (marker.transformationMatrix != null) {
                
                Size size = new Size(200, 60);

                Bitmap bmap = Bitmap.createBitmap(size.getWidth(), size.getHeight(), Bitmap.Config.ARGB_8888);
                Canvas c2 = new Canvas(bmap);

//            TextView v = new TextView(this.context);
//            v.setText("narf");
//            v.layout(0, 0, 200, 200);

                LayoutInflater li = (LayoutInflater) context.getSystemService(Context.LAYOUT_INFLATER_SERVICE);
                View v = li.inflate(R.layout.frekvens_layout, null);

                int widthSpec = View.MeasureSpec.makeMeasureSpec(size.getWidth(), View.MeasureSpec.EXACTLY);
                int heightSpec = View.MeasureSpec.makeMeasureSpec(size.getHeight(), View.MeasureSpec.EXACTLY);
                v.measure(widthSpec, heightSpec);

                v.layout(0, 0, size.getWidth(), size.getHeight());

                v.draw(c2);

                marker.transformationMatrix.postScale((float) scale.val[0], (float) scale.val[1]);
                marker.transformationMatrix.postTranslate(offset[0], offset[1]);

                canvas.drawBitmap(bmap, marker.transformationMatrix, paintGreen);
            }

            drawLines(canvas, marker.contours, paintShine);
        }
    }

    private void drawLines(Canvas canvas, List<MatOfPoint> lines, Paint paint) {
        if (lines != null){

            MatOfPoint transformed = new MatOfPoint();

            for (MatOfPoint mat : lines) {

                Core.multiply(mat, this.scale, transformed);
                List<Point> points = transformed.toList();

                if (points.size() < 3) {
                    continue;
                }

                for (int i=0; i<points.size(); i++) {
                    Point prevPoint = points.get(i);
                    Point p = points.get((i+1)%points.size());

                    canvas.drawLine(
                            (float) (prevPoint.x + this.offset[0]),
                            (float) (prevPoint.y + this.offset[1]),
                            (float) (p.x + this.offset[0]),
                            (float) (p.y + this.offset[1]),
                            paint
                    );
                }
            }
        }
    }

    public void addLines(List<MatOfPoint> contours) {
        this.contours = contours;
        this.invalidate();
    }

    public void addMarkers(List<Marker> markers) {
        this.markers = markers;
        this.invalidate();
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        Log.d(TAG, "foo");
        return this.gestureDetector.onTouchEvent(event);
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        if (onReadyCallback != null) {
            onReadyCallback.onSurfaceReady(this);
        }
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {

    }

    public static class DrawSurfaceCallback {

        public DrawSurfaceCallback() {
        }

        public void onSurfaceReady(DrawSurface surface) {

        }
    }
}