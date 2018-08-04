package jejunu.com.humanactivityrecognition;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.TextView;

import com.scichart.charting.ClipMode;
import com.scichart.charting.model.dataSeries.XyDataSeries;
import com.scichart.charting.modifiers.AxisDragModifierBase;
import com.scichart.charting.modifiers.ModifierGroup;
import com.scichart.charting.modifiers.SourceMode;
import com.scichart.charting.visuals.SciChartSurface;
import com.scichart.charting.visuals.axes.AutoRange;
import com.scichart.charting.visuals.axes.AxisAlignment;
import com.scichart.charting.visuals.axes.IAxis;
import com.scichart.charting.visuals.renderableSeries.IRenderableSeries;
import com.scichart.charting.visuals.renderableSeries.LineDrawMode;
import com.scichart.core.annotations.Orientation;
import com.scichart.drawing.utility.ColorUtil;
import com.scichart.extensions.builders.SciChartBuilder;

import java.util.Collections;

public class MainActivity extends AppCompatActivity implements SensorEventListener {

    private SensorManager mSensorManager;
    private Sensor mAccelerometer;
    private Sensor mGyroscope;
    private TextView mGyroscopeTxt;
    private TextView mAccelerometerTxt;

    private Toolbar mToolbar;
    private SciChartSurface mGyrsChartSurface;
    private SciChartSurface mAccChartSurface;

    int mCapacitySize = 100;

    private LinearLayout mChartLayout;
    // Obtain the SciChartBuilder instance
    private SciChartBuilder mChartBuilder;
    private XyDataSeries mGyrsLineXData, mGyrsLineYData, mGyrsLineZData;
    private XyDataSeries mAccLineXData, mAccLineYData, mAccLineZData;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        initComponents();
    }

    private void initComponents() {
        initToolbar();
        initComponentViews();
        initDataViews();
    }

    private void initDataViews() {
    }

    private void initToolbar() {
        mToolbar = findViewById(R.id.toolbar);
        setSupportActionBar(mToolbar);
        if (getSupportActionBar() != null) {
            getSupportActionBar().setDisplayHomeAsUpEnabled(true);
            getSupportActionBar().setDisplayShowHomeEnabled(true);
        }
        mToolbar.setNavigationOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                onBackPressed();
            }
        });
    }

    private void initComponentViews() {
        mSensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        assert mSensorManager != null;
        mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        mGyroscope = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        mAccelerometerTxt = findViewById(R.id.accelerometer_txt);
        mGyroscopeTxt = findViewById(R.id.gyroscope_txt);

        mChartLayout = findViewById(R.id.gyroscope_chart);
        mGyrsChartSurface = new SciChartSurface(this);
        mAccChartSurface = new SciChartSurface(this);

        mChartLayout.addView(mGyrsChartSurface);
        mChartLayout.addView(mAccChartSurface);

        SciChartBuilder.init(this);

        // Set layout parameters for both surfaces
        LinearLayout.LayoutParams layoutParams = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.MATCH_PARENT, 1.0f);

        mGyrsChartSurface.setLayoutParams(layoutParams);
        mAccChartSurface.setLayoutParams(layoutParams);

        mChartBuilder = SciChartBuilder.instance();

        // Create a numeric X axis
        final IAxis xGyrsAxis = mChartBuilder.newNumericAxis()
                .withAxisTitle("X Axis - Gyroscope")
                .withAutoRangeMode(AutoRange.Always)
                .withAutoFitMarginalLabels(true)
                .build();

        // Create a numeric Y axis
        final IAxis yGyrsAxis = mChartBuilder.newNumericAxis()
                .withAxisTitle("Y Axis - Gyroscope")
                .withAxisAlignment(AxisAlignment.Left)
                .withDrawMajorTicks(true)
                .withVisibleRange(-10, 10)
                .withAutoFitMarginalLabels(true)
                .build();

        // Create a numeric X axis
        final IAxis xAccAxis = mChartBuilder.newNumericAxis()
                .withAxisTitle("X Axis - Accelerometer")
                .withAutoRangeMode(AutoRange.Always)
                .build();


        // Create another numeric axis
        final IAxis yAccAxis = mChartBuilder.newNumericAxis()
                .withAxisTitle("Y Axis - Accelerometer")
                .withAxisAlignment(AxisAlignment.Left)
                .withDrawMajorTicks(true)
                .withVisibleRange(-20, 20)
                .withAutoFitMarginalLabels(true)
                .build();


        // Create interactivity modifiers
        ModifierGroup chartModifiers = mChartBuilder.newModifierGroup()
                .withPinchZoomModifier().build()
                .withZoomPanModifier().withReceiveHandledEvents(true).build()
                .withZoomExtentsModifier().withReceiveHandledEvents(true).build()
                .withXAxisDragModifier().withReceiveHandledEvents(true).withDragMode(AxisDragModifierBase.AxisDragMode.Scale).withClipModex(ClipMode.None).build()
                .withYAxisDragModifier().withReceiveHandledEvents(true).withDragMode(AxisDragModifierBase.AxisDragMode.Pan).build()
                .withLegendModifier().withOrientation(Orientation.VERTICAL).withShowCheckBoxes(true).withShowSeriesMarkers(true).withSourceMode(SourceMode.AllVisibleSeries).build()
                .build();

        ModifierGroup chartModifiers2 = mChartBuilder.newModifierGroup()
                .withPinchZoomModifier().build()
                .withZoomPanModifier().withReceiveHandledEvents(true).build()
                .withZoomExtentsModifier().withReceiveHandledEvents(true).build()
                .withXAxisDragModifier().withReceiveHandledEvents(true).withDragMode(AxisDragModifierBase.AxisDragMode.Scale).withClipModex(ClipMode.None).build()
                .withYAxisDragModifier().withReceiveHandledEvents(true).withDragMode(AxisDragModifierBase.AxisDragMode.Pan).build()
                .withLegendModifier().withOrientation(Orientation.VERTICAL).withShowCheckBoxes(true).withShowSeriesMarkers(true).withSourceMode(SourceMode.AllVisibleSeries).build()
                .build();

        Collections.addAll(mGyrsChartSurface.getYAxes(), yGyrsAxis);
        Collections.addAll(mGyrsChartSurface.getXAxes(), xGyrsAxis);
        Collections.addAll(mGyrsChartSurface.getChartModifiers(), chartModifiers);
        Collections.addAll(mAccChartSurface.getYAxes(), yAccAxis);
        Collections.addAll(mAccChartSurface.getXAxes(), xAccAxis);
        Collections.addAll(mAccChartSurface.getChartModifiers(), chartModifiers2);


        mGyrsLineXData = mChartBuilder
                .newXyDataSeries(Long.class, Double.class)
                .withFifoCapacity(mCapacitySize)
                .withSeriesName("X Gyroscope")
                .build();

        mGyrsLineYData = mChartBuilder
                .newXyDataSeries(Long.class, Double.class)
                .withFifoCapacity(mCapacitySize)
                .withSeriesName("Y Gyroscope")
                .build();

        mGyrsLineZData = mChartBuilder
                .newXyDataSeries(Long.class, Double.class)
                .withFifoCapacity(mCapacitySize)
                .withSeriesName("Z Gyroscope")
                .build();


        mAccLineXData = mChartBuilder
                .newXyDataSeries(Long.class, Double.class)
                .withFifoCapacity(mCapacitySize)
                .withSeriesName("X Accelerometer")
                .build();

        mAccLineYData = mChartBuilder
                .newXyDataSeries(Long.class, Double.class)
                .withFifoCapacity(mCapacitySize)
                .withSeriesName("Y Accelerometer")
                .build();

        mAccLineZData = mChartBuilder
                .newXyDataSeries(Long.class, Double.class)
                .withFifoCapacity(mCapacitySize)
                .withSeriesName("Z Accelerometer")
                .build();

        // Create and configure a line series
        final IRenderableSeries lineGyrsXSeries = mChartBuilder.newLineSeries()
                .withStrokeStyle(ColorUtil.LightBlue, 2f, true)
                .withDrawLineMode(LineDrawMode.ClosedLines)
                .withDataSeries(mGyrsLineXData)
                .build();

        final IRenderableSeries lineGyrsYSeries = mChartBuilder.newLineSeries()
                .withStrokeStyle(ColorUtil.Yellow, 2f, true)
                .withDrawLineMode(LineDrawMode.ClosedLines)
                .withDataSeries(mGyrsLineYData)
                .build();

        final IRenderableSeries lineGyrsZSeries = mChartBuilder.newLineSeries()
                .withStrokeStyle(ColorUtil.Red, 2f, true)
                .withDrawLineMode(LineDrawMode.ClosedLines)
                .withDataSeries(mGyrsLineZData)
                .build();

        // Add a RenderableSeries onto the SciChartSurface
        mGyrsChartSurface.getRenderableSeries().add(lineGyrsXSeries);
        mGyrsChartSurface.getRenderableSeries().add(lineGyrsYSeries);
        mGyrsChartSurface.getRenderableSeries().add(lineGyrsZSeries);


        // Create and configure a line series
        final IRenderableSeries lineAccXSeries = mChartBuilder.newLineSeries()
                .withStrokeStyle(ColorUtil.LightBlue, 2f, true)
                .withDrawLineMode(LineDrawMode.ClosedLines)
                .withDataSeries(mAccLineXData)
                .build();

        final IRenderableSeries lineAccYSeries = mChartBuilder.newLineSeries()
                .withStrokeStyle(ColorUtil.Yellow, 2f, true)
                .withDrawLineMode(LineDrawMode.ClosedLines)
                .withDataSeries(mAccLineYData)
                .build();

        final IRenderableSeries lineAccZSeries = mChartBuilder.newLineSeries()
                .withStrokeStyle(ColorUtil.Red, 2f, true)
                .withDrawLineMode(LineDrawMode.ClosedLines)
                .withDataSeries(mAccLineZData)
                .build();

        mAccChartSurface.getRenderableSeries().add(lineAccXSeries);
        mAccChartSurface.getRenderableSeries().add(lineAccYSeries);
        mAccChartSurface.getRenderableSeries().add(lineAccZSeries);

    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        long time = System.currentTimeMillis();
        switch (event.sensor.getType()) {
            case Sensor.TYPE_ACCELEROMETER:
                mAccelerometer = event.sensor;
                mAccLineXData.append(time, (double) event.values[0]);
                mAccLineYData.append(time, (double) event.values[1]);
                mAccLineZData.append(time, (double) event.values[2]);
                mAccelerometerTxt.setText("Accelerometer: X: " + event.values[0] + "; Y: " + event.values[1] + "; Z: " + event.values[2] + ";\n" +
                        "Power: " + mAccelerometer.getPower());
                break;

            case Sensor.TYPE_GYROSCOPE:
                mGyroscope = event.sensor;
                mGyrsLineXData.append(time, (double) event.values[0]);
                mGyrsLineYData.append(time, (double) event.values[1]);
                mGyrsLineZData.append(time, (double) event.values[2]);
                mGyroscopeTxt.setText("Gyroscope: X: " + event.values[0] + "; Y: " + event.values[1] + "; Z: " + event.values[2] + ";\n" +
                        "Power: " + mAccelerometer.getPower());
                break;
        }

    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }

    @Override
    protected void onResume() {
        super.onResume();
        mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_NORMAL);
        mSensorManager.registerListener(this, mGyroscope, SensorManager.SENSOR_DELAY_NORMAL);
    }

    @Override
    protected void onPause() {
        super.onPause();
        mSensorManager.unregisterListener(this);
    }
}
