package jejunu.com.humanactivityrecognition;

import android.annotation.SuppressLint;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.widget.CompoundButton;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;

import com.scichart.charting.ClipMode;
import com.scichart.charting.model.dataSeries.IDataSeries;
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

import java.math.BigDecimal;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;

import uk.me.berndporr.iirj.Butterworth;

public class MainActivity extends AppCompatActivity implements SensorEventListener {

    private SensorManager mSensorManager;
    private Sensor mAccelerometer;
    private Sensor mGyroscope;

    private Toolbar mToolbar;
    private SciChartSurface mGyrsChartSurface;
    private SciChartSurface mAccChartSurface;

    int mCapacitySize = 200;

    private LinearLayout mChartLayout;
    // Obtain the SciChartBuilder instance
    private SciChartBuilder mChartBuilder;
    private XyDataSeries mGyrsLineXData, mGyrsLineYData, mGyrsLineZData;
    private XyDataSeries mAccLineXData, mAccLineYData, mAccLineZData;

    // remember to change this (128 - 200)
    private static final int N_SAMPLES = 200;
    private static List<Float> accX, accY, accZ, gyrsX, gyrsY, gyrsZ;
    private TextView mTimeTextView, mHealthInfo;
    private Long mLastAccTimer = 0L;
    private Long mLastGyroTimer = 0L;
    private Long mStartTime = 0L;

    private String[] labels = {"Downstairs", "Upstairs", "Walking", "Sitting", "Laying", "Standing"};
    //    private String[] mLabels = {"WAKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"};
    private TensorFlowClassifier mClassifier;
    private TextView mActWalking, mActUpstairs, mActDownstairs, mActSitting, mActStanding, mActLaying;
    private float[] mResults;
    private Butterworth mButterworth;

    private ToggleButton mShowPredictionBtn, mShowButterworthBtn, mShowMedianFilterBtn;
    private boolean isShowButterworth = true;
    private boolean isShowMedianFilter = true;
    private LinearLayout mPredictionFrame;

    private HumanActivity mHumanActivity = new HumanActivity();


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
        accX = new ArrayList<>();
        accY = new ArrayList<>();
        accZ = new ArrayList<>();

        gyrsX = new ArrayList<>();
        gyrsY = new ArrayList<>();
        gyrsZ = new ArrayList<>();

        mClassifier = new TensorFlowClassifier(getApplicationContext());

        mButterworth = new Butterworth();
        mButterworth.lowPass(3, 50, 20);

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
        mPredictionFrame = findViewById(R.id.prediction_frame);
        mShowPredictionBtn = findViewById(R.id.show_prediction_btn);
        mShowPredictionBtn.setChecked(true);
        mShowPredictionBtn.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
                if (b)
                    mPredictionFrame.setVisibility(View.VISIBLE);
                else
                    mPredictionFrame.setVisibility(View.GONE);
            }
        });

        mShowButterworthBtn = findViewById(R.id.show_butterworth_btn);
        mShowButterworthBtn.setChecked(isShowButterworth);
        mShowButterworthBtn.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
                isShowButterworth = b;
                Toast.makeText(getApplicationContext(), "3rd low-pass Butterworth Filter: " + isShowButterworth, Toast.LENGTH_SHORT).show();

            }
        });

        mShowMedianFilterBtn = findViewById(R.id.show_median_btn);
        mShowMedianFilterBtn.setChecked(isShowMedianFilter);
        mShowMedianFilterBtn.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
                isShowMedianFilter = b;
                Toast.makeText(getApplicationContext(), "Median Filter: " + isShowMedianFilter, Toast.LENGTH_SHORT).show();
            }
        });


        mSensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        mGyroscope = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);

        mChartLayout = findViewById(R.id.chart);
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

        // Create interactivity modifiers
        Collections.addAll(mGyrsChartSurface.getYAxes(), createAxis(false, "Gyroscope", 10));
        Collections.addAll(mGyrsChartSurface.getXAxes(), createAxis(true, "Gyroscope", 0));
        Collections.addAll(mGyrsChartSurface.getChartModifiers(), buildModifier());
        Collections.addAll(mAccChartSurface.getYAxes(), createAxis(false, "Linear Accelerometer", 20));
        Collections.addAll(mAccChartSurface.getXAxes(), createAxis(true, "Linear Accelerometer", 0));
        Collections.addAll(mAccChartSurface.getChartModifiers(), buildModifier());

        mGyrsLineXData = buildDataSeries("X Gyroscope");
        mGyrsLineYData = buildDataSeries("Y Gyroscope");
        mGyrsLineZData = buildDataSeries("Z Gyroscope");

        mAccLineXData = buildDataSeries("X Accelerometer");
        mAccLineYData = buildDataSeries("Y Accelerometer");
        mAccLineZData = buildDataSeries("Z Accelerometer");

        mGyrsChartSurface.getRenderableSeries().add(renderLineSeries(ColorUtil.LightBlue, mGyrsLineXData));
        mGyrsChartSurface.getRenderableSeries().add(renderLineSeries(ColorUtil.Yellow, mGyrsLineYData));
        mGyrsChartSurface.getRenderableSeries().add(renderLineSeries(ColorUtil.Red, mGyrsLineZData));

        mAccChartSurface.getRenderableSeries().add(renderLineSeries(ColorUtil.LightBlue, mAccLineXData));
        mAccChartSurface.getRenderableSeries().add(renderLineSeries(ColorUtil.Yellow, mAccLineYData));
        mAccChartSurface.getRenderableSeries().add(renderLineSeries(ColorUtil.Red, mAccLineZData));

        mActWalking = findViewById(R.id.walking_prob);
        mActUpstairs = findViewById(R.id.walking_upstairs_prob);
        mActDownstairs = findViewById(R.id.walking_downstairs_prob);
        mActSitting = findViewById(R.id.sitting_prob);
        mActLaying = findViewById(R.id.laying_prob);
        mActStanding = findViewById(R.id.standing_prob);

        mTimeTextView = findViewById(R.id.timer);
        mHealthInfo = findViewById(R.id.health_info);

    }

    private IAxis createAxis(boolean isX, String name, double absoluteYRange) {
        if (isX)
            return mChartBuilder.newNumericAxis()
                    .withAxisTitle("X Axis - " + name)
                    .withAutoRangeMode(AutoRange.Always)
                    .withAutoFitMarginalLabels(true)
                    .build();

        absoluteYRange = Math.abs(absoluteYRange);
        return mChartBuilder.newNumericAxis()
                .withAxisTitle("Y Axis - " + name)
                .withAxisAlignment(AxisAlignment.Left)
                .withDrawMajorTicks(true)
                .withVisibleRange(-1 * absoluteYRange, absoluteYRange)
                .withAutoFitMarginalLabels(true)
                .build();
    }

    private ModifierGroup buildModifier() {
        return mChartBuilder.newModifierGroup()
                .withPinchZoomModifier().build()
                .withZoomPanModifier().withReceiveHandledEvents(true).build()
                .withZoomExtentsModifier().withReceiveHandledEvents(true).build()
                .withXAxisDragModifier().withReceiveHandledEvents(true).withDragMode(AxisDragModifierBase.AxisDragMode.Scale).withClipModex(ClipMode.None).build()
                .withYAxisDragModifier().withReceiveHandledEvents(true).withDragMode(AxisDragModifierBase.AxisDragMode.Pan).build()
                .withLegendModifier().withOrientation(Orientation.VERTICAL).withShowCheckBoxes(true).withShowSeriesMarkers(true).withSourceMode(SourceMode.AllVisibleSeries).build()
                .build();
    }

    private XyDataSeries buildDataSeries(String name) {
        return mChartBuilder
                .newXyDataSeries(Long.class, Double.class)
                .withFifoCapacity(mCapacitySize)
                .withSeriesName(name)
                .build();
    }

    private IRenderableSeries renderLineSeries(int seriesColor, IDataSeries dataSeries) {
        return mChartBuilder.newLineSeries()
                .withStrokeStyle(seriesColor, 2f, true)
                .withDrawLineMode(LineDrawMode.ClosedLines)
                .withDataSeries(dataSeries)
                .build();
    }

    private static int MEDIAN_FILTER_SIZE = 5;
    private ArrayList<Double> arrFilterAccX = new ArrayList<Double>();
    private ArrayList<Double> arrFilterAccY = new ArrayList<Double>();
    private ArrayList<Double> arrFilterAccZ = new ArrayList<Double>();
    private ArrayList<Double> arrFilterGyrsX = new ArrayList<Double>();
    private ArrayList<Double> arrFilterGyrsY = new ArrayList<Double>();
    private ArrayList<Double> arrFilterGyrsZ = new ArrayList<Double>();


    @Override
    public void onSensorChanged(SensorEvent event) {
        long currentTime = (new Date()).getTime() + (event.timestamp - System.nanoTime()) / 1000000L;
        double x, y, z;
        if (isShowButterworth) {
            x = mButterworth.filter(event.values[0]);
            y = mButterworth.filter(event.values[1]);
            z = mButterworth.filter(event.values[2]);
        } else {
            x = event.values[0];
            y = event.values[1];
            z = event.values[2];
        }

        if (event.sensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION) {
            if (isShowMedianFilter) {
                arrFilterAccX.add(x);
                arrFilterAccY.add(y);
                arrFilterAccZ.add(z);

                if (arrFilterAccX.size() == MEDIAN_FILTER_SIZE) {
                    x = FilterAlgorithm.MedianFilter(arrFilterAccX);
                    arrFilterAccX.clear();
                } else {
                    return;
                }

                if (arrFilterAccY.size() == MEDIAN_FILTER_SIZE) {
                    y = FilterAlgorithm.MedianFilter(arrFilterAccY);
                    arrFilterAccY.clear();
                } else {
                    return;
                }

                if (arrFilterAccZ.size() == MEDIAN_FILTER_SIZE) {
                    z = FilterAlgorithm.MedianFilter(arrFilterAccZ);
                    arrFilterAccZ.clear();
                } else {
                    return;
                }

            }

            // Accelerometer Sensor
            mAccLineXData.append(currentTime, x);
            mAccLineYData.append(currentTime, y);
            mAccLineZData.append(currentTime, z);

            if (mLastAccTimer == 0 && accX.size() < N_SAMPLES) {
                mLastAccTimer = currentTime;
                mStartTime = currentTime;
                accX.add((float) x);
                accY.add((float) y);
                accZ.add((float) z);
            } else {

                long timeDifference = currentTime - mLastAccTimer;
                if (timeDifference >= 20 && accX.size() < N_SAMPLES) {
                    mLastAccTimer = currentTime;
                    accX.add((float) x);
                    accY.add((float) y);
                    accZ.add((float) z);
                }
            }

        } else if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE) {
            if (isShowMedianFilter) {
                arrFilterGyrsX.add(x);
                arrFilterGyrsY.add(y);
                arrFilterGyrsZ.add(z);

                if (arrFilterGyrsX.size() == MEDIAN_FILTER_SIZE) {
                    x = FilterAlgorithm.MedianFilter(arrFilterGyrsX);
                    arrFilterGyrsX.clear();
                } else {
                    return;
                }

                if (arrFilterGyrsY.size() == MEDIAN_FILTER_SIZE) {
                    y = FilterAlgorithm.MedianFilter(arrFilterGyrsY);
                    arrFilterGyrsY.clear();
                } else {
                    return;
                }

                if (arrFilterGyrsZ.size() == MEDIAN_FILTER_SIZE) {
                    z = FilterAlgorithm.MedianFilter(arrFilterGyrsZ);
                    arrFilterGyrsZ.clear();
                } else {
                    return;
                }

            }
            mGyrsLineXData.append(currentTime, x);
            mGyrsLineYData.append(currentTime, y);
            mGyrsLineZData.append(currentTime, z);

            if (mLastGyroTimer == 0 && gyrsX.size() < N_SAMPLES) {
                mLastGyroTimer = currentTime;
                mStartTime = currentTime;
                gyrsX.add((float) x);
                gyrsY.add((float) y);
                gyrsZ.add((float) z);
            } else {
                long timeDifference = currentTime - mLastGyroTimer;
                if (timeDifference >= 20 && gyrsX.size() < N_SAMPLES) {
                    mLastGyroTimer = currentTime;
                    gyrsX.add((float) x);
                    gyrsY.add((float) x);
                    gyrsZ.add((float) z);
                }
            }

        }
//        activityPrediction(currentTime);
        activityPrediction();


//        long time = System.currentTimeMillis();
//        switch (event.sensor.getType()) {
//            case Sensor.TYPE_ACCELEROMETER:
//                mAccelerometer = event.sensor;
//                mAccLineXData.append(time, (double) event.values[0]);
//                mAccLineYData.append(time, (double) event.values[1]);
//                mAccLineZData.append(time, (double) event.values[2]);
//                mAccelerometerTxt.setText("Accelerometer: X: " + event.values[0] + "; Y: " + event.values[1] + "; Z: " + event.values[2] + ";\n" +
//                        "Power: " + mAccelerometer.getPower());
//                break;
//
//            case Sensor.TYPE_GYROSCOPE:
//                mGyroscope = event.sensor;
//                mGyrsLineXData.append(time, (double) event.values[0]);
//                mGyrsLineYData.append(time, (double) event.values[1]);
//                mGyrsLineZData.append(time, (double) event.values[2]);
//                mGyroscopeTxt.setText("Gyroscope: X: " + event.values[0] + "; Y: " + event.values[1] + "; Z: " + event.values[2] + ";\n" +
//                        "Power: " + mAccelerometer.getPower());
//                break;
//        }
//        activityPrediction();
//        accX.add(event.values[0]);
//        accY.add(event.values[1]);
//        accZ.add(event.values[2]);
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }

    @Override
    protected void onResume() {
        super.onResume();
        mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_FASTEST);
        mSensorManager.registerListener(this, mGyroscope, SensorManager.SENSOR_DELAY_FASTEST);
    }

    @Override
    protected void onPause() {
        super.onPause();
        mSensorManager.unregisterListener(this);
    }

    @SuppressLint("SetTextI18n")
    private void activityPrediction() {
        if (accX.size() == N_SAMPLES && accY.size() == N_SAMPLES && accZ.size() == N_SAMPLES) {
            List<Float> data = new ArrayList<>();
            data.addAll(accX);
            data.addAll(accY);
            data.addAll(accZ);

            mResults = mClassifier.predictProbabilities(toFloatArray(data));

            mHumanActivity.addCurrentCaloriesBurning(mResults);
            mHealthInfo.setText("Calories burning: " + String.format("%.5f", mHumanActivity.getCurrentCalories()) + " cal\n" +
                    "Current Activity: " + mHumanActivity.getCurrentActivity());

            mActWalking.setText(Float.toString(round(mResults[0], 2)));
            mActUpstairs.setText(Float.toString(round(mResults[1], 2)));
            mActDownstairs.setText(Float.toString(round(mResults[2], 2)));
            mActSitting.setText(Float.toString(round(mResults[3], 2)));
            mActStanding.setText(Float.toString(round(mResults[4], 2)));
            mActLaying.setText(Float.toString(round(mResults[5], 2)));

            accX.clear();
            accY.clear();
            accZ.clear();
        }
    }

    private void activityPrediction(long eventTime) {
        if (accX.size() == N_SAMPLES && accY.size() == N_SAMPLES && accZ.size() == N_SAMPLES
                && gyrsX.size() == N_SAMPLES && gyrsY.size() == N_SAMPLES && gyrsZ.size() == N_SAMPLES) {
            List<Float> data = new ArrayList<>();
            data.addAll(gyrsX);
            data.addAll(gyrsY);
            data.addAll(gyrsZ);
            data.addAll(accX);
            data.addAll(accY);
            data.addAll(accZ);


            mResults = mClassifier.predictProbabilities(toFloatArray(data));


            mActWalking.setText(Float.toString(round(mResults[0], 2)));
            mActUpstairs.setText(Float.toString(round(mResults[1], 2)));
            mActDownstairs.setText(Float.toString(round(mResults[2], 2)));
            mActSitting.setText(Float.toString(round(mResults[3], 2)));
            mActStanding.setText(Float.toString(round(mResults[4], 2)));
            mActLaying.setText(Float.toString(round(mResults[5], 2)));

            Date date = new Date(eventTime - mStartTime);
            DateFormat formatter = new SimpleDateFormat("HH:mm:ss:SSS");
            String dateFormatted = formatter.format(date);

            mTimeTextView.setText(dateFormatted + "Number of Reading: " + Integer.toString(accX.size()));

            //invalidate();
            accX.clear();
            accY.clear();
            accZ.clear();
            gyrsX.clear();
            gyrsY.clear();
            gyrsZ.clear();

            mStartTime = 0L;
            mLastAccTimer = 0L;
            mLastGyroTimer = 0L;
        }
    }

    private float[] toFloatArray(List<Float> list) {
        int i = 0;
        float[] array = new float[list.size()];

        for (Float f : list) {
            array[i++] = (f != null ? f : Float.NaN);
        }
        return array;
    }

    private static float round(float d, int decimalPlace) {
        BigDecimal bd = new BigDecimal(Float.toString(d));
        bd = bd.setScale(decimalPlace, BigDecimal.ROUND_HALF_UP);
        return bd.floatValue();
    }
}
