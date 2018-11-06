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
    private Sensor mLineAccelerometer;
    private Sensor mGyroscope;
    private Sensor mGravity;

    private Toolbar mToolbar;
    private SciChartSurface mGyrsChartSurface;
    private SciChartSurface mLineAccChartSurface;

    int mCapacitySize = 200;

    private LinearLayout mChartLayout;
    // Obtain the SciChartBuilder instance
    private SciChartBuilder mChartBuilder;
    private XyDataSeries mGyrsLineXData, mGyrsLineYData, mGyrsLineZData;
    private XyDataSeries mAccLineXData, mAccLineYData, mAccLineZData;

    // remember to change this (128 - 200)
    // 200 official
    private static final int N_SAMPLES = 128;
    // 128 test

    private static List<Float> mLineAccX, mLineAccY, mLineAccZ, mGyrsX, mGyrsY, mGyrsZ, mGravityX, mGravityY, mGravityZ;
    private TextView mTimeTextView, mHealthInfo;
    private Long mLastLineAccTimer = 0L;
    private Long mLastGyrsTimer = 0L;
    private Long mLastGravityTimer = 0L;

    private Long mStartLineAccTime = 0L;
    private Long mStartGyrsTime = 0L;
    private Long mStartGravityTime = 0L;


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
        mLineAccX = new ArrayList<>();
        mLineAccY = new ArrayList<>();
        mLineAccZ = new ArrayList<>();

        mGyrsX = new ArrayList<>();
        mGyrsY = new ArrayList<>();
        mGyrsZ = new ArrayList<>();

        mGravityX = new ArrayList<>();
        mGravityY = new ArrayList<>();
        mGravityZ = new ArrayList<>();

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
        mLineAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        mGyroscope = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        mGravity = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
//        mGravity = mSensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY);

        mChartLayout = findViewById(R.id.chart);
        mGyrsChartSurface = new SciChartSurface(this);
        mLineAccChartSurface = new SciChartSurface(this);

        mChartLayout.addView(mGyrsChartSurface);
        mChartLayout.addView(mLineAccChartSurface);

        SciChartBuilder.init(this);

        // Set layout parameters for both surfaces
        LinearLayout.LayoutParams layoutParams = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.MATCH_PARENT, 1.0f);

        mGyrsChartSurface.setLayoutParams(layoutParams);
        mLineAccChartSurface.setLayoutParams(layoutParams);

        mChartBuilder = SciChartBuilder.instance();

        // Create interactivity modifiers
        Collections.addAll(mGyrsChartSurface.getYAxes(), createAxis(false, "Gyroscope", 10));
        Collections.addAll(mGyrsChartSurface.getXAxes(), createAxis(true, "Gyroscope", 0));
        Collections.addAll(mGyrsChartSurface.getChartModifiers(), buildModifier());
        Collections.addAll(mLineAccChartSurface.getYAxes(), createAxis(false, "Linear Accelerometer", 20));
        Collections.addAll(mLineAccChartSurface.getXAxes(), createAxis(true, "Linear Accelerometer", 0));
        Collections.addAll(mLineAccChartSurface.getChartModifiers(), buildModifier());

        mGyrsLineXData = buildDataSeries("X Gyroscope");
        mGyrsLineYData = buildDataSeries("Y Gyroscope");
        mGyrsLineZData = buildDataSeries("Z Gyroscope");

        mAccLineXData = buildDataSeries("X Accelerometer");
        mAccLineYData = buildDataSeries("Y Accelerometer");
        mAccLineZData = buildDataSeries("Z Accelerometer");

        mGyrsChartSurface.getRenderableSeries().add(renderLineSeries(ColorUtil.LightBlue, mGyrsLineXData));
        mGyrsChartSurface.getRenderableSeries().add(renderLineSeries(ColorUtil.Yellow, mGyrsLineYData));
        mGyrsChartSurface.getRenderableSeries().add(renderLineSeries(ColorUtil.Red, mGyrsLineZData));

        mLineAccChartSurface.getRenderableSeries().add(renderLineSeries(ColorUtil.LightBlue, mAccLineXData));
        mLineAccChartSurface.getRenderableSeries().add(renderLineSeries(ColorUtil.Yellow, mAccLineYData));
        mLineAccChartSurface.getRenderableSeries().add(renderLineSeries(ColorUtil.Red, mAccLineZData));

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
    private ArrayList<Double> arrFilterLineAccX = new ArrayList<Double>();
    private ArrayList<Double> arrFilterLineAccY = new ArrayList<Double>();
    private ArrayList<Double> arrFilterLineAccZ = new ArrayList<Double>();
    private ArrayList<Double> arrFilterGravityX = new ArrayList<Double>();
    private ArrayList<Double> arrFilterGravityY = new ArrayList<Double>();
    private ArrayList<Double> arrFilterGravityZ = new ArrayList<Double>();
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
                arrFilterLineAccX.add(x);
                arrFilterLineAccY.add(y);
                arrFilterLineAccZ.add(z);

                if (arrFilterLineAccX.size() == MEDIAN_FILTER_SIZE) {
                    x = FilterAlgorithm.MedianFilter(arrFilterLineAccX);
                    arrFilterLineAccX.clear();
                } else {
                    return;
                }

                if (arrFilterLineAccY.size() == MEDIAN_FILTER_SIZE) {
                    y = FilterAlgorithm.MedianFilter(arrFilterLineAccY);
                    arrFilterLineAccY.clear();
                } else {
                    return;
                }

                if (arrFilterLineAccZ.size() == MEDIAN_FILTER_SIZE) {
                    z = FilterAlgorithm.MedianFilter(arrFilterLineAccZ);
                    arrFilterLineAccZ.clear();
                } else {
                    return;
                }

            }

            // Accelerometer Sensor
            mAccLineXData.append(currentTime, x);
            mAccLineYData.append(currentTime, y);
            mAccLineZData.append(currentTime, z);

            if (mLastLineAccTimer == 0 && mLineAccX.size() < N_SAMPLES) {
                mLastLineAccTimer = currentTime;
                mStartLineAccTime = currentTime;
                mLineAccX.add((float) x);
                mLineAccY.add((float) y);
                mLineAccZ.add((float) z);
            } else {

                long timeDifference = currentTime - mLastLineAccTimer;
                if (timeDifference >= 20 && mLineAccX.size() < N_SAMPLES) {
                    mLastLineAccTimer = currentTime;
                    mLineAccX.add((float) x);
                    mLineAccY.add((float) y);
                    mLineAccZ.add((float) z);
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

            if (mLastGyrsTimer == 0 && mGyrsX.size() < N_SAMPLES) {
                mLastGyrsTimer = currentTime;
                mStartGyrsTime = currentTime;
                mGyrsX.add((float) x);
                mGyrsY.add((float) y);
                mGyrsZ.add((float) z);
            } else {
                long timeDifference = currentTime - mLastGyrsTimer;
                if (timeDifference >= 20 && mGyrsX.size() < N_SAMPLES) {
                    mLastGyrsTimer = currentTime;
                    mGyrsX.add((float) x);
                    mGyrsY.add((float) x);
                    mGyrsZ.add((float) z);
                }
            }

        } else if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            if (isShowMedianFilter) {
                arrFilterGravityX.add(x);
                arrFilterGravityY.add(y);
                arrFilterGravityZ.add(z);

                if (arrFilterGravityX.size() == MEDIAN_FILTER_SIZE) {
                    x = FilterAlgorithm.MedianFilter(arrFilterGravityX);
                    arrFilterGravityX.clear();
                } else {
                    return;
                }

                if (arrFilterGravityY.size() == MEDIAN_FILTER_SIZE) {
                    y = FilterAlgorithm.MedianFilter(arrFilterGravityY);
                    arrFilterGravityY.clear();
                } else {
                    return;
                }

                if (arrFilterGravityZ.size() == MEDIAN_FILTER_SIZE) {
                    z = FilterAlgorithm.MedianFilter(arrFilterGravityZ);
                    arrFilterGravityZ.clear();
                } else {
                    return;
                }

            }
//            mGyrsLineXData.append(currentTime, x);
//            mGyrsLineYData.append(currentTime, y);
//            mGyrsLineZData.append(currentTime, z);

            if (mLastGravityTimer == 0 && mGravityX.size() < N_SAMPLES) {
                mLastGravityTimer = currentTime;
                mStartGravityTime = currentTime;
                mGravityX.add((float) x);
                mGravityY.add((float) y);
                mGravityZ.add((float) z);
            } else {
                long timeDifference = currentTime - mLastGravityTimer;
                if (timeDifference >= 20 && mGravityX.size() < N_SAMPLES) {
                    mLastGravityTimer = currentTime;
                    mGravityX.add((float) x);
                    mGravityY.add((float) x);
                    mGravityZ.add((float) z);
                }
            }

        }

//        activityPredictionOfficial();
        activityPredictionTest();

//        activityPrediction(currentTime);


//        long time = System.currentTimeMillis();
//        switch (event.sensor.getType()) {
//            case Sensor.TYPE_ACCELEROMETER:
//                mLineAccelerometer = event.sensor;
//                mAccLineXData.append(time, (double) event.values[0]);
//                mAccLineYData.append(time, (double) event.values[1]);
//                mAccLineZData.append(time, (double) event.values[2]);
//                mAccelerometerTxt.setText("Accelerometer: X: " + event.values[0] + "; Y: " + event.values[1] + "; Z: " + event.values[2] + ";\n" +
//                        "Power: " + mLineAccelerometer.getPower());
//                break;
//
//            case Sensor.TYPE_GYROSCOPE:
//                mGyroscope = event.sensor;
//                mGyrsLineXData.append(time, (double) event.values[0]);
//                mGyrsLineYData.append(time, (double) event.values[1]);
//                mGyrsLineZData.append(time, (double) event.values[2]);
//                mGyroscopeTxt.setText("Gyroscope: X: " + event.values[0] + "; Y: " + event.values[1] + "; Z: " + event.values[2] + ";\n" +
//                        "Power: " + mLineAccelerometer.getPower());
//                break;
//        }
//        activityPrediction();
//        mLineAccX.add(event.values[0]);
//        mLineAccY.add(event.values[1]);
//        mLineAccZ.add(event.values[2]);
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }

    @Override
    protected void onResume() {
        super.onResume();
        mSensorManager.registerListener(this, mLineAccelerometer, SensorManager.SENSOR_DELAY_FASTEST);
        mSensorManager.registerListener(this, mGyroscope, SensorManager.SENSOR_DELAY_FASTEST);
        mSensorManager.registerListener(this, mGravity, SensorManager.SENSOR_DELAY_FASTEST);

//        mSensorManager.registerListener(this, mGravity, SensorManager.SENSOR_DELAY_FASTEST);
    }

    @Override
    protected void onPause() {
        super.onPause();
        mSensorManager.unregisterListener(this);
    }

    @SuppressLint("SetTextI18n")
    private void activityPredictionOfficial() {
        if (mLineAccX.size() == N_SAMPLES && mLineAccY.size() == N_SAMPLES && mLineAccZ.size() == N_SAMPLES) {
            List<Float> data = new ArrayList<>();
            data.addAll(mLineAccX);
            data.addAll(mLineAccY);
            data.addAll(mLineAccZ);

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

            mLineAccX.clear();
            mLineAccY.clear();
            mLineAccZ.clear();
        }
    }

    private void activityPredictionTest() {
        if (mLineAccX.size() == N_SAMPLES && mLineAccY.size() == N_SAMPLES && mLineAccZ.size() == N_SAMPLES
                && mGyrsX.size() == N_SAMPLES && mGyrsY.size() == N_SAMPLES && mGyrsZ.size() == N_SAMPLES
                && mGravityX.size() == N_SAMPLES && mGravityY.size() == N_SAMPLES && mGravityZ.size() == N_SAMPLES) {
            List<Float> data = new ArrayList<>();
            data.addAll(mLineAccX);
            data.addAll(mLineAccY);
            data.addAll(mLineAccZ);
            data.addAll(mGyrsX);
            data.addAll(mGyrsY);
            data.addAll(mGyrsZ);
            data.addAll(mGravityX);
            data.addAll(mGravityY);
            data.addAll(mGravityZ);


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

            mLineAccX.clear();
            mLineAccY.clear();
            mLineAccZ.clear();
            mGyrsX.clear();
            mGyrsY.clear();
            mGyrsZ.clear();
            mGravityX.clear();
            mGravityY.clear();
            mGravityZ.clear();
        }
    }

    private void activityPrediction(long eventTime) {
        if (mLineAccX.size() == N_SAMPLES && mLineAccY.size() == N_SAMPLES && mLineAccZ.size() == N_SAMPLES
                && mGyrsX.size() == N_SAMPLES && mGyrsY.size() == N_SAMPLES && mGyrsZ.size() == N_SAMPLES) {
            List<Float> data = new ArrayList<>();
            data.addAll(mGyrsX);
            data.addAll(mGyrsY);
            data.addAll(mGyrsZ);
            data.addAll(mLineAccX);
            data.addAll(mLineAccY);
            data.addAll(mLineAccZ);


            mResults = mClassifier.predictProbabilities(toFloatArray(data));


            mActWalking.setText(Float.toString(round(mResults[0], 2)));
            mActUpstairs.setText(Float.toString(round(mResults[1], 2)));
            mActDownstairs.setText(Float.toString(round(mResults[2], 2)));
            mActSitting.setText(Float.toString(round(mResults[3], 2)));
            mActStanding.setText(Float.toString(round(mResults[4], 2)));
            mActLaying.setText(Float.toString(round(mResults[5], 2)));

            Date date = new Date(eventTime - mStartGravityTime);
            DateFormat formatter = new SimpleDateFormat("HH:mm:ss:SSS");
            String dateFormatted = formatter.format(date);

            mTimeTextView.setText(dateFormatted + "Number of Reading: " + Integer.toString(mLineAccX.size()));

            //invalidate();
            mLineAccX.clear();
            mLineAccY.clear();
            mLineAccZ.clear();
            mGyrsX.clear();
            mGyrsY.clear();
            mGyrsZ.clear();

            mStartGravityTime = 0L;
            mLastLineAccTimer = 0L;
            mLastGyrsTimer = 0L;
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
