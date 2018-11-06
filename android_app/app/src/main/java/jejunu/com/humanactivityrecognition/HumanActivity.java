package jejunu.com.humanactivityrecognition;

public class HumanActivity {

    public static final int FLAG_WAKING = 0;
    public static final int FLAG_UPSTAIRS = 1;
    public static final int FLAG_DOWNSTAIRS = 2;
    public static final int FLAG_SITTING = 3;
    public static final int FLAG_STANDING = 4;
    public static final int FLAG_LAYING = 5;
    private static final double HOURS_TO_MILISECONDS = 3600000.0;

    private long mTimeStart;
    private long mLastMeasureTime;
    private double mCurrentCaloriesBurning = 0;
    private double mWeight = 80;
    private int mCurrentActivityFlag = -1;

    public HumanActivity() {
        mTimeStart = System.currentTimeMillis();
        mLastMeasureTime = System.currentTimeMillis();
    }

    public double getCurrentCalories() {
        return mCurrentCaloriesBurning;
    }

    public String getCurrentActivity() {
        return mapActivity(mCurrentActivityFlag);
    }

    public void addCurrentCaloriesBurning(float mResults[]) {
        long now = System.currentTimeMillis();
        long timeLong = now - mLastMeasureTime;
        mCurrentCaloriesBurning += mapMet(getCurrentActivity(mResults)) * mWeight * (timeLong / HOURS_TO_MILISECONDS);
        mLastMeasureTime = System.currentTimeMillis();
    }

    public int getCurrentActivity(float mResults[]) {
        float max = mResults[0];
        int flagMax = 0;
        for (int i = 1; i < mResults.length; i++) {
            if (max < mResults[i]) {
                max = mResults[i];
                flagMax = i;
            }
        }
        mCurrentActivityFlag = flagMax;
        return flagMax;
    }

    public static String mapActivity(int flag) {
        switch (flag) {
            case FLAG_WAKING:
                return "Walking";
            case FLAG_LAYING:
                return "Laying";
            case FLAG_SITTING:
                return "Sitting";
            case FLAG_UPSTAIRS:
                return "Upstairs";
            case FLAG_DOWNSTAIRS:
                return "Downstairs";
            case FLAG_STANDING:
                return "Standing";
            default:
                return "Unknown";
        }
    }

    public static double mapMet(int flag) {
        switch (flag) {
            case FLAG_WAKING:
                return 2.9;
            case FLAG_LAYING:
                return 0.9;
            case FLAG_SITTING:
                return 1.5;
            case FLAG_UPSTAIRS:
                return 4;
            case FLAG_DOWNSTAIRS:
                return 3.5;
            case FLAG_STANDING:
                return 1.3;
            default:
                return 0;
        }
    }

}
