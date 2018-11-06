package jejunu.com.humanactivityrecognition;

import android.content.Context;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class TensorFlowClassifier {

    private TensorFlowInferenceInterface inferenceInterface;
    private static final String MODEL_FILE = "file:///android_asset/frozen_model.pb";
    private static final String INPUT_NODE = "inputs";
    private static final String[] OUTPUT_NODES = {"y_"};
    private static final String OUTPUT_NODE = "y_";
    private static final long[] INPUT_SIZE = {1, 200, 3};
    private static final int OUTPUT_SIZE = 6;

// Frozen_har_6ip
//    private static final String INPUT_NODE = "input";
//    private static final String[] OUTPUT_NODES = {"y_"};
//    private static final String OUTPUT_NODE = "y_";
//    private static final long[] INPUT_SIZE = {1, 128, 6};
//    private static final int OUTPUT_SIZE = 6;

    // ANN_Model
//    private static final String INPUT_NODE = "dense_1_input";
//    private static final String[] OUTPUT_NODES = {"output_node0"};
//    private static final String OUTPUT_NODE = "output_node0";
////    private static final long[] INPUT_SIZE = {1, 200, 3};

    // RNN_Model
    //    gru_1_input - output_node0
//    private static final String INPUT_NODE = "gru_1_input";
//    private static final String[] OUTPUT_NODES = {"output_node0"};
//    private static final String OUTPUT_NODE = "output_node0";
//    private static final long[] INPUT_SIZE = {1, 128, 6};
//    private static final int OUTPUT_SIZE = 6;

    // LSTM model
//    lstm_1_input - output_node0
//    private static final String INPUT_NODE = "lstm_1_input";
//    private static final String[] OUTPUT_NODES = {"output_node0"};
//    private static final String OUTPUT_NODE = "output_node0";
//    private static final long[] INPUT_SIZE = {9, 128, 32};
//
//    private static final int OUTPUT_SIZE = 6;


    public TensorFlowClassifier(final Context context) {
        inferenceInterface = new TensorFlowInferenceInterface(context.getAssets(), MODEL_FILE);
    }

    public float[] predictProbabilities(float[] data) {
        float[] result = new float[OUTPUT_SIZE];
        inferenceInterface.feed(INPUT_NODE, data, INPUT_SIZE);
        inferenceInterface.run(OUTPUT_NODES);
        inferenceInterface.fetch(OUTPUT_NODE, result);

        return result;
    }
}
