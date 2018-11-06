package jejunu.com.humanactivityrecognition;

import java.util.ArrayList;

public class FilterAlgorithm {
    public static Double MedianFilter(ArrayList<Double> a) {
        double temp;
        int asize = a.size();
        //sort the array in increasing order
        for (int i = 0; i < asize; i++)
            for (int j = i + 1; j < asize; j++)
                if (a.get(i) > a.get(j)) {
                    temp = a.get(i);
                    a.set(i, a.get(j));
                    a.set(j, temp);
                }
        //if it's odd
        if (asize % 2 == 1)
            return a.get(asize / 2);
        else
            return ((a.get(asize / 2) + a.get(asize / 2 - 1)) / 2);
    }
}
