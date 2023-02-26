package com.example.ml.response;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

//@Data
//@AllArgsConstructor
//@NoArgsConstructor
//public class PredictionResponse {
//    //private List<String> labels;
//    private double[] distribution;
//    private Map<String, Double> metrics;
//
//
//}

@Data
@AllArgsConstructor
@NoArgsConstructor
public class PredictionResponse {
    private String prediction;
    private String evaluation;
    private double[] probabilities;
}
