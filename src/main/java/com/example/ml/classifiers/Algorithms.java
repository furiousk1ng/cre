package com.example.ml.classifiers;

import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

@AllArgsConstructor
@Getter
public enum Algorithms {
    RF("Random Forest"), AB("AdaBoostM1"), NB("NaiveBayes");
    private String algorithmName;

    public static List<String> valuesList() {
        return Arrays.stream(Algorithms.values())
                .map(Algorithms::getAlgorithmName)
                .collect(Collectors.toList());
    }
}