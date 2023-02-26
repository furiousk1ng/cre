package com.example.ml.interfaces;

import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

public interface MachineLearningAlgorithm {
    void buildClassifier(Instances data) throws Exception;
    double classifyInstance(Instance instance) throws Exception;
    double[] distributionForInstance(Instance instance) throws Exception;
    Evaluation evaluateModel(Instances data) throws Exception;
}