package com.example.ml.classifiers;

import com.example.ml.interfaces.MachineLearningAlgorithm;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

public class J48Classifier implements MachineLearningAlgorithm {
    private final J48 classifier;

    public J48Classifier() {
        this.classifier = new J48();
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        classifier.buildClassifier(data);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return classifier.classifyInstance(instance);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return classifier.distributionForInstance(instance);
    }

    @Override
    public Evaluation evaluateModel(Instances data) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(classifier, data);
        return eval;
    }
    public void setOptions(String[] options) throws Exception {
        this.classifier.setOptions(options);
    }
}


