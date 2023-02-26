package com.example.ml.classifiers;

import java.util.Random;

import com.example.ml.interfaces.MachineLearningAlgorithm;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;

public class RandomForestClassifier implements MachineLearningAlgorithm {

    private RandomForest classifier;

    public RandomForestClassifier() {
        this.classifier = new RandomForest();
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        this.classifier.buildClassifier(data);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return this.classifier.classifyInstance(instance);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return this.classifier.distributionForInstance(instance);
    }

    @Override
    public Evaluation evaluateModel(Instances data) throws Exception {
        Evaluation evaluation = new Evaluation(data);
        evaluation.evaluateModel(this.classifier, data, new Random(1));
        return evaluation;
    }
}
