package com.example.ml.classifiers;

import com.example.ml.interfaces.MachineLearningAlgorithm;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

public class NaiveBayesClassifier implements MachineLearningAlgorithm {

    private NaiveBayes classifier;

    public NaiveBayesClassifier() {
        this.classifier = new NaiveBayes();
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


    public Evaluation evaluateModel(Instances testData) throws Exception {
        Evaluation eval = new Evaluation(testData);
        eval.evaluateModel(this.classifier, testData);
        return eval;
    }
    public void setOptions(String[] options) throws Exception {
        this.classifier.setOptions(options);
    }
}
