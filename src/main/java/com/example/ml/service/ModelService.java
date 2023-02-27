package com.example.ml.service;


import com.example.ml.classifiers.*;
import com.example.ml.interfaces.MachineLearningAlgorithm;
import com.example.ml.request.InstanceRequest;
import com.example.ml.request.ModelRequest;
import com.example.ml.request.AttributeRequest;
import com.example.ml.request.PredictionRequest;
import com.example.ml.response.PredictionResponse;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

@Service
public class ModelService {


    private Map<String, Classifier> models = new HashMap<>();


    public List<String> getAlgorithms() {
        return Algorithms.valuesList();

    }


    public void createModel(ModelRequest request) throws Exception {
        // Parse attributes

        ArrayList<Attribute> attInfo = parseAttr(request);
        // Create instances
        Instances instances = new Instances(request.getName(), attInfo, request.getCapacity());
        instances.setClassIndex(attInfo.size() - 1);
        for (InstanceRequest instance : request.getInstances()) {
            double[] values = new double[attInfo.size()];
            for (int i = 0; i < values.length; i++) {
                Attribute attribute = attInfo.get(i);
                if (attribute.isNominal()) {
                    String valueAsString = instance.getValues().get(i);
                    int valueAsInt = attribute.indexOfValue(valueAsString);
                    values[i] = valueAsInt;
                } else {
                    String valueAsString = instance.getValues().get(i);
                    values[i] = Double.parseDouble(valueAsString);
                }
            }
            Instance instanceToAdd = new DenseInstance(1.0, values);
            instances.add(instanceToAdd);
        }

        // Create model
       Classifier classifier = createClassifier(request);
        classifier.buildClassifier(instances);

        // Store model
        models.put(request.getName(), classifier);
    }

    public PredictionResponse predict(PredictionRequest request) throws Exception {
        // Get model
        Classifier classifier = models.get(request.getModelName());
        if (classifier == null) {
            throw new Exception("Model not found");
        }

        // Parse attributes
        ArrayList<Attribute> attInfo = new ArrayList<>();
        for (AttributeRequest attribute : request.getAttributes()) {
            if (attribute.getType().equals("numeric")) {
                attInfo.add(new Attribute(attribute.getName()));
            } else if (attribute.getType().equals("nominal")) {
                ArrayList<String> nominalValues = new ArrayList<>(attribute.getNominalValues());
                attInfo.add(new Attribute(attribute.getName(), nominalValues));
            }
        }

        // Create instances
        Instances instances = new Instances(request.getName(), attInfo, 1);
        instances.setClassIndex(attInfo.size() - 1);
        double[] values = new double[attInfo.size()];
        for (int i = 0; i < values.length; i++) {
            Attribute attribute = attInfo.get(i);
            if (attribute.isNominal()) {
                String valueAsString = request.getAttributesMatrix().get(i).toString();
                int valueAsInt = attribute.indexOfValue(valueAsString);
                values[i] = valueAsInt;
            } else {
                String valueAsString = request.getAttributesMatrix().get(i).toString();
                values[i] = Double.parseDouble(valueAsString);
            }
        }
        Instance instance = new DenseInstance(1.0, values);
        instances.add(instance);

        // Make prediction
        double prediction = classifier.classifyInstance(instances.firstInstance());

        // Convert prediction to nominal value (if applicable)
        Attribute classAttribute = instances.classAttribute();
        String predictionAsString = classAttribute.value((int) prediction);

        // Evaluate model
        Evaluation evaluation = new Evaluation(instances);
        evaluation.evaluateModel(classifier, instances);


        double[] probabilities = null;
        if (classAttribute.isNominal()) {
            int predictionIndex = (int) prediction;
            probabilities = classifier.distributionForInstance(instances.firstInstance());
            predictionAsString += " (probabilities: " + Arrays.toString(probabilities) + ")";
        }

        // Return prediction and evaluation
        return new PredictionResponse(predictionAsString, evaluation.toSummaryString(), probabilities);
    }

    public static ArrayList<Attribute> parseAttr(ModelRequest request){
        ArrayList<Attribute> attInfo = new ArrayList<>();
        for (AttributeRequest attribute : request.getAttributes()) {
            if (attribute.getType().equals("numeric")) {
                attInfo.add(new Attribute(attribute.getName()));
            } else if (attribute.getType().equals("nominal")) {
                ArrayList<String> nominalValues = new ArrayList<>(attribute.getNominalValues());
                attInfo.add(new Attribute(attribute.getName(), nominalValues));
            }
        }
        return attInfo;
    }

    public static ArrayList<Attribute> parseAttr(PredictionRequest request){
        ArrayList<Attribute> attInfo = new ArrayList<>();
        for (AttributeRequest attribute : request.getAttributes()) {
            if (attribute.getType().equals("numeric")) {
                attInfo.add(new Attribute(attribute.getName()));
            } else if (attribute.getType().equals("nominal")) {
                ArrayList<String> nominalValues = new ArrayList<>(attribute.getNominalValues());
                attInfo.add(new Attribute(attribute.getName(), nominalValues));
            }
        }
        return attInfo;
    }

    public Classifier createClassifier(ModelRequest request) throws Exception {
       Classifier classifier = null;
        switch (request.getAlgorithm()) {
            case "J48":
                classifier = new J48();
                if (request.getHyperparameters() != null) {
                    for (HyperParameter param : request.getHyperparameters()) {
                        ((J48) classifier).setOptions(new String[]{param.getName(), param.getValue()});
                    }
                }
                break;
            case "NaiveBayes":
                classifier = new NaiveBayes();
                if (request.getHyperparameters() != null) {
                    for (HyperParameter param : request.getHyperparameters()) {
                        ((NaiveBayes) classifier).setOptions(new String[]{param.getName(), param.getValue()});
                    }
                }
                break;
            case "RandomForest":
                classifier = new RandomForest();
                if (request.getHyperparameters() != null) {
                    for (HyperParameter param : request.getHyperparameters()) {
                        ((RandomForest) classifier).setOptions(new String[]{param.getName(), param.getValue()});
                    }
                }
                break;
            default:
                throw new Exception("Invalid algorithm");
        }
        return classifier;
    }

}
