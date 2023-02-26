package com.example.try1231231;


import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.ErrorResponse;
import org.springframework.web.bind.annotation.*;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.*;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToNominal;

import java.util.*;

@RestController
@RequestMapping("/model")
public class ModelController {


    private Map<String, Classifier> models = new HashMap<>();
    private Map<String, Instances> inst = new HashMap<>();



    @PostMapping("/create")
    public ResponseEntity<String> createModel(@RequestBody ModelRequest request) {
        try {
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

            // Create m

            // Create model
            Classifier classifier = null;
            switch (request.getAlgorithm()) {
                case "J48":
                    classifier = new J48();
                    //((J48) classifier).setOptions(request.getHyperparameters());
                    break;
                case "NaiveBayes":
                    classifier = new NaiveBayes();
                    // ((NaiveBayes) classifier).setOptions(request.getHyperparameters());
                    break;
                case "RandomForest":
                    classifier = new RandomForest();
                    ((RandomForest) classifier).setOptions(weka.core.Utils.splitOptions("-I 100 -K 10"));
                    //((RandomForest) classifier).setOptions(request.getHyperparameters());
                    break;
                // добавьте другие алгоритмы здесь
                default:
                    return ResponseEntity.status(HttpStatus.BAD_REQUEST).body("Invalid algorithm");
            }
            classifier.buildClassifier(instances);

            // Store model
            models.put(request.getName(), classifier);

            return ResponseEntity.ok("Model created.");
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(e.getMessage());
        }
    }
    @PostMapping("/predict")
    public ResponseEntity<Object> predict(@RequestBody PredictionRequest request) {
        try {
            // Get model
            Classifier classifier = models.get(request.getModelName());
            if (classifier == null) {
                return ResponseEntity.badRequest().body("Model not found");
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

            // Get probabilities (if available)
            double[] probabilities = null;
            if (classAttribute.isNominal()) {
                int predictionIndex = (int) prediction;
                probabilities = classifier.distributionForInstance(instances.firstInstance());
                predictionAsString += " (probabilities: " + Arrays.toString(probabilities) + ")";
            }

            // Return prediction and evaluation
            return ResponseEntity.ok(new PredictionResponse(predictionAsString, evaluation.toSummaryString(), probabilities));
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(e.getMessage());
        }
    }




}