package com.example.try1231231;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
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

import java.util.*;

@RestController
@RequestMapping("/model")
public class ModelController {


    private Map<String, Classifier> models = new HashMap<>();

    @PostMapping("/create")
    public ResponseEntity<?> createModel(@RequestBody ModelRequest request) {
        try {
            // create attributes
            ArrayList<Attribute> attInfo = new ArrayList<>();
            for (AttributeRequest attribute : request.getAttributes()) {
                if (attribute.getType().equals("numeric")) {
                    attInfo.add(new Attribute(attribute.getName()));
                } else if (attribute.getType().equals("nominal")) {
                    attInfo.add(new Attribute(attribute.getName(), attribute.getValues()));
                } else {
                    throw new IllegalArgumentException("Unsupported attribute type: " + attribute.getType());
                }
            }
            Instances instances = new Instances(request.getName(), attInfo, request.getCapacity());
            instances.setClassIndex(attInfo.size() - 1);
            for (InstanceRequest instance : request.getInstances()) {
                double[] values = new double[attInfo.size()];
                for (int i = 0; i < values.length; i++) {
                    values[i] = instance.getValues().get(i);
                }
                instances.add(new DenseInstance(1.0, values));
            }

            // create model
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

            // build classifier
            classifier.buildClassifier(instances);

            // save model
            models.put(request.getName(), classifier);

            return ResponseEntity.status(HttpStatus.OK).body("Model created");
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(e.getMessage());
        }
    }

    @PostMapping("/predict")
    public ResponseEntity<PredictionResponse> predict(@RequestBody PredictionRequest request) {
        try {
            Classifier classifier = models.get(request.getModelName());
            Instances instances = new Instances(request.getName(), request.getAttributes(), 1);
            instances.setClassIndex(instances.numAttributes() - 1);
            Instance instance = new DenseInstance(1.0, request.getAttributesMatrix());
            instance.setDataset(instances);
            double[] predictions = classifier.distributionForInstance(instance);

            Evaluation evaluation = new Evaluation(instances);
            evaluation.evaluateModel(models.get(ModelRequest.class.getName()), instances);
            double accuracy = evaluation.pctCorrect() / 100.0;


            List<String> labels = new ArrayList<>();
            for (int i = 0; i < predictions.length; i++) {
                labels.add(instances.classAttribute().value(i) + ": " + String.format("%.2f%%", predictions[i] * 100));
            }

            PredictionResponse response = new PredictionResponse(labels, accuracy);
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(null);
        }
    }
}
