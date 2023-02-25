package com.example.try1231231;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
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
    public ResponseEntity<String> createModel(@RequestBody ModelRequest request) {
        try {
            Classifier classifier = null;
            switch (request.getAlgorithm()) {
                case "J48":
                    J48 j48 = new J48();
                    j48.setOptions(request.getHyperparameters());
                    classifier = j48;
                    break;
                case "NaiveBayes":
                    NaiveBayes naiveBayes = new NaiveBayes();
                    naiveBayes.setOptions(request.getHyperparameters());
                    classifier = naiveBayes;
                    break;
                case "RandomForest":
                    RandomForest randomForest = new RandomForest();
                    randomForest.setOptions(request.getHyperparameters());
                    classifier = randomForest;
                    break;
                default:
                    return ResponseEntity.badRequest().body("Invalid algorithm specified");
            }
            Instances instances = new Instances(request.getName(), request.getAttributes(), request.getCapacity());
            instances.setClassIndex(instances.numAttributes() - 1);
            classifier.buildClassifier(instances);
            models.put(request.getName(), classifier);
            return ResponseEntity.ok("Model created successfully");
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(e.getMessage());
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
            List<String> labels = new ArrayList<>();
            for (int i = 0; i < predictions.length; i++) {
                labels.add(instances.classAttribute().value(i) + ": " + String.format("%.2f%%", predictions[i] * 100));
            }
            PredictionResponse response = new PredictionResponse(labels);
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(null);
        }
    }
}
