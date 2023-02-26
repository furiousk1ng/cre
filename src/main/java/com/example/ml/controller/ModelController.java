package com.example.ml.controller;


import com.example.ml.request.ModelRequest;
import com.example.ml.request.PredictionRequest;
import com.example.ml.response.PredictionResponse;
import com.example.ml.service.ModelService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/model")
public class ModelController {

        @Autowired
        private ModelService modelService;

    @GetMapping("/algorithms")
    public List<String> getAll() {

        return modelService.getAlgorithms();
    }

        @PostMapping("/create")
        public ResponseEntity<String> createModel(@RequestBody ModelRequest request) {
            try {
                modelService.createModel(request);
                return ResponseEntity.ok("Model created.");
            } catch (Exception e) {
                return ResponseEntity.badRequest().body(e.getMessage());
            }
        }



    @PostMapping("/predict")
    public ResponseEntity<Object> predict(@RequestBody PredictionRequest request) {
        try {
            PredictionResponse response = modelService.predict(request);
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(e.getMessage());
        }
    }




}