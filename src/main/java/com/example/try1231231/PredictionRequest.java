package com.example.try1231231;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import weka.core.Attribute;

import java.util.ArrayList;

@NoArgsConstructor
@AllArgsConstructor
@Data
public class PredictionRequest {

    private String modelName;
    private String name;
    private ArrayList<Attribute> attributes;
    private double[] attributesMatrix;
}