package com.example.ml.request;

import com.example.ml.classifiers.HyperParameter;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ModelRequest {
    private String name;
    private String algorithm;
    private List<HyperParameter> hyperparameters;
    private int capacity;
    private List<AttributeRequest> attributes;
    private List<InstanceRequest> instances;
}