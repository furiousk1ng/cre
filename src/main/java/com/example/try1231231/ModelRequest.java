package com.example.try1231231;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import weka.core.Attribute;

import java.util.ArrayList;
import java.util.List;

@NoArgsConstructor
@AllArgsConstructor
@Data
public class ModelRequest {
    private String algorithm;
    private String name;
    private ArrayList<Attribute> attributes;
    private int capacity;
    private String[] hyperparameters;
}
