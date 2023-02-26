package com.example.try1231231;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import weka.core.Attribute;
import weka.core.Instance;

import java.util.ArrayList;
import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ModelRequest {
    private String name;
    private String algorithm;
    private int capacity;
    private List<AttributeRequest> attributes;
    private List<InstanceRequest> instances;
}