package com.example.try1231231;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.Getter;
import lombok.NoArgsConstructor;
import weka.core.Attribute;

import java.util.ArrayList;
import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Getter
public class PredictionRequest {
    private String modelName;
    private String name;
    private List<AttributeRequest> attributes;
    private List<Object> attributesMatrix;

    // getters and setters
}


