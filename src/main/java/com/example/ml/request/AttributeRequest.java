package com.example.ml.request;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class AttributeRequest {
    private String name;
    private String type;
    private List<String> nominalValues;
}


