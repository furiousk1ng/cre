package com.example.ml.request;

import com.example.ml.request.AttributeRequest;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.Getter;
import lombok.NoArgsConstructor;

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


