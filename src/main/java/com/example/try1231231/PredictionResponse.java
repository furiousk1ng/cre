package com.example.try1231231;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class PredictionResponse {
    private List<String> labels;
    private  double accuracy;
}