package com.example.ml;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonNode;
import weka.core.Attribute;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class AttributeDeserializer extends JsonDeserializer<Attribute> {
    @Override
    public Attribute deserialize(JsonParser jp, DeserializationContext ctxt) throws IOException, JsonProcessingException {
        ObjectCodec oc = jp.getCodec();
        JsonNode node = oc.readTree(jp);

        String name = node.get("name").asText();
        String type = node.get("type").asText();

        if ("numeric".equals(type)) {
            return new Attribute(name);
        } else if ("nominal".equals(type)) {
            List<String> values = new ArrayList<>();
            JsonNode valuesNode = node.get("values");
            for (JsonNode valueNode : valuesNode) {
                values.add(valueNode.asText());
            }
            return new Attribute(name, values);
        } else {
            throw new RuntimeException("Unsupported attribute type: " + type);
        }
    }
}
