package com.zipris.vision.object.detection.model;

import ai.djl.modality.cv.output.BoundingBox;

public interface Detection {

  BoundingBox getBoundingBox(); // TODO: wrap my own object?

  String getName();

  double getProbability();
}
