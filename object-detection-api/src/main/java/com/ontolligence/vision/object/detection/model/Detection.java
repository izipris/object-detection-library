package com.ontolligence.vision.object.detection.model;

import ai.djl.modality.cv.output.BoundingBox;

public interface Detection {

  BoundingBox getBoundingBox();

  String getName();

  double getProbability();
}
