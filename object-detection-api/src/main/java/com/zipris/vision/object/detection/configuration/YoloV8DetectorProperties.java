package com.zipris.vision.object.detection.configuration;

import java.util.List;

public interface YoloV8DetectorProperties {

  String getModelPath();

  float getConfidenceThreshold();

  int getImageSize();

  List<String> getOrderedClassNames();

  String getEngineName();
}
