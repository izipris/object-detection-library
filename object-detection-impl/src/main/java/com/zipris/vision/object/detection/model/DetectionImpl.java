package com.zipris.vision.object.detection.model;

import ai.djl.modality.cv.output.BoundingBox;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@NoArgsConstructor
@AllArgsConstructor
@Builder
@Data
public class DetectionImpl implements Detection {

  private BoundingBox boundingBox;
  private String name;
  private double probability;
}
