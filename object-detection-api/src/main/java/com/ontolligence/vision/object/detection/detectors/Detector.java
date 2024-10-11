package com.ontolligence.vision.object.detection.detectors;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.BoundingBox;
import com.ontolligence.vision.object.detection.exceptions.DetectImageException;
import com.ontolligence.vision.object.detection.model.Detection;

import java.util.List;
import java.util.function.Function;

public interface Detector extends AutoCloseable {

  default List<Detection> detectImage(Image image) throws DetectImageException {

    return detectImage(image, Function.identity());
  }

  List<Detection> detectImage(Image image, Function<BoundingBox, BoundingBox> boundingBoxTransformer)
          throws DetectImageException;
}
