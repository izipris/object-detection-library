package com.zipris.vision.object.detection.detectors;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import com.zipris.vision.object.detection.exceptions.DetectImageException;
import com.zipris.vision.object.detection.model.Detection;
import com.zipris.vision.object.detection.model.DetectionImpl;

import java.io.IOException;
import java.util.List;
import java.util.function.Function;

import static java.util.stream.Collectors.toList;

public class GenericDetector implements Detector {

  private final ZooModel<Image, DetectedObjects> model;
  private final Predictor<Image, DetectedObjects> predictor;

  public GenericDetector(Criteria<Image, DetectedObjects> criteria) {
    try {
      model = criteria.loadModel();
    } catch (IOException | ModelNotFoundException | MalformedModelException e) {
      throw new IllegalArgumentException("Failed to load model from the provided criteria", e);
    }
    predictor = model.newPredictor();
  }

  @Override
  public List<Detection> detectImage(Image image, Function<BoundingBox, BoundingBox> boundingBoxTransformer) throws DetectImageException {

    try {
      DetectedObjects detectedObjects = predictor.predict(image);
      return generateDetections(detectedObjects, boundingBoxTransformer);
    } catch (TranslateException e) {
      throw new DetectImageException("Failed to predict", e);
    }
  }

  private List<Detection> generateDetections(DetectedObjects detectedObjects, Function<BoundingBox, BoundingBox> boundingBoxTransformer) {

    return detectedObjects.items()
            .stream()
            .parallel()
            .map(detectedObject -> DetectionImpl.builder()
                    .name(detectedObject.getClassName())
                    .probability(detectedObject.getProbability())
                    .boundingBox(boundingBoxTransformer.apply(((DetectedObjects.DetectedObject) detectedObject).getBoundingBox()))
                    .build())
            .collect(toList());
  }

  @Override
  public void close() {

    predictor.close();
    model.close();
  }
}
