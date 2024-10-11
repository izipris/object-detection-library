package com.ontolligence.vision.object.detection.detectors;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.YoloV8Translator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Translator;
import com.ontolligence.vision.object.detection.configuration.YoloV8DetectorProperties;
import com.ontolligence.vision.object.detection.exceptions.DetectImageException;
import com.ontolligence.vision.object.detection.model.Detection;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.function.Function;

public class YoloV8Detector implements Detector {

  private final YoloV8DetectorProperties properties;
  private final Detector genericDetector;

  public YoloV8Detector(YoloV8DetectorProperties yoloV8DetectorProperties) {

    properties = yoloV8DetectorProperties;
    genericDetector = new GenericDetector(createCriteria(properties));
  }

  @Override
  public List<Detection> detectImage(Image image) throws DetectImageException {

    return genericDetector.detectImage(image, this::resizeBoundingBox);
  }

  @Override
  public List<Detection> detectImage(Image image, Function<BoundingBox, BoundingBox> boundingBoxTransformer) throws DetectImageException {

    return genericDetector.detectImage(image, boundingBox -> boundingBoxTransformer.apply(resizeBoundingBox(boundingBox)));
  }

  private Criteria<Image, DetectedObjects> createCriteria(YoloV8DetectorProperties yoloV8DetectorProperties) {

    try {
      Path path = Paths.get(yoloV8DetectorProperties.getModelPath());
      Pipeline pipeline = new Pipeline(new Resize(yoloV8DetectorProperties.getImageSize()), new ToTensor());
      Translator<Image, DetectedObjects> translator = YoloV8Translator.builder()
              .setPipeline(pipeline)
              .optSynset(yoloV8DetectorProperties.getOrderedClassNames())
              .optThreshold(yoloV8DetectorProperties.getConfidenceThreshold())
              .build();
      return Criteria.builder()
              .setTypes(Image.class, DetectedObjects.class)
              .optModelUrls(path.getParent().toString())
              .optModelName(path.getFileName().toString())
              .optTranslator(translator)
              .optEngine(yoloV8DetectorProperties.getEngineName())
              .optDevice(yoloV8DetectorProperties.getDevice())
              .build();

    } catch (Exception e) {
      throw new IllegalArgumentException("Failed to create criteria from the provided properties", e);
    }
  }

  private BoundingBox resizeBoundingBox(BoundingBox boundingBox) {

    Rectangle rec = boundingBox.getBounds();
    double x = rec.getX() / properties.getImageSize();
    double y = rec.getY() / properties.getImageSize();
    double width = rec.getWidth() / properties.getImageSize();
    double height = rec.getHeight() / properties.getImageSize();
    return new Rectangle(x, y, width, height);
  }

  @Override
  public void close() throws Exception {

    genericDetector.close();
  }
}
