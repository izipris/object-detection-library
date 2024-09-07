package com.zipris.vision.object.detection.detectors;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import com.zipris.vision.object.detection.exceptions.DetectImageException;
import com.zipris.vision.object.detection.model.Detection;
import lombok.SneakyThrows;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.List;

import static java.util.Collections.singletonList;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class GenericDetectorTest {

  private static final String CLASS = "Car";
  private static final double PROBABILITY = 0.3;
  private static final int RECTANGLE_EDGE = 10;

  @Test
  @SneakyThrows
  public void shouldPredictWithoutTransformation() {

    Criteria<Image, DetectedObjects> criteria = mockCriteria();
    GenericDetector detector = new GenericDetector(criteria);
    List<Detection> detections = detector.detectImage(mock(Image.class));
    assertNotNull(detections);
    assertFalse(detections.isEmpty());
    assertEquals(CLASS, detections.getFirst().getName());
    assertEquals(PROBABILITY, detections.getFirst().getProbability());
    assertEquals(RECTANGLE_EDGE, detections.getFirst().getBoundingBox().getBounds().getWidth());
    assertEquals(RECTANGLE_EDGE, detections.getFirst().getBoundingBox().getBounds().getHeight());
  }

  @Test
  @SneakyThrows
  public void shouldPredictWithTransformation() {

    Criteria<Image, DetectedObjects> criteria = mockCriteria();
    GenericDetector detector = new GenericDetector(criteria);
    List<Detection> detections = detector.detectImage(mock(Image.class), boundingBox -> {

      Rectangle rectangle = new Rectangle(0, 0, 1, 1);
      BoundingBox newBoundingBox = mock(BoundingBox.class);
      when(newBoundingBox.getBounds()).thenReturn(rectangle);
      return newBoundingBox;
    });
    assertNotNull(detections);
    assertFalse(detections.isEmpty());
    assertEquals(CLASS, detections.getFirst().getName());
    assertEquals(PROBABILITY, detections.getFirst().getProbability());
    assertEquals(1, detections.getFirst().getBoundingBox().getBounds().getWidth());
    assertEquals(1, detections.getFirst().getBoundingBox().getBounds().getHeight());

  }

  @Test
  @SneakyThrows
  public void shouldThrowDetectImageException() {

    Criteria<Image, DetectedObjects> criteria = mock(Criteria.class);
    ZooModel<Image, DetectedObjects> model = mock(ZooModel.class);
    Predictor<Image, DetectedObjects> predictor = mock(Predictor.class);
    when(criteria.loadModel()).thenReturn(model);
    when(model.newPredictor()).thenReturn(predictor);
    when(predictor.predict(any())).thenThrow(TranslateException.class);
    GenericDetector detector = new GenericDetector(criteria);
    assertThrows(DetectImageException.class, () -> detector.detectImage(mock(Image.class)));
  }

  @Test
  @SneakyThrows
  public void shouldFailOnModelNotFoundException() {

    Criteria<Image, DetectedObjects> criteria = mock(Criteria.class);
    when(criteria.loadModel()).thenThrow(ModelNotFoundException.class);
    assertLoadModelFailure(criteria);
  }

  @Test
  @SneakyThrows
  public void shouldFailOnMalformedModelException() {

    Criteria<Image, DetectedObjects> criteria = mock(Criteria.class);
    when(criteria.loadModel()).thenThrow(MalformedModelException.class);
    assertLoadModelFailure(criteria);
  }

  @Test
  @SneakyThrows
  public void shouldFailOnIOException() {

    Criteria<Image, DetectedObjects> criteria = mock(Criteria.class);
    when(criteria.loadModel()).thenThrow(IOException.class);
    assertLoadModelFailure(criteria);
  }

  private void assertLoadModelFailure(Criteria<Image, DetectedObjects> criteria) {

    assertThrows(IllegalArgumentException.class, () -> new GenericDetector(criteria));
  }

  @SneakyThrows
  private Criteria<Image, DetectedObjects> mockCriteria() {

    Criteria<Image, DetectedObjects> criteria = mock(Criteria.class);
    ZooModel<Image, DetectedObjects> model = mock(ZooModel.class);
    Predictor<Image, DetectedObjects> predictor = mock(Predictor.class);
    DetectedObjects detectedObjects = mockDetectedObjects();
    when(criteria.loadModel()).thenReturn(model);
    when(model.newPredictor()).thenReturn(predictor);
    when(predictor.predict(any())).thenReturn(detectedObjects);
    return criteria;
  }

  private DetectedObjects mockDetectedObjects() {

    Rectangle rectangle = new Rectangle(0, 0, RECTANGLE_EDGE, RECTANGLE_EDGE);
    DetectedObjects detectedObjects = mock(DetectedObjects.class);
    DetectedObjects.DetectedObject firstObject = mock(DetectedObjects.DetectedObject.class);
    BoundingBox boundingBox = mock(BoundingBox.class);
    when(boundingBox.getBounds()).thenReturn(rectangle);
    when(firstObject.getClassName()).thenReturn(CLASS);
    when(firstObject.getProbability()).thenReturn(PROBABILITY);
    when(firstObject.getBoundingBox()).thenReturn(boundingBox);
    when(detectedObjects.items()).thenReturn(singletonList(firstObject));
    return detectedObjects;
  }

}