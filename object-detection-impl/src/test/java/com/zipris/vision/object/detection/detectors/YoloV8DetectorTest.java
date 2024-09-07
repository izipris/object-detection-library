package com.zipris.vision.object.detection.detectors;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.Rectangle;
import com.zipris.vision.object.detection.configuration.YoloV8DetectorProperties;
import com.zipris.vision.object.detection.configuration.YoloV8DetectorPropertiesImpl;
import com.zipris.vision.object.detection.model.Detection;
import lombok.SneakyThrows;
import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.io.File;
import java.util.List;

import static java.util.Arrays.asList;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class YoloV8DetectorTest {

  private final File YOLOV8_MODEL_FILE = new File("src/test/resources/yolov8n.onnx");
  private final File DOG_IMAGE_FILE = new File("src/test/resources/golden.jpeg");
  private static final List<String> CLASSES =
          asList("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
                  "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                  "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                  "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                  "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
                  "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush");

  @Test
  @SneakyThrows
  public void shouldDetectOneDogInImage() {

    YoloV8DetectorProperties properties = generateProperties();
    List<Detection> detections;
    try (YoloV8Detector detector = new YoloV8Detector(properties)) {
      Image image = ImageFactory.getInstance().fromImage(ImageIO.read(DOG_IMAGE_FILE));
      detections = detector.detectImage(image);
    }
    assertOneDog(detections);
  }

  @Test
  @SneakyThrows
  public void shouldDetectAndTransformBoundingBox() {

    YoloV8DetectorProperties properties = generateProperties();
    int transformedSize = properties.getImageSize() * 2;
    List<Detection> detections;
    try (YoloV8Detector detector = new YoloV8Detector(properties)) {
      Image image = ImageFactory.getInstance().fromImage(ImageIO.read(DOG_IMAGE_FILE));
      detections = detector.detectImage(image, boundingBox -> {

        Rectangle rectangle = new Rectangle(0, 0, transformedSize, transformedSize);
        BoundingBox newBoundingBox = mock(BoundingBox.class);
        when(newBoundingBox.getBounds()).thenReturn(rectangle);
        return newBoundingBox;
      });
    }
    assertOneDog(detections);
    assertEquals(transformedSize, detections.getFirst().getBoundingBox().getBounds().getWidth());
    assertEquals(transformedSize, detections.getFirst().getBoundingBox().getBounds().getHeight());
  }

  private YoloV8DetectorProperties generateProperties() {

    return YoloV8DetectorPropertiesImpl.builder()
            .modelPath(YOLOV8_MODEL_FILE.getPath())
            .confidenceThreshold(0.3f)
            .orderedClassNames(CLASSES)
            .build();
  }

  private void assertOneDog(List<Detection> detections) {

    assertNotNull(detections);
    assertFalse(detections.isEmpty());
    assertEquals(1, detections.size());
    detections.forEach(detection -> assertEquals("dog", detection.getName()));
  }
}