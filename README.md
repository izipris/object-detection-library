# Object Detection Library

Object Detection Library (ODL, pronounced: O-DEL) is a Java library which utilizes [djl.ai](https://djl.ai/) to provide a plug-and-play
components for objects detection in images.

## Usage

Import the following dependency to your `pom.xml`:

```xml
TBD
```

### YoloV8 Detector

ODL provides a detector for the popular YoloV8 model, which can be composed over a classical YoloV8 model, or a custom-trained one:

```java
import java.io.File;

public List<Detections> detectDogs() {

  YoloV8DetectorProperties properties = YoloV8DetectorPropertiesImpl.builder()
          .modelPath("~/workspace/models/yolov8n.onnx")
          .confidenceThreshold(0.3f)
          .build();
  try (YoloV8Detector detector = new YoloV8Detector(properties)) {

    Image image = ImageFactory.getInstance().fromImage(ImageIO.read(new File("~/workspace/images/dogs.jpeg")));
    return detector.detectImage(image);
  }
}
```

### Generic Detector
ODL provides a generic detector which can be used with a custom-defined [djl.ai Criteria](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/repository/zoo/Criteria.html):
```java
public List<Detections> detectSomething() {
  Criteria<Image, Classifications> criteria = Criteria.builder()
          .setTypes(Image.class, Classifications.class)
          .optTranslator(ImageClassificationTranslator.builder().setSynsetArtifactName("synset.txt").build())
          .optModelUrls("file:///var/models/my_resnet50")
          .optModelName("resnet50")
          .build();
  try (GenericDetector detector = new GenericDetector(criteria)) {
    Image image = ImageFactory.getInstance().fromImage(ImageIO.read(new File("~/workspace/images/dogs.jpeg")));
    return detector.detectImage(image);
  }
}
```