package com.ontolligence.vision.object.detection.configuration;

import ai.djl.Device;
import com.ontolligence.vision.object.detection.configuration.YoloV8DetectorProperties;
import com.ontolligence.vision.object.detection.constants.YoloV8Constant;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@AllArgsConstructor
@NoArgsConstructor
@Builder
@Data
public class YoloV8DetectorPropertiesImpl implements YoloV8DetectorProperties {

  private String modelPath;
  private float confidenceThreshold;
  @Builder.Default
  private int imageSize = 640;
  @Builder.Default
  private List<String> orderedClassNames = YoloV8Constant.ORDERED_CLASSES;
  @Builder.Default
  private String engineName = "OnnxRuntime";
  private Device device;
}
