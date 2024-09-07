package com.zipris.vision.object.detection.configuration;

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
  private List<String> orderedClassNames;
  @Builder.Default
  private String engineName = "OnnxRuntime";
}
