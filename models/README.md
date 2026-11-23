# Models

Place the exported ONNX model here, for example `models/yolo11n.onnx`.

The recommended export (Ultralytics YOLO11n, COCO classes, 640 input):

```bash
pip install ultralytics onnx onnxruntime
yolo export model=yolo11n.pt format=onnx imgsz=640 opset=12 simplify=True dynamic=False
```

YOLOv8n also works:

```bash
yolo export model=yolov8n.pt format=onnx imgsz=640 opset=12 simplify=True dynamic=False
```

The app expects the raw Ultralytics detection output `[1, 84, 8400]` (or the
transposed `[1, 8400, 84]`). The 84 channels are 4 box values plus 80 COCO
class scores. Legacy 85-channel YOLOv5-style output is also accepted.

Model files and the ONNX Runtime shared library are deliberately ignored by
git. Use `scripts/fetch-model.ps1` / `scripts/fetch-model.sh` for a quick smoke
test model, or export your own using the commands above.
