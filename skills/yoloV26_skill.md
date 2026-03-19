# Agent Skill: YOLO26 Object Detection

## 1. Tool Context (For LLM Understanding)
You are equipped with the YOLO26 object detection model. YOLO26 is a state-of-the-art model released by Ultralytics in January 2026. It features end-to-end NMS-free (Non-Maximum Suppression) inference and is highly optimized. 

**Crucial API Note:** YOLO26 is fully integrated into the standard `ultralytics` Python package. You interact with it using the exact same API syntax as YOLOv8. You load it using `from ultralytics import YOLO`.

## 2. Description
Use this tool to detect, classify, count, and locate objects within an image. Trigger this skill whenever the user asks questions like "what is in this image?", "count the items," or requires bounding box coordinates for specific objects.

## 3. Required Inputs
When writing the execution code for this tool, you must accept the following inputs:
* `image_path` (string): The absolute or relative file path to the target image.
* `model_weight_path` (string): The path to the fine-tuned YOLO26 weights (e.g., `"model_weights/best.pt"`). If not provided, default to `"yolo26n.pt"`.
* `confidence_threshold` (float, optional): Minimum confidence score (0.0 to 1.0). Default is `0.25`.

## 4. Expected Execution Logic
When you execute this skill, write Python code that follows this exact pattern:

```python
from ultralytics import YOLO

# 1. Initialize the model
model = YOLO(model_weight_path)

# 2. Run inference
results = model.predict(source=image_path, conf=confidence_threshold)

# 3. Extract bounding boxes, labels, and confidences from the 'results' object