### 📊 Dataset Access (Roboflow Integration)
This vehicle dataset was independently collected and annotated from CCTV footage at intersections in Malang City, then managed via the Roboflow platform.

To download the dataset ready for training on the YOLOv12 architecture:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("afifulislam").project("revised-thesis-dataset")
version = project.version(4)
dataset = version.download("yolov12")
```
