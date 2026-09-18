 Real-Time Object Detection Using YOLO and OpenCV

A command-line executable computer vision project that detects objects in images, videos, and live webcam input using **YOLO** and **OpenCV**.

The system displays each detected object's class name and confidence score and draws a bounding box around the object.

Features

- Real-time webcam object detection
- Image object detection
- Video object detection
- Bounding boxes around detected objects
- Object class labels
- Confidence scores
- Adjustable confidence threshold
- Adjustable YOLO inference image size
- Optional saving of processed image/video
- Fully executable from a terminal

 Technology Stack

- Python 3.10+
- YOLO via Ultralytics
- OpenCV
- Command-line interface using Python `argparse`

 Project Structure

```text
real-time-object-detection/
├── README.md
├── main.py
├── detect.py
├── config.py
├── requirements.txt
├── .gitignore
├── input/
│   └── README.md
├── models/
│   └── README.md
├── output/
│   └── .gitkeep
└── docs/
    └── project_report.md
```

 System Requirements

Recommended:

- Windows, Linux, or macOS
- Python 3.10 or newer
- 4 GB RAM minimum; 8 GB recommended
- Webcam for live detection
- Internet connection on the first run so the YOLO model can be downloaded automatically

A GPU is optional. The project can run on CPU, although inference is generally faster with a supported GPU.




2. Create a Virtual Environment

 Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

 Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

After activation, your terminal should show something similar to:

```text
(.venv)
```

 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

 4. YOLO Model

The default model is:

```text
yolo11n.pt
```

When the program first loads this model, Ultralytics can download the required model weights automatically.

The model is intentionally kept out of GitHub because model weight files can be large.

You can change the model with:

```bash
python main.py --source 0 --model yolo11n.pt
```

 5. Run with Webcam

Start your default webcam:

```bash
python main.py --source 0
```

If you have another camera:

```bash
python main.py --source 1
```

A detection window will open. Press:

```text
Q
```

to stop.

6. Run on an Image

Place an image inside `input/`.

Example:

```bash
python main.py --source input/sample.jpg
```

To save the detected image:

```bash
python main.py --source input/sample.jpg --output output/result.jpg
```

7. Run on a Video

```bash
python main.py --source input/video.mp4
```

To save the processed video:

```bash
python main.py --source input/video.mp4 --output output/result.mp4
```

Press `Q` to stop playback early.

8. Change Confidence Threshold

The default confidence threshold is `0.40`.

For example:

```bash
python main.py --source 0 --conf 0.60
```

Higher values generally reduce low-confidence detections.

9. Change Image Size

Default:

```text
640
```

Example:

```bash
python main.py --source 0 --imgsz 832
```

Larger inference sizes can improve detection of some objects but may reduce speed.

 10. Command-Line Arguments

| Argument | Required | Description | Example |
|---|---|---|---|
| `--source` | Yes | Webcam index, image path, or video path | `--source 0` |
| `--model` | No | YOLO model name/path | `--model yolo11n.pt` |
| `--conf` | No | Confidence threshold | `--conf 0.5` |
| `--imgsz` | No | YOLO inference size | `--imgsz 640` |
| `--output` | No | Save result to a file | `--output output/result.mp4` |

See all arguments:

```bash
python main.py --help
```

How the System Works

```text
Input
  │
  ├── Webcam
  ├── Image
  └── Video
       │
       ▼
   OpenCV reads frame
       │
       ▼
    YOLO model
       │
       ▼
 Object detection
       │
       ▼
Bounding boxes + class + confidence
       │
       ▼
 OpenCV annotation
       │
       ├── Display
       └── Optional output file
```

Detection Output

For every detected object, the application displays:

```text
object_name confidence
```

For example:

```text
person 0.91
bottle 0.84
cell phone 0.77
```

The confidence value ranges from 0 to 1.

Troubleshooting

`python` is not recognized

Install Python and ensure it is added to PATH. Then reopen the terminal.

 `pip install` fails

Try:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Webcam does not open

Try another camera index:

```bash
python main.py --source 1
```

Also check that another application is not currently using the camera.

Model download fails

Check your internet connection and run the program again.

No objects are detected

Try a lower confidence threshold:

```bash
python main.py --source input/sample.jpg --conf 0.25
```

 Limitations

- Detection accuracy depends on the trained YOLO model.
- CPU-only systems may have lower real-time FPS.
- Poor lighting or highly crowded scenes can reduce detection quality.
- The default pretrained model detects the classes included in its training dataset.

Future Enhancements

- Add object tracking using ByteTrack or another tracker.
- Display FPS.
- Add a detection counter.
- Add region-of-interest detection.
- Add custom training for college-specific objects.
- Add CSV logging of detected objects.
- Add distance estimation.
- Add GPU configuration.
- Add a web/API deployment option.

 