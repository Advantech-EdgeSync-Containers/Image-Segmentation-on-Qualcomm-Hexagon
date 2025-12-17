import argparse
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
import sys
import signal
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate
import time
import threading
import queue
import subprocess
import re
import glob
import select
import gi
gi.require_version('Gst', '1.0')
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GLib
USE_THREAD = 0

shutdown_requested = False

main_frame_height = 0
main_frame_width = 0
# Declare as global variables, can be updated based trained model image size
img_width = 640
img_height = 640


LABEL_PATH = "/etc/labels/coco_labels.txt"
input_details = None
output_details = None

SCORE_THRESHOLD = 0.20
NMS_IOU_THRESHOLD = 0.5
INFERENCE_IMG_SIZE = 640
MAX_DETS = 100

ANCHORS = [[[81, 82], [135, 169], [344, 319]], [[23, 27], [37, 58], [81, 82]]]
SIGMOID_FACTOR = [1.05, 1.05]
NUM_ANCHORS = 3
STRIDES = [32, 16]
GRID_SIZES = [int(INFERENCE_IMG_SIZE / s) for s in STRIDES]

COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

class QuitListener(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_flag = False

    def run(self):
        global shutdown_requested
        print("[INFO] Press 'q' Then 'Enter' to exit the application.")
        while not self.stop_flag:
            try:
                if sys.stdin.read(1) == 'q':
                    print("[INFO] Quit command received.")
                    shutdown_requested = True
                    return
            except Exception:
                # stdin not available, try again
                time.sleep(0.1)

class VideoCaptureThread(threading.Thread):
    def __init__(self, args, queue_size=10):
        super().__init__()
        self.cap = cv2.VideoCapture(args.source)
        set_resolution = args.source.startswith("/dev/video")
        self.enable_drop = args.source.startswith("rtsp://")# Enable frame drop in RTSP only to reduce H264 error.
        self.show_warning = False

        if set_resolution:
            # Enable MJPG compression and desired resolution + frame rate for live camera
            if args.cam_format == "MJPG":
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) #Fix: Webcam formate set issue

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))

        # Convert the integer FourCC code to a string
        self.fourcc_str = "".join([chr((self.fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

        print(f"video format (FourCC): {self.fourcc_str}")
        self.queue = queue.Queue(maxsize=queue_size)
        self.running = True

    def run(self):
        while self.running:
            if self.queue.full() and self.enable_drop:
                try:
                    self.queue.get_nowait()  # Drop the oldest frame
                    #print("Warning: YOLO is lagging, dropping frame")
                except queue.Empty:
                    pass
        
            elif not self.queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.running = False
                    self.cap.release()
                    break
                else:
                    if self.fourcc_str == 'YUYV':
                        if len(frame.shape) == 3 and frame.shape[2] == 2:
                            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
                        else:
                            if self.show_warning == False:
                                print("Warning: Expected 2-channel YUYV frame, got", frame.shape)
                                self.show_warning = True
                    self.queue.put(frame)
            else:
                time.sleep(0.01)  # avoid busy-waiting
                #print("Warning: YOLO is lagging, waiting...")

    def stop(self):
        self.running = False
        self.join()  # <- wait for thread to finish
        self.cap.release()
        

class LetterBox:
    """Resizes and reshapes images while maintaining aspect ratio by adding padding, suitable for YOLO models."""

    def __init__(
        self, new_shape=(img_width, img_height), auto=False, scaleFill=False, scaleup=True, center=True, stride=32
    ):
        """Initializes LetterBox with parameters for reshaping and transforming image while maintaining aspect ratio."""
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left

    def __call__(self, labels=None, image=None):
        """Return updated labels and image with added border."""
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels."""
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels


class YolovTFLite:
    """Class for performing object detection using YOLOv8 model converted to TensorFlow Lite format."""

    def __init__(self, tflite_model, input_image, confidence_thres, iou_thres):
        """
        Initializes an instance of the Yolov8TFLite class.

        Args:
            tflite_model: Path to the TFLite model.
            input_image: Path to the input image.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """
        self.tflite_model = tflite_model
        self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # Load the class names from the COCO dataset
        # Load labels (if available)
        if os.path.exists(LABEL_PATH):
            with open(LABEL_PATH, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
        else:
            self.classes = COCO_CLASS_NAMES

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.letterbox = LetterBox(new_shape=[img_width, img_height], auto=False, stride=32)


    def draw_detections(self, img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """
        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img,
            (int(label_x), int(label_y - label_height)),
            (int(label_x + label_width), int(label_y + label_height)),
            color,
            cv2.FILLED,
        )

        # Draw the label text on the image
        cv2.putText(img, label, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)

    def preprocess(self):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        #print("image before", self.img)
        # Get the height and width of the input image
        self.img_height, self.img_width = self.input_image.shape[:2]

        image = self.letterbox(image=self.input_image)
        image = [image]
        image = np.stack(image)
        image = image[..., ::-1].transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(image)
        # n, h, w, c
        return img.astype(np.float32)/255

    def postprocess_yolo8(self, input_image, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """
        boxes = []
        scores = []
        class_ids = []
        for pred in output:
            pred = np.transpose(pred)
            for box in pred:
                x, y, w, h = box[:4]
                x1 = x - w / 2
                y1 = y - h / 2
                boxes.append([x1, y1, w, h])
                idx = np.argmax(box[4:])
                scores.append(box[idx + 4])
                class_ids.append(idx)

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            gain = min(img_width / self.img_width, img_height / self.img_height)
            pad = (
                round((img_width - self.img_width * gain) / 2 - 0.1),
                round((img_height - self.img_height * gain) / 2 - 0.1),
            )
            box[0] = (box[0] - pad[0]) / gain
            box[1] = (box[1] - pad[1]) / gain
            box[2] = box[2] / gain
            box[3] = box[3] / gain
            score = scores[i]
            class_id = class_ids[i]
            if score > 0.5:
                #print(box, score, class_id)
                # Draw the detection on the input image
                self.draw_detections(input_image, box, score, class_id)

        return input_image

    def postprocess_yolo8_opt(self, input_image, output):
        output = output[0]  # remove batch dimension
        output = output.T  # shape: (N, 85) — [x, y, w, h, conf, class_scores...]

        scores = output[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

        mask = confidences > self.confidence_thres
        boxes = output[mask, :4]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return input_image

        # Convert xywh to xyxy
        boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]      # x2
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]      # y2

        # Scale to original image size
        gain = min(img_width / self.img_width, img_height / self.img_height)
        pad_x = (img_width - self.img_width * gain) / 2
        pad_y = (img_height - self.img_height * gain) / 2

        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y
        boxes /= gain

        boxes = boxes.astype(np.int32).tolist()
        confidences = confidences.tolist()
        class_ids = class_ids.tolist()

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_thres, self.iou_thres)
        
        for i in indices:
            i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
            x1, y1, x2, y2 = boxes[i]
            score = confidences[i]
            class_id = class_ids[i]
            w, h = x2 - x1, y2 - y1
            self.draw_detections(input_image, (x1, y1, w, h), score, class_id)

        return input_image
        
        
    def detect_on_frame_yolo8n_full_quanta(self, frame):     
        """
        Performs inference using a TFLite model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """
        # Store the shape of the input for later use
        input_shape = input_details[0]["shape"]
        self.input_width = input_shape[1]
        self.input_height = input_shape[2]

        # Preprocess the image data
        img_data = self.preprocess()
        # Set the input tensor to the interpreter
        img_data = img_data.transpose((0, 2, 3, 1))
        
  
        # Quantization handling
        if input_details[0]['dtype'] == np.uint8:
            img_data = (img_data * 255).astype(np.uint8)
        elif input_details[0]['dtype'] == np.int8:
            scale, zero_point = input_details[0]["quantization"]
            img_data = (img_data / scale + zero_point).astype(np.int8)
        else:
            img_data = img_data.astype(np.float32)  # for FP32 model
        self.model.set_tensor(input_details[0]["index"], img_data)

        # Run inference
        self.model.invoke()

        # Get the output tensor from the interpreter
        output = self.model.get_tensor(output_details[0]["index"])
        scale, zero_point = output_details[0]["quantization"]
        output = (output.astype(np.float32) - zero_point) * scale

        output[:, [0, 2]] *= img_width
        output[:, [1, 3]] *= img_height
        # Perform post-processing on the outputs to obtain output image.
        return self.postprocess_yolo8_opt(frame, output)


ALLOWED_RESOLUTIONS = [(640, 360), (640, 480), (1280, 720), (1920, 1080)]

def get_video_nodes():
    return glob.glob("/dev/video*")

def is_streaming(dev, num_buffers=3):
    """Check if the video node can stream."""
    try:
        subprocess.run(
            ["timeout", "5", "gst-launch-1.0", "-q",
             "v4l2src", f"device={dev}", f"num-buffers={num_buffers}", "!", "fakesink"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError:
        return False

def get_format_info(dev):
    """Return only supported (w,h,pixfmt) among 640x480, 1280x720, 1920x1080."""
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", dev, "--list-formats-ext"],
            capture_output=True, text=True, check=True
        )
        output = result.stdout
        formats = []
        current_pixfmt = None

        for line in output.splitlines():
            line = line.strip()
            if line.startswith("["):
                pixfmt_match = re.search(r"'(\w+)'", line)
                if pixfmt_match:
                    current_pixfmt = pixfmt_match.group(1)
            elif "Size:" in line and "Discrete" in line:
                size_match = re.search(r"(\d+)x(\d+)", line)
                if size_match and current_pixfmt:
                    w, h = map(int, size_match.groups())
                    if (w, h) in ALLOWED_RESOLUTIONS and current_pixfmt in ("YUYV", "BGRx", "MJPG"):
                        formats.append((w, h, current_pixfmt))
        return formats
    except subprocess.CalledProcessError:
        return []

def select_webcam():
    print("Available supported video nodes for streaming:")
    print("-------------------------------------")
    nodes = get_video_nodes()
    working_nodes = []

    # Check each node
    for dev in nodes:
        if is_streaming(dev):
            formats = get_format_info(dev)
            if not formats:
                continue
            print(f"{len(working_nodes)+1}. {dev}")
            for idx, (w, h, fmt) in enumerate(formats, start=1):
                print(f"   [{idx}] {w}x{h}, PixelFormat: {fmt}")
            working_nodes.append((dev, formats))

    if not working_nodes:
        print("No working video nodes found with standard resolutions.")
        return

    # Ask user to select node
    while True:
        choice = input(f"\nSelect a video node [1-{len(working_nodes)}]: ")
        if choice.isdigit():
            choice = int(choice)
            if 1 <= choice <= len(working_nodes):
                selected_node, formats = working_nodes[choice-1]
                break
        print("Invalid choice, try again.")

    # Show available standard resolutions for selected node
    print(f"\nAvailable supported resolutions for {selected_node}:")
    for idx, (w, h, fmt) in enumerate(formats, start=1):
        print(f"   [{idx}] {w}x{h}, PixelFormat: {fmt}")

    # Ask user to select resolution
    while True:
        res_choice = input(f"Select a resolution [1-{len(formats)}]: ")
        if res_choice.isdigit():
            res_choice = int(res_choice)
            if 1 <= res_choice <= len(formats):
                width, height, pixfmt = formats[res_choice-1]
                break
        print("Invalid choice, try again.")

    # Display final selection
    print("\nSelected video node parameters:")
    print(f"Device: {selected_node}")
    print(f"Resolution: {width}x{height}")
    print(f"Pixel Format: {pixfmt}")
    print(f"Framerate: 30/1")

    return {
        "device": selected_node,
        "width": width,
        "height": height,
        "pixfmt": pixfmt,
        "framerate": 30
    }

def get_video_properties(args):
    cap = cv2.VideoCapture(args.source)
    
    set_resolution = args.source.startswith("/dev/video")

    if set_resolution:
        # Enable MJPG compression and desired resolution + frame rate for live camera
        if args.cam_format == "MJPG":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) #Fix: Webcam formate set issue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
        cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        #raise RuntimeError("Failed to open video source")
        print(f"\n[INFO] Failed to open video source")
        shutdown_requested = True
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()
    return width, height, fps

def signal_handler(sig, frame):
    global shutdown_requested
    print(f"\n[INFO] Caught signal {sig}. Requesting shutdown...")
    shutdown_requested = True

 
def main(args):
    global img_width, img_height, input_details, output_details
    global shutdown_requested
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGHUP, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)


    # Set the environment
    #os.environ["XDG_RUNTIME_DIR"] = "/dev/socket/weston"
    #os.environ["WAYLAND_DISPLAY"] = "wayland-1"
    frame_count = 0

    # Initialize GStreamer
    Gst.init(None)
    
    def periodic_check():
        if shutdown_requested:
            print("[INFO] Main loop termination requested (from periodic_check).")
            pipeline.set_state(Gst.State.NULL)
            return False  # stop timer
        return True  # continue timer

    GLib.timeout_add(200, periodic_check)   # check every 200 ms 

    # Set backend_type value to 'htp'
    TFLiteDelegate_lib_path = '/workspace/libs/libQnnTFLiteDelegate.so'
    delegate = load_delegate(
       TFLiteDelegate_lib_path,
       options={
               'backend_type': 'htp',
               'htp_performance_mode': 3,
               }
      )        
    interpreter = Interpreter(model_path=args.model, experimental_delegates=[delegate])
    interpreter.allocate_tensors()
    detection = YolovTFLite(args.model, None, args.conf_thres, args.iou_thres)
    detection.model = interpreter
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    prev_frame_time = 0
    new_frame_time = 0
    fps = 0
    frame = None
    # Video source - GStreamer pipeline or camera
    main_frame_width, main_frame_height, source_fps = get_video_properties(args)
    source_fps_numerator = int(round(source_fps))
    source_fps_denominator = 1
    
    if args.source.startswith("/dev/video"):
        print("Requested Resolution:",args.cam_width,"x",args.cam_height)
    print("Current Resolution:",main_frame_width,"x",main_frame_height)
    print("FPS:",source_fps)
    
    cap_thread = VideoCaptureThread(args)
    cap_thread.start()
    
    quit_thread = QuitListener() 
    quit_thread.start()   

    
    # Optional video writer
    out_writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(args.save, fourcc, source_fps, (main_frame_width, main_frame_height))

    # Setup GStreamer pipeline for display
    # Decide sync behavior based on your conditions
    use_sync = not (args.source.startswith("/dev/video") or args.source.startswith("rtsp://") or source_fps > 30)
    print(f"Sync enabled: {use_sync}")
    
    gst_pipeline = (
        f'appsrc name=src is-live=true block=true format=time ! '
        f'videoconvert ! fpsdisplaysink video-sink=glimagesink sync={str(use_sync).lower()}'
    )
    pipeline = Gst.parse_launch(gst_pipeline)

    appsrc = pipeline.get_by_name("src")
    # Setup caps, assuming 3 channel BGR
    caps = Gst.Caps.from_string(
        f"video/x-raw,format=BGR,width={main_frame_width},height={main_frame_height},framerate={source_fps_numerator}/{source_fps_denominator}"
    )
    
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_gst_message(bus, message, loop=None):
        global shutdown_requested
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"[GSTREAMER ERROR] {err}\n{debug}")
            shutdown_requested = True
        elif t == Gst.MessageType.EOS:
            print("[INFO] End-of-stream received from GStreamer")
            shutdown_requested = True
        return True

    bus.connect("message", on_gst_message)

    # Set frame duration based on input FPS
    appsrc.set_property("caps", caps) 
    appsrc.set_property("format", Gst.Format.TIME)
    pipeline.set_state(Gst.State.PLAYING)

    duration = Gst.util_uint64_scale_int(1, Gst.SECOND, source_fps_numerator)

    def push_frame_to_gst(frame):
        nonlocal frame_count
        frame = cv2.resize(frame, (main_frame_width, main_frame_height))  # Ensure match with caps
        data = frame.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.duration = duration
        buf.pts = buf.dts = frame_count * duration
        frame_count += 1
        retval = appsrc.emit("push-buffer", buf)
        if retval != Gst.FlowReturn.OK:
            print("Warning: push-buffer returned", retval)

    try:
        while True:
            if shutdown_requested:
                print("[INFO] Shutdown requested. Breaking main loop.")
                break

            if not cap_thread.queue.empty():
                frame = cap_thread.queue.get()
            else:
                if not cap_thread.running and cap_thread.queue.empty():
                    print(f"FPS: {fps:.2f}")
                    print("End of stream detected, exiting main loop.")
                    break
                continue
            #main_frame_height, main_frame_width = frame.shape[:2]
            detection.input_image = frame

            output_image = detection.detect_on_frame_yolo8n_full_quanta(frame)
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            
            # Draw FPS on the top-left corner
            #print(f"FPS: {fps:.2f}")
            cv2.putText(output_image, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Push frame to GStreamer display
            push_frame_to_gst(output_image)
            if out_writer:
                out_writer.write(output_image)


            # Check for quit (non-blocking)
            #if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            #    c = sys.stdin.read(1)
            #    if c == 'q':
            #        break

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        if out_writer:
            out_writer.release()
        cap_thread.stop()
        quit_thread.stop_flag = True
        appsrc.emit("end-of-stream")
        pipeline.set_state(Gst.State.NULL)


def validate_probability(value):
    f = float(value)
    if not (0.0 <= f <= 1.0):
        raise argparse.ArgumentTypeError("Threshold must be between 0.0 and 1.0")
    return f

ALLOWED_FOURCC = {"YUY","YUY2", "BGRx", "MJPG", "GRAY8","GREY"}
def validate_cam_format(fmt):
    fmt = fmt.upper()
    if fmt not in ALLOWED_FOURCC:
        raise argparse.ArgumentTypeError(
            f"Invalid camera format '{fmt}'. Supported: {', '.join(ALLOWED_FOURCC)}"
        )
    return fmt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 TFLite Object Detection with GStreamer Display")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the TFLite YOLOv8 model",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="/etc/media/video.mp4",
        help=(
            "Input source. Can be:\n"
            "  • A video file path (e.g., /path/video.mp4)\n"
            "  • A camera device (e.g., /dev/video0)\n"
            "  • An RTSP stream (e.g., rtsp://<ip>:<port>/path)\n"
            "Default: /etc/media/video.mp4"
        ),
    )
    parser.add_argument("--conf-thres", type=validate_probability, default=0.5, help="Object confidence threshold (0.0–1.0). Default: 0.5")
    parser.add_argument("--iou-thres", type=validate_probability, default=0.5, help="IoU threshold for Non-Maximum Suppression (0.0–1.0). Default: 0.5")
    parser.add_argument("--cam-width" , type=int, default=1920, help="Camera capture width when using /dev/video* as source. Default: 1920.")
    parser.add_argument("--cam-height", type=int, default=1080, help="Camera capture height when using /dev/video* as source. Default: 1080.")
    parser.add_argument("--cam-format", type=validate_cam_format, default="YUY2",
                        help=(
                            "Camera pixel format (V4L2). Allowed values:\n"
                            "  'YUY', 'YUY2', 'BGRx', 'MJPG', 'GRAY8', 'GREY'\n"
                            "Default: YUY2.\n"
                            "Note: The actual supported formats depend on the camera hardware."
                        ))
    parser.add_argument("--save", type=str, default=None, help=(
        "Enable saving the output video.\n"
        "Provide the file path where the MP4 should be saved, e.g.: --save output.mp4\n"
        "If omitted, saving is disabled."
    ))
    args = parser.parse_args()
    
    
    # Validate model file exists
    if not os.path.isfile(args.model):
        print(f"\n[ERROR] Model file not found at path: {args.model}\n")
        sys.exit(1)
    else:
        print(f"[INFO] Model file found: {args.model}")

    # Validate source file (if not camera or RTSP)
    if not args.source.startswith("/dev/video") and not args.source.startswith("rtsp://") and not args.source.lower().startswith("discover"):
        if not os.path.exists(args.source):
            print(f"[ERROR] Video source does not exist: {args.source}")
            sys.exit(1)

    if args.source and args.source.lower() == "discover":
        cam_params = select_webcam()
        if not cam_params:
            print("[ERROR] No valid camera parameters found. Exiting.")
            sys.exit(1)

        args.source = cam_params["device"]
        args.cam_width, args.cam_height = cam_params["width"], cam_params["height"]
        args.cam_format = cam_params["pixfmt"]

        # Normalize pixel format for GStreamer compatibility
        if args.cam_format.upper() in ("YUYV", "YUY2"):
            args.cam_format = "YUY2"
        elif args.cam_format.upper() in ("GREY", "GRAY8"):
            args.cam_format = "GRAY8"
        elif args.cam_format.upper() in ("MJPG"):
            args.cam_format = "MJPG"
        else:
            args.cam_format = "BGRx"  # safe fallback for RGB-like formats

        print(f"[INFO] Selected Camera: {args.source}")
        print(f"[INFO] Resolution: {args.cam_width}x{args.cam_height}, Format: {args.cam_format}")

    main(args)


