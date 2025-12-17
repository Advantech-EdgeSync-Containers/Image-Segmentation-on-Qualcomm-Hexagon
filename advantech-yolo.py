import argparse
import os


# Limit number of threads used by NumPy/OpenBLAS
subprocess_thread = 4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # Or 1 for deterministic performance
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
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

shutdown_requested = False
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
# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(COCO_CLASS_NAMES), 3))


#File save thread
frame_queue = queue.Queue(maxsize=20)

def writer_thread(out_writer, stop_event):
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.5)
            out_writer.write(frame)
        except queue.Empty:
            continue
    out_writer.release()
            
def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3, mask_maps=None):
    img_height, img_width = image.shape[:2]
    size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    mask_img = draw_masks(image, boxes, class_ids, mask_alpha, mask_maps)

    # Draw bounding boxes and labels of detections
    for box, score, class_id in zip(boxes, scores, class_ids):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw rectangle
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, 2)

        label = COCO_CLASS_NAMES[class_id]
        caption = f'{label} {int(score * 100)}%'
        (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=size, thickness=text_thickness)
        th = int(th * 1.2)

        cv2.rectangle(mask_img, (x1, y1),
                      (x1 + tw, y1 - th), color, -1)

        cv2.putText(mask_img, caption, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

    return mask_img

def draw_masks(image, boxes, class_ids, mask_alpha=0.8, mask_maps=None):
    
    if mask_maps is None or len(mask_maps) == 0:
        return image

    mask_img = image.copy()
    
    # Precompute colors for all masks
    mask_colors = colors[class_ids].astype(np.uint8)

    # Composite mask image (blank)
    composite_mask = np.zeros_like(mask_img, dtype=np.uint8)  # Black mask


    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        crop_mask = mask_maps[i, y1:y2, x1:x2]
        if crop_mask.shape[0] == 0 or crop_mask.shape[1] == 0:
            continue

        # Resize mask to match box size
        resized_mask = cv2.resize(crop_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)

        # Threshold mask
        binary_mask = (resized_mask > 0.5).astype(np.uint8)

        # Create colored mask
        colored_mask = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8) #black mask
        for c in range(3):
            colored_mask[:, :, c] = binary_mask * mask_colors[i][c]

        # Overlay into composite mask
        composite_mask[y1:y2, x1:x2] = np.maximum(composite_mask[y1:y2, x1:x2], colored_mask) #black mask

    # Blend final composite mask once
    return cv2.addWeighted(composite_mask, mask_alpha, image, 1, 0)

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
        
class YOLOSeg:

    def __init__(self, path, conf_thres=0.7, iou_thres=0.5, num_masks=32):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.num_masks = num_masks

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.segment_objects(image)

    def initialize_model(self, path):
        # Set backend_type value to 'htp'
        TFLiteDelegate_lib_path = '/workspace/libs/libQnnTFLiteDelegate.so'
        delegate = load_delegate(
           TFLiteDelegate_lib_path,
           options={
                   'backend_type': 'htp',
                   'htp_performance_mode': 3,
                   }
          )        
        self.session = Interpreter(model_path=path, experimental_delegates=[delegate])
        self.session.allocate_tensors()

        # Get model info
        self.get_input_details()
        self.get_output_details()

    def segment_objects(self, image):
        img_data = self.prepare_input(image)
        # Quantization handling
        if self.model_inputs[0]['dtype'] == np.uint8:
            img_data = (img_data * 255).astype(np.uint8)
        elif self.model_inputs[0]['dtype'] == np.int8:
            scale, zero_point = self.model_inputs[0]["quantization"]
            img_data = (img_data / scale + zero_point).astype(np.int8)
        else:
            img_data = img_data.astype(np.float32)  # for FP32 model
        
        self.session.set_tensor(self.model_inputs[0]["index"], img_data)
        start = time.perf_counter()
        self.session.invoke()
        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")    
        out0 = self.session.get_tensor(self.model_outputs[0]["index"]) # (1, 160, 160, 32)
        out1 = self.session.get_tensor(self.model_outputs[1]["index"]) # (1, 116, 8400)
       
        #print("out1 shape: ",out1.shape) #out1 shape:  (1, 116, 8400)
        
        self.boxes, self.scores, self.class_ids, mask_pred = self.process_box_output(out1)#0 originally
        self.mask_maps = self.process_mask_output(mask_pred, out0) #1 originally
        return self.boxes, self.scores, self.class_ids, self.mask_maps

    def prepare_input(self, image):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # Get the height and width of the input image
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def process_box_output(self, box_output):

        predictions = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - self.num_masks - 4

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:4+num_classes], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], [], np.array([])

        # Scale to image dimensions
        if self.input_width == self.input_height :
            box_predictions = predictions[..., :num_classes+4]*self.input_width
        else:
            box_predictions = predictions[..., :num_classes+4]
            box_predictions[:, [0, 2]] *= self.input_width
            box_predictions[:, [1, 3]] *= self.input_height
            
        mask_predictions = predictions[..., num_classes+4:]

        # Get the class with the highest confidence
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(box_predictions)


        # Prepare bounding boxes for OpenCV NMS: convert to [x, y, w, h]
        cv2_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2_boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(
            bboxes=cv2_boxes,
            scores=scores.tolist(),
            score_threshold=self.conf_threshold,
            nms_threshold=self.iou_threshold
        )

        if len(indices) == 0:
            return [], [], [], np.array([])

        # cv2.dnn.NMSBoxes returns [[i], [j], ...], flatten it
        indices = np.array(indices).flatten()

        return boxes[indices], scores[indices], class_ids[indices], mask_predictions[indices]

    def process_mask_output(self, mask_predictions, mask_output):
        if mask_predictions.shape[0] == 0:
            return []

        mask_output = np.squeeze(mask_output)  # (H, W, N)
        mask_output = np.transpose(mask_output, (2, 0, 1))  # (N, H, W)

        mask_height, mask_width = mask_output.shape[1:]

        # Flatten mask_output for matrix multiplication
        mask_output_flat = mask_output.reshape(self.num_masks, -1)

        # Fast sigmoid: mask = 1 / (1 + exp(-x))
        masks = mask_predictions @ mask_output_flat
        np.exp(-masks, out=masks)
        masks += 1
        np.reciprocal(masks, out=masks)
        masks = masks.reshape((-1, mask_height, mask_width))

        # Rescale boxes to mask resolution
        scale_boxes = self.rescale_boxes(self.boxes,
                                         (self.img_height, self.img_width),
                                         (mask_height, mask_width))

        mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width), dtype=np.uint8)

        for i in range(len(scale_boxes)):
            scale_x1 = int(np.floor(scale_boxes[i][0]))
            scale_y1 = int(np.floor(scale_boxes[i][1]))
            scale_x2 = int(np.ceil(scale_boxes[i][2]))
            scale_y2 = int(np.ceil(scale_boxes[i][3]))

            x1 = int(np.floor(self.boxes[i][0]))
            y1 = int(np.floor(self.boxes[i][1]))
            x2 = int(np.ceil(self.boxes[i][2]))
            y2 = int(np.ceil(self.boxes[i][3]))

            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            if scale_crop_mask.shape[0] == 0 or scale_crop_mask.shape[1] == 0:
                continue

            crop_mask = cv2.resize(scale_crop_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
            mask_maps[i, y1:y2, x1:x2] = (crop_mask > 0.5).astype(np.uint8)

        return mask_maps

    def extract_boxes(self, box_predictions):
        # Extract boxes from predictions
        boxes = box_predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes,
                                   (self.input_height, self.input_width),
                                   (self.img_height, self.img_width))

        # Convert boxes to xyxy format
        boxes[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
        boxes[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
        boxes[..., 2] = boxes[..., 0] + boxes[..., 2] 
        boxes[..., 3] = boxes[..., 1] + boxes[..., 3] 

        # Check the boxes are within the image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)

        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)

    def draw_masks(self, image, draw_scores=True, mask_alpha=0.5):
        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha, mask_maps=self.mask_maps)

    def get_input_details(self):
        self.model_inputs = self.session.get_input_details()

        self.input_shape = self.model_inputs[0]["shape"]
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]

    def get_output_details(self):
        self.model_outputs = self.session.get_output_details()

    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        """
        Rescale bounding boxes from input resolution to original image resolution.
        boxes: numpy array of shape (N, 4), format [x1, y1, x2, y2]
        input_shape: (height, width) of model input
        image_shape: (height, width) of the original image
        """
        input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])
        return boxes
       
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


    prev_frame_time = 0
    new_frame_time = 0
    fps = 0
    frame = None
    # Video source - GStreamer pipeline or camera
    main_frame_width, main_frame_height, source_fps = get_video_properties(args)
    num_cores = os.cpu_count()
    print(f"Logical CPU cores: {num_cores}")
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

    # Initialize YOLOv8 Instance Segmentator
    yoloseg = YOLOSeg(args.model, args.conf_thres, args.iou_thres)
    
    # Optional video writer
    out_writer = None
    stop_event = threading.Event()
    if args.save:
        out_writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*'mp4v'),
                                     source_fps, (main_frame_width, main_frame_height))
        wt = threading.Thread(target=writer_thread, args=(out_writer, stop_event), daemon=True)
        wt.start()

    # Setup GStreamer pipeline for display
    # Decide sync behavior based on your conditions
    use_sync = not (args.source.startswith("/dev/video") or args.source.startswith("rtsp://") or source_fps > 20)
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
        cv2.setUseOptimized(True)
        cv2.setNumThreads(subprocess_thread)  # Or adjust depending on CPU cores

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

            yoloseg(frame)
            output_image = yoloseg.draw_masks(frame)
            
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            
            # Draw FPS on the top-left corner
            cv2.putText(output_image, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Push frame to GStreamer display
            push_frame_to_gst(output_image)
            if out_writer:
                frame_queue.put(output_image)


            # Check for quit (non-blocking)
            #if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            #    c = sys.stdin.read(1)
            #    if c == 'q':
            #        break

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        if out_writer:
            stop_event.set()
            wt.join()
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
    parser = argparse.ArgumentParser(description="YOLOv8 TFLite Image Segmentation with GStreamer Display")
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
    parser.add_argument("--conf-thres", type=validate_probability, default=0.3,
                        help="Object confidence threshold (0.0–1.0). Default: 0.3")
    parser.add_argument("--iou-thres", type=validate_probability, default=0.5,
                        help="IoU threshold for Non-Maximum Suppression (0.0–1.0). Default: 0.5")
    parser.add_argument("--cam-width", type=int, default=1920,
                        help="Camera capture width when using /dev/video* as source. Default: 1920.")
    parser.add_argument("--cam-height", type=int, default=1080,
                        help="Camera capture height when using /dev/video* as source. Default: 1080.")
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


