"""
SpotBox - Object Detection with YOLOv8
Detects objects in images or videos and saves the output.
"""

import cv2
from ultralytics import YOLO
import argparse
import os
from datetime import datetime


def load_model(model_path="yolov8n.pt"):
    """
    Load YOLO model.
    
    Args:
        model_path (str): Path to YOLO weights file
    
    Returns:
        YOLO: Loaded YOLO model
    """
    if not os.path.exists(model_path):
        print(f"⚠️ Model not found at {model_path}")
        print("📥 Downloading YOLOv8n model...")
        model = YOLO(model_path)  # This auto-downloads if missing
    else:
        model = YOLO(model_path)
    return model


def detect_image(model, image_path, output_path=None, conf_threshold=0.5):
    """
    Run detection on a single image.
    
    Args:
        model: YOLO model
        image_path (str): Path to input image
        output_path (str): Path to save output image
        conf_threshold (float): Confidence threshold
    
    Returns:
        str: Path to saved output image
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Run detection
    results = model(img, conf=conf_threshold)
    
    # Annotate image
    annotated_img = results[0].plot()
    
    # Generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output_{base_name}_{timestamp}.jpg"
    
    # Save output
    cv2.imwrite(output_path, annotated_img)
    print(f"✅ Detection saved to: {output_path}")
    
    return output_path


def detect_video(model, video_path, output_path=None, conf_threshold=0.5):
    """
    Run detection on a video.
    
    Args:
        model: YOLO model
        video_path (str): Path to input video (or 0 for webcam)
        output_path (str): Path to save output video
        conf_threshold (float): Confidence threshold
    
    Returns:
        str: Path to saved output video
    """
    # Open video source
    if video_path == "0":
        cap = cv2.VideoCapture(0)
        source_name = "webcam"
    else:
        cap = cv2.VideoCapture(video_path)
        source_name = os.path.splitext(os.path.basename(video_path))[0]
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Generate output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output_{source_name}_{timestamp}.mp4"
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"🎥 Processing video... Press 'q' to stop early")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model(frame, conf=conf_threshold)
        annotated_frame = results[0].plot()
        
        # Write frame
        out.write(annotated_frame)
        frame_count += 1
        
        # Show progress every 30 frames
        if frame_count % 30 == 0:
            print(f"📹 Processed {frame_count} frames...")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"✅ Video saved to: {output_path}")
    print(f"📊 Total frames processed: {frame_count}")
    
    return output_path


def main():
    """
    Main function - handles command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="SpotBox - YOLOv8 Object Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect.py --source image.jpg
  python detect.py --source video.mp4 --conf 0.7
  python detect.py --source 0 --output webcam_output.mp4
        """
    )
    
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to image or video file, or '0' for webcam"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save output (optional)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLO model weights (default: yolov8n.pt)"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    print("""
    ╔═══════════════════════════════════╗
    ║         SpotBox Detection         ║
    ║    YOLOv8 Object Detection Tool   ║
    ╚═══════════════════════════════════╝
    """)
    
    print(f"📁 Source: {args.source}")
    print(f"📦 Model: {args.model}")
    print(f"🎯 Confidence threshold: {args.conf}")
    
    # Load model
    print("⏳ Loading model...")
    model = load_model(args.model)
    print("✅ Model loaded!")
    
    # Determine if source is image or video
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    source_lower = args.source.lower()
    
    if source_lower == "0":
        detect_video(model, args.source, args.output, args.conf)
    
    elif any(source_lower.endswith(ext) for ext in image_extensions):
        detect_image(model, args.source, args.output, args.conf)
    
    elif any(source_lower.endswith(ext) for ext in video_extensions):
        detect_video(model, args.source, args.output, args.conf)
    
    else:
        print(f"❌ Unsupported file type: {args.source}")
        print("Supported image types: .jpg, .jpeg, .png, .bmp, .tiff")
        print("Supported video types: .mp4, .avi, .mov, .mkv, .webm")
        return
    
    print("\n✨ Detection complete! ✨")


if __name__ == "__main__":
    main()
