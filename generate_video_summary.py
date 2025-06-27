import os
import argparse
import csv
import json
import textwrap
from typing import List, Dict, Optional
import numpy as np
import cv2
import torch
from tqdm import tqdm
from pydantic import BaseModel, Field
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from utils import encode_images, get_video_metadata


class Config:
    """Configuration constants for the video processing pipeline."""
    DINO_MODEL_NAME = "IDEA-Research/grounding-dino-base"
    GPT_MODEL_NAME = "gpt-4.1"
    GPT_TEMPERATURE = 0.0
    DEFAULT_BOX_THRESHOLD = 0.3
    DEFAULT_TEXT_THRESHOLD = 0.25
    MAX_CSV_LINES = 4000
    NUM_FRAMES = 32
    CSV_HEADER = ["frame_idx", "label", "box_x1", "box_y1", "box_x2", "box_y2", "score"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llm = ChatOpenAI(model=Config.GPT_MODEL_NAME, temperature=Config.GPT_TEMPERATURE, disable_streaming=True)

try:
    dino_processor = AutoProcessor.from_pretrained(Config.DINO_MODEL_NAME)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(Config.DINO_MODEL_NAME).to(device)
    print(f"Grounding DINO model loaded successfully on {device}")
except Exception as e:
    print(f"Error loading DINO model: {e}")
    dino_processor = None
    dino_model = None

def process_video(
    video_path: str, 
    target_classes: List[str], 
    box_threshold: float = Config.DEFAULT_BOX_THRESHOLD, 
    text_threshold: float = Config.DEFAULT_TEXT_THRESHOLD, 
    csv_path: str = "detection_results.csv"
) -> str:
    """
    Perform object detection on video frames using DINO model.
    
    Args:
        video_path: Path to the input video file
        target_classes: List of object classes to detect
        box_threshold: Confidence threshold for bounding box detection
        text_threshold: Confidence threshold for text matching
        csv_path: Path to save CSV results
        
    Returns:
        str: CSV formatted string containing detection results
        
    Raises:
        ValueError: If video cannot be opened or models are not initialized
    """
    if dino_processor is None or dino_model is None:
        raise ValueError("DINO model not properly initialized")
    if not target_classes:
        raise ValueError("Target classes list cannot be empty")
    
    text_query = " . ".join(target_classes)
    print(f"Detection query: {text_query}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        detection_results = []

        with tqdm(total=frame_count, desc="Processing frames") as pbar:
            frame_idx = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                detections = _process_frame(frame, text_query, target_classes, box_threshold, text_threshold)
                if detections:
                    detection_results.append({
                        "frame_idx": frame_idx,
                        "detections": detections
                    })

                frame_idx += 1
                pbar.update(1)
                
    finally:
        cap.release()

    _save_detection_results(detection_results, csv_path)
    return _format_results_as_csv_string(detection_results)


def _process_frame(
    frame: np.ndarray, 
    text_query: str, 
    target_classes: List[str], 
    box_threshold: float, 
    text_threshold: float
) -> List[Dict]:
    """Process a single frame for object detection."""
    height, width = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    target_sizes = [(width, height)]
    
    try:
        inputs = dino_processor(
            text=text_query, 
            images=frame_rgb, 
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = dino_model(**inputs)
            outputs.logits = outputs.logits.cpu()
            outputs.pred_boxes = outputs.pred_boxes.cpu()
            
            results = dino_processor.post_process_grounded_object_detection(
                outputs=outputs, 
                input_ids=inputs.input_ids,
                box_threshold=box_threshold,
                target_sizes=target_sizes
            )
            
            detections = []
            for score, label, box in zip(results[0]["scores"], results[0]["labels"], results[0]["boxes"]):
                if score > text_threshold and str(label) in target_classes:
                    box_coords = [round(float(coord)) for coord in box.tolist()]
                    detections.append({
                        "label": str(label), 
                        "box": box_coords, 
                        "score": round(float(score), 2)
                    })
            
            return detections
            
    except Exception as e:
        print(f"Error processing frame: {e}")
        return []


def _save_detection_results(detection_results: List[Dict], csv_path: str) -> None:
    """Save detection results to JSON and CSV files."""
    json_path = csv_path.replace('.csv', '.json')
    try:
        with open(json_path, mode="w", encoding="utf-8") as jsonfile:
            json.dump(detection_results, jsonfile, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: Could not save JSON file {json_path}: {e}")
        
    all_rows = []
    for result in detection_results:
        frame_idx = result['frame_idx']
        for detection in result["detections"]:
            row = [
                frame_idx,
                detection['label'],
                detection['box'][0],
                detection['box'][1],
                detection['box'][2],
                detection['box'][3],
                detection['score']
            ]
            all_rows.append(row)

    try:
        with open(csv_path, mode="w", encoding="utf-8", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(Config.CSV_HEADER)
            writer.writerows(all_rows)
                    
    except Exception as e:
        print(f"Error saving CSV file {csv_path}: {e}")


def _format_results_as_csv_string(detection_results: List[Dict]) -> str:
    """Format detection results as CSV string with 4000 line limit."""
    if not detection_results:
        return ""
        
    lines = [",".join(Config.CSV_HEADER)]
    
    data_lines = []
    for result in detection_results:
        frame_idx = result['frame_idx']
        for detection in result["detections"]:
            row = [
                str(frame_idx),
                detection['label'],
                str(detection['box'][0]),
                str(detection['box'][1]),
                str(detection['box'][2]),
                str(detection['box'][3]),
                str(detection['score'])
            ]
            data_lines.append(",".join(row))
    
    if len(data_lines) > Config.MAX_CSV_LINES:
        print(f"Applying {Config.MAX_CSV_LINES} line limit: {len(data_lines)} -> ", end="")
        filtered_data_lines = filter_csv_data(data_lines, Config.MAX_CSV_LINES)
        print(f"{len(filtered_data_lines)} lines")
        lines.extend(filtered_data_lines)
    else:
        lines.extend(data_lines)
    
    return "\n".join(lines) + "\n" if lines else ""


def object_detection(base_path: str, image_path: str, video_metadata: Dict[str, str], output_dir: str = "data") -> Dict[str, str]:
    """
    Perform object detection on all frames and return the results.
    
    Args:
        base_path: The base path for saving results
        image_path: Path to the video frames
        video_metadata: Metadata about the video including category and video ID
        output_dir: The directory to save the output results

    Returns:
        dict: A dictionary containing detected objects for each frame
    """
    print("=== Performing object detection ===")
    video_id = os.path.basename(image_path)
    csv_path = os.path.join(output_dir, f"{video_id}.csv")

    if os.path.exists(csv_path):
        detection_results_str = _process_existing_csv(csv_path)
        if detection_results_str:
            newline = '\n'
            print(f"Loaded existing CSV with {len(detection_results_str.split(newline))} lines")
            return {"detected_objects": detection_results_str}
    
    print("Generating new object detections...")
    target_objects = _generate_target_objects(image_path, video_id, output_dir)
    
    video_path = os.path.join(base_path, video_metadata["category"], video_metadata["video_id"])
    detection_results = process_video(
        video_path, 
        target_objects, 
        Config.DEFAULT_BOX_THRESHOLD, 
        Config.DEFAULT_TEXT_THRESHOLD, 
        csv_path=csv_path
    )
    
    return {"detected_objects": detection_results}


def _process_existing_csv(csv_path: str) -> Optional[str]:
    """
    Process existing CSV file and return formatted detection results.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Optional[str]: Formatted CSV string or None if processing fails
    """
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            all_rows = list(reader)
            
        if not all_rows:
            print(f"Warning: Empty CSV file at {csv_path}")
            return None
            
        header = all_rows[0]
        data_rows = all_rows[1:]

        if len(header) < len(Config.CSV_HEADER):
            print(f"Updating CSV header from {len(header)} to {len(Config.CSV_HEADER)} columns")
            header = Config.CSV_HEADER

        filtered_rows = filter_csv_data(data_rows, Config.MAX_CSV_LINES)
        
        output_lines = [",".join(header)]
        for row in filtered_rows:
            output_lines.append(",".join(map(str, row)))
        
        result = "\n".join(output_lines)
        if result:
            result += "\n"
            
        return result
        
    except Exception as e:
        print(f"Error processing CSV file {csv_path}: {e}")
        return None


def filter_csv_data(data, max_lines):
    """
    Filter CSV data using frame-based sampling with divisors.
    
    Args:
        data: List of data (can be rows as lists or strings)
        max_lines: Maximum number of lines to keep (including header if present)
    
    Returns:
        Filtered list of data with the same type as input
    """
    if len(data) <= max_lines:
        return data

    has_header = False
    data_rows = data
    header_row = None

    if data:
        first_item = data[0]
        if isinstance(first_item, list) and first_item:
            try:
                int(first_item[0])
            except (ValueError, IndexError):
                has_header = True
                header_row = first_item
                data_rows = data[1:]
        elif isinstance(first_item, str):
            parts = first_item.split(',')
            if parts:
                try:
                    int(parts[0])
                except ValueError:
                    has_header = True
                    header_row = first_item
                    data_rows = data[1:]
    
    if len(data_rows) <= max_lines:
        return data
    
    final_filtered_rows = []
    
    for divisor in range(2, len(data_rows) + 3):
        current_filtered_rows = []
        
        for data_item in data_rows:
            if not data_item:
                continue
                
            try:
                frame_index_str = None
                if isinstance(data_item, list):
                    frame_index_str = data_item[0] if data_item else None
                elif isinstance(data_item, str):
                    parts = data_item.split(',')
                    frame_index_str = parts[0] if parts else None
                
                if frame_index_str is not None:
                    frame_index = int(frame_index_str)
                    if frame_index % divisor == 0:
                        current_filtered_rows.append(data_item)
            except (ValueError, IndexError):
                pass
        
        # Check if current filtering meets the size requirement
        total_lines = len(current_filtered_rows) + (1 if has_header else 0)
        if total_lines <= max_lines:
            final_filtered_rows = current_filtered_rows
            break

        if divisor == (len(data_rows) + 2):
            final_filtered_rows = current_filtered_rows
    
    # Reconstruct the result with header if present
    result = []
    if has_header and header_row is not None:
        result.append(header_row)
    result.extend(final_filtered_rows)
    
    return result


def _generate_target_objects(image_path: str, video_id: str, output_dir: str) -> List[str]:
    """
    Generate list of target objects for detection using GPT vision analysis.
    
    Args:
        image_path: Path to video frames
        video_id: Video identifier
        output_dir: Output directory for saving object list
        
    Returns:
        List[str]: List of target object classes
    """
    try:
        encoded_frames = encode_images(image_path, frame_num=Config.NUM_FRAMES, detail="auto")
    except Exception as e:
        print(f"Error encoding frames: {e}")
        return []

    system_message = textwrap.dedent("""
        You are an AI assistant that analyzes video frames and extracts objects based on the visual content.
        Use the provided object detection labels to identify distinct objects across all frames.
        Ensure that no object class includes parentheses () in its name.
        Respond in list format, ensuring each object type is listed only once.
    """)

    user_message = textwrap.dedent("""
        These are frames from a video. Use the visual cues and the provided object detection labels to extract all unique objects present in the frames.
        List each object type only once.
    """)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message),
        HumanMessage(content=[{"type": "text", "text": user_message}] + encoded_frames)            
    ])

    class ObjectList(BaseModel):
        detection_results: List[str] = Field(
            ...,
            description=(
                "The objects detected in the video frames. "
                "Each entry contains a list of detected objects with their labels. "
                "Example: ['person', 'car', 'dog']"
            )
        )

    structured_llm = llm.with_structured_output(ObjectList)
    chain = prompt | structured_llm
    
    try:
        output: ObjectList = chain.invoke({"selected_frames": encoded_frames})
        target_objects = output.detection_results
        
        print(f"Target objects: {target_objects}")
        
        object_list_path = os.path.join(output_dir, f"{video_id}_objects.txt")
        with open(object_list_path, 'w', encoding='utf-8') as f:
            for obj in target_objects:
                f.write(f"{obj}\n")
                
        return target_objects
        
    except Exception as e:
        print(f"Error generating target objects: {e}")
        return []


def _create_system_message() -> str:
    """Create the system message for video summarization."""
    return textwrap.dedent("""
        You are an AI assistant that summarizes the content of a video by analyzing object detection results from all frames and visual information from sampled frames.
        Your task is to create a comprehensive, chronological summary by integrating these two sources of information.

        Please structure your output EXACTLY as follows:

        **1. Overall Video Overview:**
            - A brief, one or two-sentence overview of the video's general content, setting, and primary subject matter.

        **2. Key Objects and Their General Behavior:**
            - List the most prominent objects observed throughout the video.
            - For each object, describe its general behavior, common locations, and overall temporal patterns (e.g., "A 'person' is consistently present in the foreground, primarily interacting with a 'table' from frame X to frame Y."). This description should be based on your integrated understanding of both CSV data and visual cues.

        **3. Detailed Chronological Summary of Events and Scenes:**
            - Describe significant scenes, events, and transitions in chronological order.
            - For each event or scene:
                - **Frame Range/Specific Frame (from Object Detection results across all frames):** Clearly state the relevant frame index or range of frame indices from the entire video, identified by analyzing the object detection CSV data (e.g., "Frames 100-150 (object 'X' appears based on CSV):", "Around frame 230 (interaction between 'Y' and 'Z' detected in CSV):", "Frame 450 (object 'A' disappears according to CSV):"). The frame indices mentioned here are not limited to the sampled frames but should cover the entire video based on the CSV analysis.
                - **Description:** Detail what happens in that scene/event, focusing on object interactions, movements, appearances, and disappearances as indicated by the object detection data. Supplement this with visual details from sampled frames where relevant. For example: "Frames 50-75: A 'dog' is detected entering the scene from the left and approaches a 'ball' also detected in this range. Visually, the dog appears to be a golden retriever." or "Frame 180: Object detection data indicates a significant shift in the types and locations of objects, suggesting a scene transition. Visually, the setting changes from an indoor office to an outdoor park." Do NOT explicitly state whether information came from the CSV or sampled frames in your description, but ensure the core events and frame numbers are driven by the object detection analysis of all frames.

        **4. Concluding Remarks (Optional):**
            - Briefly mention any overarching patterns, the video's likely purpose if discernible, or any unresolved/ambiguous aspects, based on your integrated analysis.

        Integrate information from both the OBJECT DETECTION RESULTS (CSV data) and the EVENLY SAMPLED FRAMES for visual details to form a cohesive understanding.
        Provide explicit frame numbers for key events, appearances, disappearances, and scene transitions.
        Your final output should describe the video content based on your combined understanding of these sources, without detailing which specific piece of information came from which source.
    """)


def _create_user_message(video_metadata: Dict[str, str], detected_objects: List[str], indices: np.ndarray) -> str:
    """Create the user message for video summarization."""
    return textwrap.dedent(f"""
        Below is the information about a video:
        - Video Width: {video_metadata["width"]}
        - Video Height: {video_metadata["height"]}
        - Total Frames: {video_metadata["total_frames"]}

        OBJECT DETECTION RESULTS (CSV format string, covering all frames in the video):
        The CSV data is a string where each line represents a detected object in a specific frame.
        The columns are typically: frame_index, object_label, x_min, y_min, x_max, y_max, confidence_score
        Example line: "10,person,100,150,200,350,0.92" (meaning in frame 10, a 'person' was detected with 92% confidence at the given bounding box)
        Actual data provided:
        {detected_objects}

        EVENLY SAMPLED FRAMES for visual analysis (you will receive these as image data):
        Frame indices for the sampled frames: {indices.tolist()}

        Provide a chronological summary that represents the entire video content, following the detailed structure specified in the system message.
    """)


def _build_prompt(encoded_frames: str, video_metadata: Dict[str, str], detected_objects: List[str], indices: np.ndarray) -> ChatPromptTemplate:
    """Build the prompt for video summarization."""
    system_message = _create_system_message()
    user_message = _create_user_message(video_metadata, detected_objects, indices)
    
    return ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message),
        HumanMessage(content=encoded_frames),
        HumanMessage(content=[{"type": "text", "text": user_message}])
    ])


def _save_summary(text_path: str, summary: str) -> None:
    """Save the video summary to a file."""
    try:
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"Summary saved to: {text_path}")
    except IOError as e:
        print(f"Error saving summary to {text_path}: {e}")
        raise


def summarize_video(image_path: str, video_metadata: Dict[str, str], detected_objects: List[str], output_dir: str) -> str:
    """
    Summarize the video content using object detection results and sampled frames.

    Args:
        image_path: Path to the directory containing extracted video frames
        video_metadata: Dictionary containing video metadata (video_id, width, height, total_frames)
        detected_objects: List of detected objects from object detection analysis
        output_dir: Directory to save the summary output

    Returns:
        The generated video summary as a string

    Raises:
        ValueError: If video metadata is invalid
        FileNotFoundError: If image path doesn't exist
        Exception: For other processing errors
    """
    print("=== Summarizing video ===")
    
    # Validate inputs
    if not video_metadata or "video_id" not in video_metadata:
        raise ValueError("Invalid video metadata: missing video_id")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")
    
    video_id = os.path.splitext(video_metadata["video_id"])[0]
    text_path = os.path.join(output_dir, f"{video_id}.txt")
    
    # Check if summary already exists
    if os.path.exists(text_path):
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                video_summary = f.read()
            print(f"Found existing summary: {text_path}")
            return video_summary
        except IOError as e:
            print(f"Error reading existing summary: {e}")
    
    try:
        print("Encoding frames for analysis...")
        encoded_frames = encode_images(image_path, frame_num=Config.NUM_FRAMES, detail="auto")
        
        total_frames = int(video_metadata.get("total_frames", 0))
        if total_frames <= 0:
            raise ValueError("Invalid total_frames in video metadata")

        indices = np.linspace(0, total_frames - 1, num=Config.NUM_FRAMES, dtype=int)

        prompt = _build_prompt(encoded_frames, video_metadata, detected_objects, indices)

        class VideoSummary(BaseModel):
            summary: str = Field(
                ...,
                description="A summary of the video content, including key points, scene transitions, and the overall flow of events with explicit frame numbers."
            )

        print("Generating video summary...")
        structured_llm = llm.with_structured_output(VideoSummary)
        chain = prompt | structured_llm
        
        output: VideoSummary = chain.invoke({
            "video_meta": video_metadata,
            "detected_objects": detected_objects,
            "selected_frames": encoded_frames
        })

        _save_summary(text_path, output.summary)
        
        print("Video summary generation completed successfully")
        return output.summary
        
    except Exception as e:
        print(f"Error during video summarization: {e}")
        raise

def generate_summary_and_objects(
    cvrr_dataset_path: str,
    image_base_path: str,
    folder_name: str,
    video_id: str,
    output_dir: str = "data"
):
    """
    Generates object detection results and video summary for a given video.
    """
    image_path = os.path.join(image_base_path, folder_name, video_id.split(".")[0])
    
    print(f"Processing video: {video_id} in folder: {folder_name}")
    print(f"Image path: {image_path}")

    if not os.path.exists(image_path):
        print(f"Error: Image path does not exist: {image_path}")
        return None, None

    video_metadata = get_video_metadata(cvrr_dataset_path, folder_name, video_id)
    if not video_metadata:
        print(f"Error: Could not retrieve metadata for video: {video_id}")
        return None, None

    object_detection_results = object_detection(cvrr_dataset_path, image_path, video_metadata, output_dir=output_dir)
    video_summary = summarize_video(image_path, video_metadata, object_detection_results, output_dir=output_dir)

    return object_detection_results, video_summary

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate object detection and summary for a video.")
    parser.add_argument("--cvrr_dataset_path", required=True, help="Path to the CVRR-ES dataset.")
    parser.add_argument("--image_base_path", required=True, help="Base path for the images.")
    parser.add_argument("--output_dir", default="data", help="Directory to save output results.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    try:
        folder_names = [name for name in os.listdir(args.cvrr_dataset_path) 
                       if os.path.isdir(os.path.join(args.cvrr_dataset_path, name))]
        print(f"Found {len(folder_names)} directories in {args.cvrr_dataset_path}: {folder_names}")
    except Exception as e:
        print(f"Error reading directories from {args.cvrr_dataset_path}: {e}")
        folder_names = []

    for folder_name in folder_names:
        print(f"\n=== Processing folder: {folder_name} ===")
        
        annotation_path = os.path.join(args.cvrr_dataset_path, folder_name, f"annotations_{folder_name}.json")
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
        except FileNotFoundError:
            print(f"Error: Annotation file not found at {annotation_path}")
            annotations = []
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {annotation_path}")
            annotations = []

        # Collect unique video IDs using a set for efficiency
        video_id_list = sorted(list(set(ann['VideoID'] for ann in annotations if 'VideoID' in ann)))

        print(f"Found {len(video_id_list)} videos to process in folder {folder_name}")
        print(video_id_list)

        for video_id in video_id_list:
            print(f"Processing video ID: {video_id}")

            obj_results, summary = generate_summary_and_objects(
                args.cvrr_dataset_path,
                args.image_base_path,
                folder_name,
                video_id,
                output_dir=args.output_dir
            )

            if obj_results and summary:
                print("\n--- Final Results ---")
                print("Object Detection:", obj_results)
                print("Video Summary:", summary)
            else:
                print("Failed to generate summary and object detection results.")

