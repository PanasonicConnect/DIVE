import os
import glob
import json
import base64
import numpy as np
import cv2
from mimetypes import guess_type
from typing import List

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

llm = ChatOpenAI(model="gpt-4.1", temperature=0.0, disable_streaming=True)

def load_annotations(folder_path, folder):
    """
    Load annotations from a JSON file.

    Args:
        folder_path (str): Path to the folder containing the annotation file.
        folder (str): Folder name to construct the annotation file path.

    Returns:
        list or None: Parsed JSON data if the file exists, otherwise None.
    """
    annotation_path = os.path.join(folder_path, f"annotations_{folder}.json")
    if not os.path.exists(annotation_path):
        return None
    with open(annotation_path, "r", encoding="utf-8") as f:
        return json.load(f)


def local_image_to_data_url(image_path):
    """
    Encode a local image into a data URL.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Data URL of the image.
    """
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    mime_type = mime_type or 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def encode_images(image_path, frame_num, detail="auto", indices=None):
    """
    Encode multiple images from a folder into data URLs.

    Args:
        image_path (str): Path to the folder containing image frames.
        frame_num (int): Number of frames to sample.
        detail (str): Detail level for the encoded images (e.g., "low", "high").

    Returns:
        list: List of dictionaries containing encoded image data.
    """
    # Get all valid image files in the folder
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    frame_path_list = sorted(
        path for path in glob.glob(os.path.join(image_path, "*"))
        if os.path.splitext(path)[1].lower() in valid_extensions
    )

    # Sample frames uniformly
    if not frame_path_list:
        print(f"No valid image files found in {image_path}")
        return []
    
    if indices is None:
       indices = np.linspace(0, len(frame_path_list) - 1, frame_num, dtype=int)
    else:
        indices = [index for index in indices if index < len(frame_path_list)]

    frames = [
        {
            "type": "image_url",
            "image_url": {
                "url": local_image_to_data_url(frame_path_list[i]),
                "detail": detail
            }
        }
        for i in indices
    ]

    print(f"Encoded {len(frames)} frames from {image_path}")
    return frames


def get_video_metadata(cvrr_dataset_path, category, video_id):
    """
    Get metadata of a video file.

    Args:
        cvrr_dataset_path (str): Path to the CVRR dataset.
        category (str): Category of the video.
        video_id (str): ID of the video.

    Returns:
        dict: Metadata of the video including width, height, total frames, duration, and frame rate.
    """
    video_path = os.path.join(cvrr_dataset_path, category, video_id)
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = round(total_frames / fps, 2) if fps else 0
    frame_rate = round(fps, 2) if fps else 0
    video.release()

    return {
        "video_id": video_id,
        "category": category,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "duration": duration,
        "frame_rate": frame_rate,
    }


def select_frames(video_id, video_metadata, current_question):
    """
    Selects relevant frame indices for analysis based on the video metadata, summary, and question.

    Args:
        video_id (str): ID of the video file.
        video_metadata (dict): Metadata dictionary for the video.
        current_question (str): The question to analyze.

    Returns:
        tuple: (selected frame indices, explanation string)
    """
    print("=== Selecting frames for analysis ===")
    base_video_id = os.path.splitext(video_id)[0]
    summary_dir = os.environ.get("VIDEO_SUMMARY_PATH", "")
    video_summary_path = os.path.join(summary_dir, f"{base_video_id}.txt")
    print(f"Looking for video summary at: {video_summary_path}")

    video_summary = ""
    if os.path.exists(video_summary_path):
        with open(video_summary_path, "r", encoding="utf-8") as f:
            video_summary = f.read()

    system_message = (
        "You are an AI assistant that analyzes video frames and selects specific frames for deeper analysis based on the provided question and video summary.\n"
        "Your task is to:\n"
        "1. Determine whether the question requires analyzing the entire video or focusing on specific parts.\n"
        "2. Select a range of frames that are most relevant to answering the question and explain why this range is chosen.\n"
        "3. Provide an explanation of the selected frame indices and their relevance to the question.\n"
        "The number of selected frames MUST be between 8 and 16.\n"
        "Never include a frame index over the total number of frames.\n"
    )

    print(f"video_metadata: {video_metadata} ({type(video_metadata)})")

    user_message = (
        f"Below is the information about the video:\n"
        f"- Video Width: {video_metadata['width']}\n"
        f"- Video Height: {video_metadata['height']}\n"
        f"- Total Frames: {video_metadata['total_frames']}\n\n"
        f"Video summary:\n{video_summary}\n\n"
        f"Question:\n{current_question}\n"
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message),
        HumanMessage(content=user_message)
    ])

    class SelectedFrames(BaseModel):
        explanation: str = Field(
            ...,
            description="An explanation of the selected frames and their relevance to the question."
        )
        frame_indices: List[int] = Field(
            ...,
            description="A list of indices representing the selected frames for analysis. The list must contain between 8 and 16 indices."
        )

    structured_llm = llm.with_structured_output(SelectedFrames)
    chain = prompt | structured_llm
    output: SelectedFrames = chain.invoke({})

    print(f"Selected frames and explanation: {output.explanation}")
    print(f"Selected frames: {output.frame_indices}")

    # Ensure all indices are within valid range
    max_index = video_metadata["total_frames"] - 1
    selected_indexes = [idx for idx in output.frame_indices if 0 <= idx <= max_index]
    if len(selected_indexes) < len(output.frame_indices) and max_index not in selected_indexes:
        selected_indexes.append(max_index)

    return selected_indexes, output.explanation


def is_data_evrything_ok(
    cvrr_dataset_path: str,
    image_base_path: str,
    video_summary_path: str
) -> bool:
    """
    Check that for each split folder under cvrr_dataset_path:
      1. Every VideoID from load_annotations has an image directory
         at image_base_path/<split>/<video_id_without_ext> containing at least one file.
      2. Every such video has a summary TXT file at
         video_summary_path/<video_id_without_ext>.txt

    Returns True if and only if all checks pass.
    """
    all_ok = True

    # 1) Verify the top-level directories exist
    for path in (cvrr_dataset_path, image_base_path, video_summary_path):
        if not os.path.isdir(path):
            print(f"[ERROR] Directory not found: {path}")
            return False

    # 2) For each split (subfolder) in the dataset
    for split in os.listdir(cvrr_dataset_path):
        split_folder = os.path.join(cvrr_dataset_path, split)
        if not os.path.isdir(split_folder):
            continue

        try:
            qa_pairs = load_annotations(split_folder, split)
        except Exception as e:
            print(f"[ERROR] Could not load annotations for split '{split}': {e}")
            return False

        for qa in qa_pairs:
            vid = qa.get("VideoID", "")
            base = os.path.splitext(vid)[0]

            # (a) image directory check
            img_dir = os.path.join(image_base_path, split, base)
            if not os.path.isdir(img_dir):
                print(f"[ERROR] Missing image directory for video '{vid}': {img_dir}")
                all_ok = False
            else:
                if not os.listdir(img_dir):
                    print(f"[ERROR] No images found in {img_dir}")
                    all_ok = False

            # (b) summary TXT check
            summary_file = os.path.join(video_summary_path, f"{base}.txt")
            if not os.path.isfile(summary_file):
                print(f"[ERROR] Missing summary TXT for video '{vid}': {summary_file}")
                all_ok = False
            else:
                # optional: ensure it's readable
                try:
                    with open(summary_file, "r", encoding="utf-8") as f:
                        _ = f.read(1)
                except Exception as e:
                    print(f"[ERROR] Cannot read summary TXT '{summary_file}': {e}")
                    all_ok = False

    return all_ok


def load_video_summary_message(video_path: str) -> str:
    """
    Given the path to a video file, look up its .txt summary in the
    VIDEO_SUMMARY_PATH environment directory and return a user-friendly message.
    """
    # extract base ID (filename without extension)
    base_video_id = os.path.splitext(os.path.basename(video_path))[0]

    # get the summary directory from env
    summary_dir = os.environ.get("VIDEO_SUMMARY_PATH", "")
    summary_file = os.path.join(summary_dir, f"{base_video_id}.txt")

    # read summary if it exists
    if os.path.isfile(summary_file):
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                video_summary = f.read().strip()
        except Exception as e:
            # on read error, fall back to empty summary
            video_summary = ""
            # optionally log the error e here
    else:
        video_summary = ""

    return f"Here is the summary of the video: {video_summary}"