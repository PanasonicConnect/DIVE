import cv2
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_frames_from_video(video_path, output_root):
    """Extract frames from a single video and save them"""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_root, video_name)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open: {video_path}")
        return

    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
        filename = os.path.join(output_dir, f"{frame_count:05d}.jpg")
        cv2.imwrite(filename, frame)

    cap.release()
    print(f"✅ Extracted {frame_count} frames from: {video_name}")

def process_all_videos(input_dir, output_root, max_workers=16):
    """Process all videos in parallel"""
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.mp4')]
    video_paths = [os.path.join(input_dir, f) for f in video_files]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(extract_frames_from_video, vp, output_root) for vp in video_paths]
        for future in as_completed(futures):
            future.result()  # Raise exception if any

if __name__ == '__main__':
    input_video_dir = '/root/CVRR-EVALUATION-SUITE/CVRR-ES-Test/test-videos'
    output_image_dir = '/root/CVRR-EVALUATION-SUITE/CVRR-ES-Test-Frames'
    process_all_videos(input_video_dir, output_image_dir, max_workers=16)
