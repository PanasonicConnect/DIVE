from langgraph.graph import StateGraph, END

from filelock import FileLock

import os
import json
import argparse
from typing import TypedDict, List, Dict, Optional

import multiprocessing
import sys
import traceback

import concurrent.futures

from utils import load_annotations, get_video_metadata, is_data_evrything_ok
from nodes import infer_question_intent, split_question, answer_question, refine_sub_questions, should_continue, finalize_answer

class AgentState(TypedDict):
    video_id: str
    original_question: str                      # original question
    question_intent: str
    image_path: str
    video_path: str
    video_metadata: Optional[Dict[str, int | float | str]]  # video metadata (may include int, float, str)
    sub_questions: Dict[int, List[str]]         # sub questions
    qa_results: List[Dict[str, str]]           # sub questions and their answers
    tool_results: List[Dict[str, str]]
    iter: int                                   # loop iteration counter
    max_iter: int                               # maximum loop iterations
    continue_flag: bool                            # whether to continue refining
    continue_reasons: List[str]                # reasons for continuing
    final_answer: Optional[str]               # final answer to the question 

def process_question(question: str, image_path: str, video_path: str, video_metadata: dict) -> str:
    """Process a question using the agent."""
    workflow = StateGraph(AgentState)

    workflow.add_node("infer_intent", infer_question_intent)
    workflow.add_node("split_question", split_question)
    workflow.add_node("answer_sub_question", answer_question)
    workflow.add_node("answer_final_question", finalize_answer)
    workflow.add_node("refine_sub_questions", refine_sub_questions)
    workflow.add_node("should_continue", should_continue)

    workflow.set_entry_point("infer_intent")
    workflow.add_edge("infer_intent", "split_question")
    workflow.add_edge("split_question", "answer_sub_question")
    workflow.add_edge("answer_sub_question", "refine_sub_questions")
    workflow.add_edge("refine_sub_questions", "should_continue")

    workflow.add_conditional_edges(
        "should_continue",
        lambda state: "continue_loop" if state["continue_flag"] else "break_loop",
        {
            "continue_loop": "answer_sub_question",
            "break_loop": "answer_final_question",
        }
    )
    workflow.add_edge("answer_final_question", END)
    

    graph = workflow.compile()

    agents_result = graph.invoke(
        {
            "video_id": os.path.basename(video_path),
            "original_question": question,
            "image_path": image_path,
            "video_path": video_path,
            "video_metadata": video_metadata,
            "sub_questions": {},
            "qa_results": [],
            "tool_results": [],
            "current_answer": None,
            "iter": 0,
            "max_iter": 25,
            "continue_flag": True,
            "continue_reasons": []
         },
        {"recursion_limit": 150}
    )

    log_thinking = {
        "sub_questions": agents_result["sub_questions"],
        "qa_results": agents_result["qa_results"],
        "continue_reasons": agents_result["continue_reasons"],
        "tool_results": agents_result.get("tool_results", []),
        "question_intent": agents_result.get("question_intent", "")
    }

    return agents_result["final_answer"], log_thinking


def process_one_qa(args):
    """
    Helper function to process a single QA pair and return the index, result, and intermediate log.
    args: (idx: int, qa: dict, encoded_frames: Any, image_path: str, video_path: str)
    return: (idx: int, result: dict, intermediate: dict)
    """
    idx, qa, image_path, video_path, video_metadata = args
    try:
        final_answer, log_thinking = process_question(
            qa["Q"], image_path, video_path, video_metadata
        )
        result = {
            "Q": qa["Q"],
            "A": final_answer,
            "VideoID": qa["VideoID"],
        }
        intermediate = {
            "Q": qa["Q"],
            "A": final_answer,
            "VideoID": qa["VideoID"],
            "sub_questions":    log_thinking["sub_questions"],
            "qa_results":       log_thinking["qa_results"],
            "continue_reasons": log_thinking["continue_reasons"],
            "tool_results":     log_thinking["tool_results"],
            "question_intent":  log_thinking["question_intent"]
        }
    except Exception as e:
        print(f"[ERROR] idx={idx} VideoID={qa['VideoID']} exception occurred:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        result = {
            "Q": qa["Q"],
            "A": None,
            "VideoID": qa["VideoID"],
            "error": str(e),
        }
        intermediate = result.copy()
    return idx, result, intermediate


def process_sigle_folder(
    folder: str,
    cvrr_dataset_path: str,
    image_base_path: str,
    output_dir: str,
    num_frames: int = 8,
    resume: bool = False,
    processes: int = 16
):
    """
    Process each folder (evaluation dimension).
    - Parallel processing for each QA pair (internal pool)
    - Flush JSON after each entry
    - Supports resume
    """
    folder_path       = os.path.join(cvrr_dataset_path, folder)
    image_folder_path = os.path.join(image_base_path,  folder)
    qa_pairs          = load_annotations(folder_path, folder)
    if not qa_pairs:
        return

    # prepare output dir
    os.makedirs(output_dir,                     exist_ok=True)
    os.makedirs(os.path.join(output_dir, "intermediate"), exist_ok=True)
    output_path       = os.path.join(output_dir,       f"{folder}.json")
    intermediate_path = os.path.join(output_dir, "intermediate", f"{folder}_intermediate.json")

    # Load existing results for resume (if any)
    results_dict = {}
    intermediate_dict = {}
    if resume and os.path.exists(output_path):
        loaded_results = json.load(open(output_path, "r", encoding="utf-8"))
        loaded_intermediate = json.load(open(intermediate_path, "r", encoding="utf-8"))
        for idx_existing, res in enumerate(loaded_results):
            results_dict[idx_existing] = res
        for idx_existing, inter in enumerate(loaded_intermediate):
            intermediate_dict[idx_existing] = inter
        retry_indices = {idx for idx, res in results_dict.items() if res.get("A") is None}
    elif os.path.exists(output_path):
        # resume off and output exists: skip
        return
    else:
        retry_indices = set()

    # prepare tasks data
    tasks = []
    last_vid = None
    frames_cache = None
    for idx, qa in enumerate(qa_pairs):
        if resume and idx in results_dict and qa_pairs and idx not in retry_indices:
            # skip successful items
            continue
        vid = qa["VideoID"]
        # encode frames
        if vid != last_vid:
            image_path = os.path.join(image_folder_path, vid.split(".")[0])
            video_path = os.path.join(folder, vid)
            video_metadata = get_video_metadata(cvrr_dataset_path, folder, vid)
            last_vid = vid
        tasks.append((idx, qa, image_path, video_path, video_metadata))

    # multi process
    with multiprocessing.Pool(processes=processes) as pool:
        for idx, result, intermediate in pool.imap(process_one_qa, tasks):
            # Store by ID
            results_dict[idx] = result
            intermediate_dict[idx] = intermediate
            # Write sorted results up to this point
            sorted_ids = sorted(results_dict)
            sorted_results = [results_dict[i] for i in sorted_ids]
            sorted_intermediate = [intermediate_dict[i] for i in sorted_ids]
            # Acquire a file lock to prevent concurrent writes
            lock_path = output_path + ".lock"
            lock = FileLock(lock_path)
            with lock:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(sorted_results, f, ensure_ascii=False, indent=2)
                with open(intermediate_path, "w", encoding="utf-8") as f:
                    json.dump(sorted_intermediate, f, ensure_ascii=False, indent=2)

    # single process
    # for idx, qa in enumerate(tasks):
    #     if "Is the person running towards the camera or running in the direction away from the camera?" not in qa[1]["Q"]:
    #         continue
    #     idx, result, intermediate = process_one_qa(qa)
    #     # Store by ID
    #     results_dict[idx] = result
    #     intermediate_dict[idx] = intermediate
    #     # Write sorted results up to this point
    #     sorted_ids = sorted(results_dict)
    #     sorted_results = [results_dict[i] for i in sorted_ids]
    #     sorted_intermediate = [intermediate_dict[i] for i in sorted_ids]
    #     # Acquire a file lock to prevent concurrent writes
    #     lock_path = output_path + ".lock"
    #     lock = FileLock(lock_path)
    #     with lock:
    #         with open(output_path, "w", encoding="utf-8") as f:
    #             json.dump(sorted_results, f, ensure_ascii=False, indent=2)
    #         with open(intermediate_path, "w", encoding="utf-8") as f:
    #             json.dump(sorted_intermediate, f, ensure_ascii=False, indent=2)



def process_videos(cvrr_dataset_path: str, image_base_path: str, output_dir: str, num_frames: int=8, resume: bool=False, processes: int=16, folder_processes: int=None):
    """Main function to process all video folders and questions with parallel processing."""
    if folder_processes is None:
        folder_processes = max(1, multiprocessing.cpu_count() // 2)
    
    child_processes = max(1, processes // folder_processes)
    
    all_folders = os.listdir(cvrr_dataset_path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=folder_processes) as executor:
        futures = []
        for folder in all_folders:
            future = executor.submit(
                process_sigle_folder, 
                folder, 
                cvrr_dataset_path, 
                image_base_path, 
                output_dir, 
                num_frames, 
                resume, 
                child_processes
            )
            futures.append(future)
        
        # Wait for all tasks to complete
        completed_folders = []
        for future in concurrent.futures.as_completed(futures):
            try:
                # The result returned by future is not used, but process it to catch errors
                future.result()
                completed_folders.append(True)
            except Exception as e:
                print(f"[ERROR] Error processing folder: {e}", file=sys.stderr)
    
    print(f"All folders processed. Completed: {len(completed_folders)}")

# def process_videos(cvrr_dataset_path: str, image_base_path: str, output_dir: str, num_frames: int=8, resume: bool=False, processes: int=16, folder_processes: int=None):
#     """Main function to process all video folders and questions."""
#     all_folders = os.listdir(cvrr_dataset_path)
#     for folder in all_folders:
#         print ("Processing folder: ", folder)
#         process_sigle_folder(folder, cvrr_dataset_path, image_base_path, output_dir, num_frames, resume, processes)
#     print("All folders processed.")


def parse_args():
    """Parse command-line arguments for video QA processing."""
    parser = argparse.ArgumentParser(description="LangChain-based video question answering")
    parser.add_argument("--cvrr_dataset_path", required=True, help="Path to the CVRR-ES dataset.")
    parser.add_argument("--image_base_path", required=True, help="Base path for the images.")
    parser.add_argument("--video_summary_path", required=True, help="Path to the video summary.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the results.")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of frames to sample from each video.")
    parser.add_argument("--processes", "-p", type=int, default=16, help="Number of worker processes")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume from the last checkpoint.")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    if is_data_evrything_ok(args.cvrr_dataset_path, args.image_base_path, args.video_summary_path):
        print ("All data ok!")
    else:
        raise RuntimeError("Data or summary TXT files are missing or unreadable.")

    # Start Process
    os.environ["VIDEO_SUMMARY_PATH"] = args.video_summary_path
    process_videos(args.cvrr_dataset_path, args.image_base_path, args.output_dir, args.num_frames, args.resume, args.processes, folder_processes=1)