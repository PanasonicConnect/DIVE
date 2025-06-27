import os
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from pydantic import BaseModel, Field
from pathlib import Path
import sys
import json

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
from utils import encode_images, select_frames


llm = ChatOpenAI(model="gpt-4.1", temperature=0.0, disable_streaming=True)

class AnswerQuestion(BaseModel):
    reason: str = Field(..., description="The reasoning behind the answer to the question based on the video frames.")
    answer: str = Field(..., description="The answer to the question based on the video frames.")
    concern: str = Field(..., description="Any concerns or issues identified in the video frames related to the question.")


@tool("gpt4_vision_tool")
def analyze_video_openai() -> str:
    """
    GPT-4.1 Tool:
    - Samples 8 frames from the video (no audio)
    - Captures fine-grained visual details
    - Detects implausible or anomalous actions
    - Interprets emotions and complex visual scenes

    Returns:
        str: A string containing the analysis based solely on those sampled frames.
    """

    print ("vision_tool called!!!")

    current_question  = os.environ["CURRENT_QUESTION"]
    qa_result_message = os.environ["QA_RESULT_MESSAGE"]
    image_path        = os.environ["IMAGE_PATH"]

    print (f"image_path: {image_path}")
    print (f"current_question: {current_question}")
    print (f"qa_result_message: {qa_result_message}")


    system_message = (
        "You are an AI assistant that analyzes video frames and answers questions based on the visual content."
        "The video frames are provided in Base64 format."
        "Use the visual cues from the frames to answer the questions."
        "Verify that the people/objects/actions/situations/numbers/feelings, etc. assumed in the question are correct."
        "Please respond in string format, including the reason for your answer."
    )

    # prepare the encoded_images
    encoded_frames = encode_images(image_path, frame_num=8, detail="auto")

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message),
        HumanMessage(content="These are frames from a video. Use the visual cues to answer the following question."),
        HumanMessage(content=qa_result_message),
        HumanMessage(content=[ {"type": "text", "text": f"Now please answer the following question.\nQ: {current_question}"}, *encoded_frames ])
    ])

    structured_llm = llm.with_structured_output(AnswerQuestion)
    chain = prompt | structured_llm
    output: AnswerQuestion = chain.invoke({})

    print (f"tool output: {output.answer}")
    return output.answer

# print(analyze_video_openai.args_schema.model_json_schema())

@tool("gpt4_vision_tool_with_frame_selection")
def analyze_video_openai_with_frame_selection() -> str:
    """
    GPT-4.1 Tool (with frame selection):
    - Selects frames based on question/context from the video (no audio)
    - Captures fine-grained visual details
    - Detects implausible or anomalous actions
    - Interprets emotions and complex visual scenes

    Returns:
        str: A string containing the analysis based solely on those selected frames.
    """

    print ("vision_tool_with_frame_selection called!!!")

    current_question  = os.environ["CURRENT_QUESTION"]
    qa_result_message = os.environ["QA_RESULT_MESSAGE"]
    image_path        = os.environ["IMAGE_PATH"]
    video_metadata_str = os.environ["VIDEO_METADATA"]
    # Fix for malformed JSON with single quotes
    if video_metadata_str and video_metadata_str.strip().startswith("{") and "'" in video_metadata_str:
        video_metadata_str = video_metadata_str.replace("'", '"')
    video_metadata = json.loads(video_metadata_str)
    video_id          = os.environ["VIDEO_ID"]

    print (f"image_path: {image_path}")
    print (f"current_question: {current_question}")
    print (f"qa_result_message: {qa_result_message}")

    selected_indices, selection_reason = select_frames(video_id, video_metadata, current_question)

    system_message = (
        "You are an AI assistant that analyzes video frames and answers questions based on the visual content.\n"
        "Use the visual cues from the frames to answer the questions.\n"
        "Verify that the people/objects/actions/situations/numbers/feelings, etc. assumed in the question are correct.\n"
        "Your response must always include 'Reason', 'Answer', and 'Concern'. "
        "'Reason' should describe the basis or evidence for your answer, "
        "'Answer' should provide a direct response to the question, "
        "and 'Concern' should mention any uncertainties or points that may require further verification."
    )

    # prepare the encoded_images
    encoded_frames = encode_images(image_path, frame_num=8, detail="auto", indices=selected_indices)

    user_message = (
        "These are frames sampled from the entire video, selected because they are considered relevant to the question.\n"
        f"Sampling reason: {selection_reason}\n\n"
        "Use the visual cues from these frames to answer the following question.\n"
        f"Q: {current_question}"
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message),
        HumanMessage(content=[*encoded_frames, {"type": "text", "text": user_message}]),
    ])

    structured_llm = llm.with_structured_output(AnswerQuestion)
    chain = prompt | structured_llm
    output: AnswerQuestion = chain.invoke({})

    print (f"tool output: {output.answer}")

    result = (
        "Answer: " + output.answer.strip() + "\n" +
        "Reason: " + output.reason.strip() + "\n" +
        "Concern: " + output.concern.strip()
    )
    return result
