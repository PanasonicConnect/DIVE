import os
import mimetypes
from google import genai
from google.genai.types import HttpOptions, Part, GenerateContentConfig, SafetySetting
from google.genai.types import HarmCategory, HarmBlockThreshold
from google.cloud import storage
from langchain.agents import tool
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_result, retry_if_exception_type, retry_any
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

client = genai.Client(
    vertexai=True,
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ["GOOGLE_CLOUD_LOCATION"],
    http_options=HttpOptions(api_version="v1")
)

def get_mime_type(gcs_uri: str) -> str:
    mime, _ = mimetypes.guess_type(gcs_uri)
    if mime:
        return mime

    _, path = gcs_uri.split("gs://", 1)
    bucket_name, blob_name = path.split("/", 1)
    client = storage.Client(project=os.environ["GOOGLE_CLOUD_PROJECT"])
    blob = client.bucket(bucket_name).get_blob(blob_name)
    return blob.content_type or "application/octet-stream"


class AnswerQuestion(BaseModel):
    answer: str = Field(..., description="The answer to the question based on the video contents.")
    reason: str = Field(..., description="The reasoning behind the answer provided.")
    concern: str = Field(..., description="The concern or aspect of the question that was addressed in the answer.")

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_any(
        retry_if_result(lambda text: not text),
        retry_if_exception_type(Exception)
    ),
    retry_error_callback=lambda state: (
        "I’m unable to perform the analysis due to the error; "
        "please use another tool."
    )
)
def ask_gemini_vertex(system_prompt: str, user_prompt: str, gcs_uri: str = "", temperature: float = 0.7) -> AnswerQuestion:
    """
    Call the Gemini 2.5 Pro model using Langchain's ChatVertexAI with structured output,
    retrying on empty or problematic responses.
    """
    llm = ChatVertexAI(
        model_name="gemini-2.5-pro-preview-05-06",
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.environ["GOOGLE_CLOUD_LOCATION"],
        temperature=temperature,
        max_output_tokens=8192,
    )

    mime_type = get_mime_type(gcs_uri)
    video_dict_part = {
        "type": "media",  # Specifies the type of media
        "file_uri": gcs_uri,  # The GCS URI of the video
        "mime_type": mime_type,  # The MIME type of the video
    }

    message_content = [
        video_dict_part,
        {"type": "text", "text": user_prompt},
    ]

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=message_content)
    ]

    chain = llm.with_structured_output(AnswerQuestion)
    output: AnswerQuestion = chain.invoke(messages)

    result = (
        "Answer: " + output.answer.strip() + "\n"
        "Reason: " + output.reason.strip() + "\n"
        "Concern: " + output.concern.strip()
    )

    return result


@tool("gemini_video_tool")
def analyze_video_gemini() -> str:
    """
    Gemini 2.5pro Tool:
    - Analyzes one frame per second, plus accompanying audio
    - Excels at sequence-of-events understanding and temporal reasoning
    - Tracks object continuity across frames
    - Parses social interactions using both visual and sound cues

    Returns:
        str: A string summarizing the combined audio-visual analysis.
    """

    current_question  = os.environ["CURRENT_QUESTION"]
    video_path        = os.getenv("VIDEO_PATH")
    bucket_name       = os.getenv("GOOGLE_CLOUD_BACKET_NAME")
    gcs_uri           = os.path.join("gs://", bucket_name, video_path)
    gcs_uri           = gcs_uri.replace("\\", "/")  # Ensure correct path format

    system_message = (
        "You are an AI assistant that analyzes video contents and answers questions.\n"
        "Use the visual and audio cues from the video to answer the questions.\n"
        "If there are environmental sounds or people speaking in the video, make sure to leverage the audio information as well.\n"
        "Verify that the people/objects/actions/situations/numbers/feelings, etc. assumed in the question are correct.\n"
        "Your response must always include 'Reason', 'Answer', and 'Concern'. "
        "'Reason' should describe the basis or evidence for your answer, "
        "'Answer' should provide a direct response to the question, "
        "and 'Concern' should mention any uncertainties or points that may require further verification."
    )

    user_message = (
        "Use the visual and audio cues from the video to answer the following question.\n"
        f"Q: {current_question}"
    )

    result = ask_gemini_vertex(system_prompt=system_message, user_prompt=user_message, gcs_uri=gcs_uri, temperature=0.0)

    print (f"Gemini Tool Result: {result}")
    return result
