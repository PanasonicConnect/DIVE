from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Dict, List
from langchain.agents import AgentExecutor, create_openai_tools_agent
# from tools.vision_tool import vision_tool
from tools.vision_tool_fixed_input import analyze_video_openai, analyze_video_openai_with_frame_selection
from tools.analyze_video_gemini import analyze_video_gemini
from utils import encode_images, load_video_summary_message
import os
import json
import textwrap


# Initialize OpenAI client
llm = ChatOpenAI(model="gpt-4.1", temperature=1.0, disable_streaming=True)

class SplitQuestionSubQuestions(BaseModel):
    questions: List[str] = Field(..., description="The list that stores sub-questions related to a video. Example: [\"Question1\", \"Question2\"]")

# class AnswerQuestion(BaseModel):
#     answer: str = Field(..., description="The answer to the question based on the video frames.")

class ContinueReasons(BaseModel):
    reasons: str = Field(..., description="The reasons for the decision to continue or stop the process.")
    continue_flag: bool = Field(..., description="The boolean value indicating whether to continue the process.")


def infer_question_intent(state) -> dict:
    print("=== Inferred Question Intent ===")
    """
    Infers the user’s true intent behind the original video question,
    e.g. “Count only people in the foreground, ignore distant figures.”

    Args:
        state (dict): Must contain "original_question".
    Returns:
        dict: { "question_intent": <str> }
    """
    original_question = state["original_question"]
    video_summary = load_video_summary_message(state["video_path"])

    system_message = "You are an expert at clarifying question intents for text-based QA. For each question about a video and its VIDEO SUMMARY, deliver only the most comprehensive and accurate interpretation of what the user truly seeks to know. Be explicit about observable evidence required (visual actions, sounds, interactions), distinguishing between similar but distinct requests (e.g., confirmation vs. description, actual vs. pretend actions, or group vs. individual counts) per the user's wording. Never add context or assumptions beyond what's given. Please express the intent concisely and directly, without explanation or introductory language."
    user_message = textwrap.dedent(f"""
        Video Summary:
        {video_summary}

        Question:
        {original_question}

        What, exactly and only as supported by what is seen or heard in the video, is the user's specific question intent—including any needed clarifications, verification steps, and constraints?
        Direct Intent:
    """)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message),
        HumanMessage(content=[{"type": "text", "text": user_message},])
    ])

    # Invoke the LLM
    chain = prompt | llm
    output = chain.invoke({"original_question": original_question})
    print(f"Intent of Question: {output.content.strip()}")
    state["question_intent"] = output.content.strip()
    return state


def split_question(state) -> dict:
    """
    Split a question into sub-questions for step-by-step analysis.

    Args:
        state (dict): The state containing the original question.
    Returns:
        dict: A dictionary containing sub-questions and the iteration count.
    """
    print('=== Breaking down question ===')

    # Prepare the input data
    original_question = state["original_question"]

    # Prepare summary message
    video_summary_message = load_video_summary_message(state["video_path"])

    # Prepare the System and User messages
    system_message = f"You are an assistant who takes a question about a video and breaks it down into multiple specific sub-questions needed for analysis. The number of sub-questions is not fixed. Please respond in form of a list of sub-questions."
    user_message = textwrap.dedent(f"""
        Below is an original question about a video.
        Please break this question into multiple sub-questions to examine them step-by-step.
        Verify that the people/objects/actions/situations/numbers/feelings, etc. assumed in the question are correct.
        Question: {original_question}
    """)

    # Prepare the prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message),
        HumanMessage(content=video_summary_message),
        HumanMessage(content=user_message)
    ])

    # Invoke the LLM
    structured_llm = llm.with_structured_output(SplitQuestionSubQuestions)
    chain = prompt | structured_llm
    output: SplitQuestionSubQuestions = chain.invoke({"original_question": original_question})

    print("Sub-questions generated:")
    for i in range(len(output.questions)):
        print(f"Q{i}: {output.questions[i]}")

    return {"sub_questions": {0: output.questions}, "iter": 0}


def answer_question(state) -> dict:
    """
    Analyzes video frames and answers a question based on the visual content.
    Now uses LangChain Agent to answer the question.
    """
    print("=== Analyzing video frames (Agent) ===")
    iter              = state["iter"]
    continue_flag     = state["continue_flag"]
    qa_results        = state["qa_results"]
    original_question = state["original_question"]
    sub_questions     = state["sub_questions"]
    tool_results      = state["tool_results"]
    video_path        = state["video_path"]

    # Prepare summary message
    video_summary_message = load_video_summary_message(video_path)
    # print (video_summary_message)

    if continue_flag:
        current_question = sub_questions[iter][0]
    else:
        current_question = original_question

    # Agent Prompt
    system_message = textwrap.dedent("""
        You are an AI assistant that analyzes video frames and answers questions based on the visual content.
        The video contents are provided via a dedicated analysis tool.
        You MUST use at least one of the available tools to extract visual or audio cues before answering any questions. 
        Never attempt to answer without first invoking a tool and using its output as evidence.
        Your answer must be based on the tool's output. Please respond in string format and include your reasoning.

        You have access to two tools: `GPT-4.1 Tool` and `Gemini 2.5pro Tool`.
        - The Gemini tool analyzes one frame per second and uses audio information. It's useful for understanding the sequence of events, temporal reasoning, object continuity, and social interactions based on both visuals and sound.
        - The GPT-4.1 tool analyzes a few sampled frames without audio. It is better at capturing fine-grained visual details, detecting implausible or anomalous actions, and complex visual scenes.
    """)

    # Prepare Intent message
    question_intent = state.get("question_intent", "")
    intent_message = f"The user's underlying intent behind this question: {question_intent}"

    # Prepare prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message),
        HumanMessage(content=intent_message),
        HumanMessage(content=video_summary_message),
        HumanMessagePromptTemplate.from_template("Now please answer the following question:\nQ: {current_question}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # prepare QA result msg
    if len(qa_results) > 0:
        qa_result_message = "You currently have the following questions and answers. You may use this as a reference.\n"
        for i in range(len(qa_results)):
            qa_result_message += f"Q{i}: {qa_results[i]['Q']}\nA{i}: {qa_results[i]['A']}\n"
    else:
        qa_result_message = ""

    # Set environment variables of the state for agent tools
    for key, value in state.items():
        os.environ[key.upper()] = str(value)
    os.environ["CURRENT_QUESTION"] = current_question
    os.environ["QA_RESULT_MESSAGE"] = qa_result_message

    inputs = {
        "qa_result_message": qa_result_message,
        "current_question": current_question,
    }

    agent_llm = llm
    agent = create_openai_tools_agent(agent_llm, [analyze_video_gemini, analyze_video_openai_with_frame_selection], prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[analyze_video_gemini, analyze_video_openai_with_frame_selection], prompt=prompt, return_intermediate_steps=True)

    # Ask Agent to answer the question and check if the vision tool was used
    for i in range(5):
        result = agent_executor.invoke(inputs)
        answer = result["output"]

        # Check if the vision tool was used
        intermediate_steps = result.get("intermediate_steps", [])
        if intermediate_steps:
            break
        else:
            print("Error: Tool was not used during agent execution. Retrying...")

    # Post process
    intermediate_steps = result.get("intermediate_steps", [])
    tools_used: List[Dict[str, str]] = [ {"tool": action.tool, "result": observation} for action, observation in intermediate_steps ]
    tool_results += tools_used
    qa_results.append({"Q": current_question, "A": answer})

    print (f"Agent output: {answer}")

    if continue_flag:
        print("Answer to the question:")
        print(f"Q{iter}: {current_question}\nA{iter}: {answer}")
        return {"qa_results": qa_results, "tool_results": tool_results}
    else:
        print("Answer to the original question:")
        print(f"Q: {original_question}\nA: {answer}")
        return {"qa_results": qa_results, "final_answer": answer, "tool_results": tool_results}


def refine_sub_questions(state) -> dict:
    """
    Refine sub-questions based on the answers obtained.
    Args:
        state (dict): The state containing the original question and sub-questions.
    """
    print("=== Refining sub-questions ===")

    # Prepare the input data
    iter              = state["iter"]
    original_question = state["original_question"]
    qa_results        = state["qa_results"]
    sub_questions     = state["sub_questions"]
    planning_question = sub_questions[iter][1:]
    iter += 1

    # Prepare the System messages
    system_message = textwrap.dedent(f"""
        You are an assistant who receives questions and their answers about a video, along with multiple specific sub-questions that are plan to validate in the future.
        Your role is to refine and enhance the sub-questions if necessary in answering the original question, and to output them as is if not necessary.
        There is no set number of sub-questions. Please respond in form of a list of sub-questions.
    """)

    # Prepare user message
    user_message = textwrap.dedent(f"""
        Below are original questions about the video.

        Original question: {original_question}

        Please break this question into multiple sub-questions to examine them step-by-step.
        Verify that the people/objects/actions/situations/numbers/feelings, etc. assumed in the question are correct.
    """)

    # Prepare QA result message
    if len(qa_results) > 0:
        qa_result_message = """
        Some questions have now been validated. I will share the question and answer with you.
        """
        for i in range(len(qa_results)):
            qa_result_message += f"Q{i}: {qa_results[i]['Q']}\nA{i}: {qa_results[i]['A']}\n"
    else:
        qa_result_message = ""

    # Prepare planning question message
    if len(planning_question) > 0:
        plan_message = f"""
        The following is a group of sub-questions that we plan to validate in the future. If you believe a change in the group of sub-questions is necessary, please change the group of sub-questions. If you do not need to change the group of sub-questions, please output the group of sub-questions as it is.
        """
        for i in range(len(planning_question)):
            plan_message += f"Q{i + len(qa_results)}: {planning_question[i]}\n"
    else:
        plan_message = "There are no additional questions that we plan to validate in the future. Please plan additional questions."

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message),
        HumanMessage(content=user_message + qa_result_message + plan_message)
    ])

    # Invoke the LLM
    structured_llm = llm.with_structured_output(SplitQuestionSubQuestions)
    chain = prompt | structured_llm
    output: SplitQuestionSubQuestions = chain.invoke({"iter": iter,
                                "original_question": original_question,
                                "qa_results": qa_results,
                                "sub_questions": planning_question})
    sub_questions[iter] = output.questions

    print("Refined sub-questions:")
    for i in range(len(output.questions)):
        print(f"Q{iter + i}: {output.questions[i]}")
    return {"sub_questions": sub_questions, "iter": iter}


def should_continue(state) -> dict:
    """
    Determine whether to continue the process based on the number of iterations.

    Args:
        state (dict): The state containing the iteration count and sub-questions.

    Returns:
        dict: A dictionary indicating whether to continue or not.
    """
    print("=== Checking if we should continue ===")

    # Prepare the input data
    iter              = state["iter"]
    max_iter          = state["max_iter"]
    original_question = state["original_question"]
    qa_results        = state["qa_results"]
    planning_question = state["sub_questions"][iter]
    continue_reasons  = state["continue_reasons"]

    # Check if the maximum number of iterations has been reached
    if iter > max_iter:
        print(f"Continue?: {False}")
        reach_max_iter_message = f"The maximum number of iterations ({max_iter}) has been reached."
        print(f"Reasons for decision: {reach_max_iter_message}")
        continue_reasons.append(reach_max_iter_message)
        return {"continue_flag": False, "continue_reasons": continue_reasons}

    # Prepare the System and User messages
    system_message = textwrap.dedent(f"""
        You are an assistant who receives questions and their answers about a video, along with multiple specific sub-questions that are plan to validate in the future.
        Your role is to verify whether sufficient information has been gathered to resolve the original question, and to determine whether additional sub-questions should be processed.
        If you have any concerns or need further clarification, please continue with sub-questions. Only conclude the process when you are absolutely confident that you can provide an accurate answer based on the information currently available.
        Please respond in string format as reasons for your decision, and respond in boolean format to indicate whether to continue the process.
    """)
    user_message = textwrap.dedent(f"""
        Below are original questions about the video.
        Original question: {original_question}

        This question was broken down into multiple sub-questions to examine them step-by-step.
        The purpose of these sub-questions is to verify that the people/objects/actions/situations/numbers/feelings, etc. assumed in the question are correct.
    """)

    # Prepare QA result message
    if len(qa_results) > 0:
        qa_result_message = """
        Some questions have now been validated. I will share the question and answer with you.
        """
        for i in range(len(qa_results)):
            qa_result_message += f"Q{i}: {qa_results[i]['Q']}\nA{i}: {qa_results[i]['A']}\n"
    else:
        qa_result_message = ""

    # Prepare planning message
    plan_message = "Please first evaluate whether the supplemental questions listed below are needed to fully answer it. Explain your reasoning in your response."
    if len(planning_question) > 0:
        for i in range(len(planning_question)):
            plan_message += f"Q{i + len(qa_results)}: {planning_question[i]}\n"
    else:
        plan_message += "There are no supplemental questions to evaluate."

    # Prepare the prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message),
        HumanMessage(content=user_message + qa_result_message + plan_message)
    ])

    # Invoke the LLM
    structured_llm = llm.with_structured_output(ContinueReasons)
    chain = prompt | structured_llm
    output: ContinueReasons = chain.invoke({"iter": iter,
                                "max_iter": max_iter,
                                "original_question": original_question,
                                "qa_results": qa_results,
                                "sub_questions": planning_question})

    print("Decision to continue:")
    print(f"Continue?: {output.continue_flag}")
    print(f"Reasons for decision: {output.reasons}")
    continue_reasons.append(output.reasons)

    return {"continue_flag": output.continue_flag, "continue_reasons": continue_reasons}


def finalize_answer(state) -> dict:
    """
    Finalize the answer based on the gathered information and sub-questions.
    Args:
        state (dict): The state containing the original question, sub-questions, and QA results.
    Returns:
        dict: A dictionary containing the final answer to the original question.
    """
    print("=== Finalizing answer ===")

    # Prepare the input data
    original_question = state["original_question"]
    qa_results        = state["qa_results"]
    question_intent   = state["question_intent"]
    video_summary     = load_video_summary_message(state["video_path"])

    # Prepare QA result message
    if len(qa_results) > 0:
        qa_result_message = """
        Some questions have now been validated. I will share the question and answer with you.
        """
        for i in range(len(qa_results)):
            qa_result_message += f"Q{i}: {qa_results[i]['Q']}\nA{i}: {qa_results[i]['A']}\n"
    else:
        qa_result_message = ""

    system_message = "You are a factual and impartial video question answering expert. Your task is to deliver precise, direct answers based solely on the observable evidence provided. Avoid speculation. When applicable, explicitly state if the available information supports, contradicts, or cannot resolve the claim in the question. Cite concise evidence directly tied to your conclusion."
    user_message = textwrap.dedent(f"""
        Using only the context below, answer the main question concisely and cite specific evidence from the video summary and QA list.

        Video Summary:
        {video_summary}

        Prior QA:
        {qa_results}

        Main Question:
        {original_question}
        Intent:
        {question_intent}

        Base your answer strictly on the details above, and clearly indicate if evidence is insufficient to answer or if the claim is supported or contradicted.
    """)

    # Prepare the prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message),
        HumanMessage(content=user_message)
    ])

    # Invoke the LLM
    chain = prompt | llm
    output = chain.invoke({})

    final_answer_text = output.content.strip()

    print("Final Answer:")
    print(final_answer_text)

    return {"final_answer": final_answer_text}
