# meeting_agent.py
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import json

# Initialize Ollama client
ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

MODEL = "llama3.2"
# MODEL = "deepseek-r1:1.5b"

# ------------------------- Memory for Task Storage -------------------------
memory: List[Dict[str, str]] = []


def add_to_memory(role: str, content: str):
    """Add messages to memory for context retention."""
    memory.append({"role": role, "content": content})
    if len(memory) > 10:  # Limit memory size
        memory.pop(0)


# ------------------------- Pydantic Models -------------------------
class MeetingInfo(BaseModel):
    """Extracted information about whether the user input relates to a meeting."""
    meeting_description: str = Field(description="Brief description of the meeting purpose.")
    is_calendar_event: bool = Field(description="Indicates if the input is a calendar event.")
    confidence_score: float = Field(description="Confidence level (0 to 1) for meeting detection.")


class MeetingDetails(BaseModel):
    """Detailed information about the meeting event."""
    title: str = Field(description="The title or subject of the meeting.")
    date: str = Field(description="The date and time of the meeting in match format %A, %d-%m-%Y.")
    duration_of_meeting: float = Field(description="Duration of the meeting in hours.")
    participants: List[str] = Field(description="List of participants attending the meeting.")


class EventConfirmation(BaseModel):
    """Confirmation message for the meeting with optional calendar link."""
    confirmation_message: str = Field(description="Natural language confirmation message for the user.")
    calendar_link: Optional[str] = Field(description="Generated calendar link, if applicable.")


class MeetingData(BaseModel):
    """Final aggregated meeting information."""
    meeting_description: str
    is_calendar_event: bool
    confidence_score: float
    title: str
    date: str
    duration_of_meeting: float
    participants: List[str]
    confirmation_message: str
    calendar_link: Optional[str] = None


# ------------------------- LLM Utility Functions -------------------------
def llm_parse(prompt: str, messages: List[dict], response_model: BaseModel):
    """Generalized LLM parser for structured output."""
    try:
        task = ollama_client.beta.chat.completions.parse(
            model=MODEL,
            messages=messages + memory,
            response_format=response_model
        )
        add_to_memory("user", prompt)
        add_to_memory("assistant", task.choices[0].message.content)
        return task.choices[0].message.parsed
    except Exception as e:
        raise ValueError(f"Failed to parse LLM response: {e}")


def extract_meeting_info(user_input: str) -> Optional[MeetingInfo]:
    """Extracts basic meeting information from user input."""
    messages = [
        {'role': 'system', 'content': 'Analyze the user input to detect if it is a meeting-related query.'},
        {'role': 'user', 'content': user_input}
    ]
    return llm_parse(user_input, messages, MeetingInfo)


def extract_meeting_details(user_input: str) -> Optional[MeetingDetails]:
    """Extracts detailed meeting information from user input."""
    messages = [
        {'role': 'system', 'content': 'Extract the meeting title, date, duration in hour, and participants.Date must be match format "%A, %d-%m-%Y"'},
        {'role': 'user', 'content': user_input}
    ]
    return llm_parse(user_input, messages, MeetingDetails)


def generate_event_confirmation(user_input: str) -> Optional[EventConfirmation]:
    """Generates a confirmation message for the meeting event."""
    messages = [
        {'role': 'system', 'content': 'Generate a confirmation message and optional calendar link for the event.'},
        {'role': 'user', 'content': user_input}
    ]
    return llm_parse(user_input, messages, EventConfirmation)


# ------------------------- Agent Function -------------------------
def agent_task(prompt: str) -> str:
    """Simple agent to delegate tasks and return results."""
    task_messages = [
        {"role": "system", "content": "You are a task-executing AI agent for meeting-related actions."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = ollama_client.chat.completions.create(
            model=MODEL,
            messages=task_messages + memory
        )
        add_to_memory("user", prompt)
        add_to_memory("assistant", response.choices[0].message.content)
        return response.choices[0].message.content
    except Exception as e:
        return f"Failed to execute agent task: {e}"


# ------------------------- Main Function -------------------------
def get_meeting_data(prompt: str) -> Optional[MeetingData]:
    """Aggregates all meeting-related information into a structured format with agent support."""
    info = extract_meeting_info(prompt)
    if not info or not info.is_calendar_event or info.confidence_score < 0.6:
        print("❌ No valid meeting detected.")
        return None

    # Extract details and confirmation
    details = extract_meeting_details(prompt)
    confirmation = generate_event_confirmation(prompt)

    # Agent task example
    agent_result = agent_task(f"Summarize the meeting details for: {details.title}")

    # Final aggregated data
    return MeetingData(
        meeting_description=info.meeting_description,
        is_calendar_event=info.is_calendar_event,
        confidence_score=info.confidence_score,
        title=details.title,
        date=details.date,
        duration_of_meeting=details.duration_of_meeting,
        participants=details.participants,
        confirmation_message=confirmation.confirmation_message + f"\n\nAgent Task: {agent_result}",
        calendar_link=confirmation.calendar_link
    )


if __name__ == "__main__":
    user_prompt = "Let's schedule a 1-hour team meeting on 25-02-2025, Tuesday at 2 PM with Alice and Bob to discuss the project roadmap."
    result = get_meeting_data(user_prompt)

    if result:
        print("\n✅ Meeting Data:")
        print(result.json(indent=4))
    else:
        print("❌ No valid meeting details found.")
