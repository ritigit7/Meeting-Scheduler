import streamlit as st
import json
from datetime import datetime, timedelta
from Meeting_AI_Agent import get_meeting_data, memory
from streamlit_calendar import calendar

# Streamlit page config
st.set_page_config(page_title="Meeting Calendar", layout="wide")
st.title("📅 Meeting Calendar App")
st.write("Extract meetings and visualize them on a calendar.")

# Memory for storing meetings
if "meetings" not in st.session_state:
    st.session_state.meetings = []

# Function to convert the date format
def convert_date_format(date_str):
    try:
        # Extract the date part "25-02-2025"
        date_part = date_str.split(",")[1].strip()
        parsed_date = datetime.strptime(date_part, "%d-%m-%Y")
        return parsed_date.strftime("%Y-%m-%dT%H:%M:%S")
    except Exception as e:
        st.error(f"Failed to convert date format: {e}")
        return None

# User input for meeting prompt
user_input = st.text_area("📝 Enter your meeting request:", height=150)

if st.button("🔍 Extract Meeting Details"):
    if user_input:
        with st.spinner("Processing..."):
            try:
                meeting_data = get_meeting_data(user_input)
            except Exception as e:
                st.error(f"Error extracting meeting data: {e}")
                meeting_data = None

        if meeting_data:
            st.success("✅ Meeting details extracted successfully!")
            st.json(meeting_data.dict())

            # Convert the date format for calendar
            start_time = convert_date_format(meeting_data.date)
            if start_time:
                end_time = (datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S") 
                            + timedelta(hours=meeting_data.duration_of_meeting)
                           ).strftime("%Y-%m-%dT%H:%M:%S")

                # Store meeting in session state
                meeting_event = {
                    "title": meeting_data.title,
                    "start": start_time,
                    "end": end_time,
                    "description": f"Participants: {', '.join(meeting_data.participants)}\n{meeting_data.confirmation_message}"
                }
                st.session_state.meetings.append(meeting_event)
            else:
                st.error("❌ Failed to parse meeting date.")
        else:
            st.error("❌ No valid meeting detected. Please refine your input.")
    else:
        st.warning("⚠️ Please enter a meeting request.")

# Display Calendar with Meetings
st.markdown("## 📅 Calendar View")

calendar_events = [
    {
        "title": event["title"],
        "start": event["start"],
        "end": event["end"],
        "extendedProps": {"description": event["description"]}
    }
    for event in st.session_state.meetings
]

calendar(
    events=calendar_events,
    options={
        "initialView": "dayGridMonth",
        "editable": True,
        "eventClick": True,
        "height":500,
        "headerToolbar": {
            "left": "prev,next today",
            "center": "title",
            "right": "dayGridMonth,timeGridWeek,timeGridDay"
        },
        "eventTimeFormat": {
            "hour": '2-digit',
            "minute": '2-digit',
            "meridiem": False
        }
    },
    key="meeting_calendar"
)

# Display memory of past interactions
st.markdown("## 🧠 Conversation Memory")
if memory:
    for msg in memory:
        st.write(f"**{msg['role'].capitalize()}:** {msg['content']}")
else:
    st.write("No memory available yet.")

st.markdown("---")
st.write("💡 Powered by LLaMA3.2, Streamlit, and Pydantic.")
