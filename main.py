import os
import json
import gradio as gr
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd
from ics import Calendar, Event
from datetime import datetime

# --- 1. Configuration and Setup ---
load_dotenv()

# IMPORTANT: Ensure your .env file has your GEMINI_API_KEY
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    raise EnvironmentError("GEMINI_API_KEY not found. Please set it in your .env file.")


# --- 2. Tool Definitions (AI Tool + Custom Tool) ---

def extract_deadlines_from_syllabus_tool(syllabus_text: str, model) -> list:
    """
    An AI-powered tool using Gemini to parse syllabus text and extract coursework deadlines.
    Returns a list of structured deadline dictionaries.
    """
    print("TOOL: Using Gemini to extract deadlines from syllabus...")

    prompt = f"""
    You are an expert academic assistant. Your job is to analyze the following course syllabus text and extract all important deadlines.
    The output must be a valid JSON array of objects.

    Each object in the array represents a single piece of coursework and must have three keys:
    1. "assignment_name": A concise name for the coursework (e.g., "Essay 1", "Midterm Exam").
    2. "due_date": The deadline for the assignment in "YYYY-MM-DD" format.
    3. "assignment_type": The type of coursework. Choose from: "Essay", "Quiz", "Exam", "Project", "Presentation", "Other".

    Syllabus Text:
    ---
    {syllabus_text}
    ---

    Analyze the text carefully. Pay attention to dates, assignment descriptions, and types.
    Return ONLY the raw JSON array. Do not include any other text or formatting.
    Example:
    [
      {{"assignment_name": "Problem Set 1", "due_date": "2025-09-15", "assignment_type": "Project"}},
      {{"assignment_name": "Midterm Examination", "due_date": "2025-10-20", "assignment_type": "Exam"}}
    ]
    """

    try:
        response = model.generate_content(prompt)
        # Clean the response to ensure it's valid JSON
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        extracted_data = json.loads(json_text)
        print("TOOL: Successfully extracted deadline data.")
        return extracted_data
    except (json.JSONDecodeError, AttributeError) as e:
        # Provide more specific feedback if parsing fails
        print(f"Error parsing JSON from Gemini response: {e}\nRaw response text: '{response.text}'")
        return []
    except Exception as e:
        print(f"An unexpected error occurred in the Gemini tool: {e}")
        return []


def generate_ical_file_tool(course_name: str, deadlines: list, output_dir: str = "calendars") -> str:
    """
    A custom, non-AI tool that takes structured deadline data and generates
    a universal .ics calendar file. Returns the path to the file.
    """
    print("TOOL: Generating .ics calendar file...")
    if not deadlines:
        return ""

    os.makedirs(output_dir, exist_ok=True)
    cal = Calendar()

    for item in deadlines:
        try:
            # Create an all-day event for the due date
            event_date = datetime.strptime(item['due_date'], "%Y-%m-%d").date()
            event = Event()
            event.name = f"[{course_name}] - {item['assignment_name']} ({item['assignment_type']})"
            event.begin = event_date
            event.make_all_day()
            cal.events.add(event)
        except (ValueError, KeyError) as e:
            print(f"Skipping invalid item: {item}. Error: {e}")
            continue

    # Generate a unique filename
    safe_course_name = "".join(c for c in course_name if c.isalnum())
    filepath = os.path.join(output_dir, f"{safe_course_name}_schedule.ics")

    with open(filepath, 'w') as f:
        f.writelines(cal)

    print(f"TOOL: Calendar file saved to {filepath}")
    return filepath


# --- 3. Main Application Logic (The Agent) ---

# Initialize the Gemini model once
gemini_model = genai.GenerativeModel('gemini-2.0-flash')


def process_syllabus(course_name: str, syllabus_text: str):
    """
    This is the main agent function that orchestrates the entire process.
    """
    # Step 1: Use the AI tool to parse the syllabus.
    extracted_data = extract_deadlines_from_syllabus_tool(syllabus_text, gemini_model)

    if not extracted_data:
        # If the AI tool fails, return an error message and empty outputs
        return "Could not extract any deadlines. The AI might have failed to understand the syllabus format. Please check the console for more details.", None, None

    # Step 2: Present the extracted data for monitoring (Bonus Feature).
    # We use a pandas DataFrame for a clean table display in Gradio.
    df = pd.DataFrame(extracted_data)

    # Step 3: Use the custom tool to generate the calendar file in the background.
    calendar_filepath = generate_ical_file_tool(course_name, extracted_data)

    # Step 4: Return all the results to the UI.
    return "Successfully extracted deadlines! Please review the table below and download your calendar file.", df, calendar_filepath


# --- 4. Gradio User Interface ---

with gr.Blocks(theme=gr.themes.Soft(), title="AI Syllabus Scheduler") as demo:
    gr.Markdown("# AI Syllabus Scheduler")
    gr.Markdown(
        "Paste your course name and syllabus text below. The AI will extract all deadlines and generate a calendar file (`.ics`) for you to import into Google Calendar, Outlook, or Apple Calendar.")

    with gr.Row():
        course_name_input = gr.Textbox(label="Course Name", placeholder="e.g., Introduction to Python")

    syllabus_input = gr.Textbox(label="Syllabus Text", lines=15,
                                placeholder="Paste the entire text from your syllabus here...")

    submit_button = gr.Button("Generate Schedule", variant="primary")

    gr.Markdown("---")
    gr.Markdown("### Results")

    status_output = gr.Textbox(label="Status", interactive=False)

    # UI for monitoring the extracted plan (Bonus Feature)
    schedule_table_output = gr.DataFrame(label="Extracted Deadlines", interactive=False)

    # UI for the final deliverable
    calendar_file_output = gr.File(label="Download Your Calendar File (.ics)")

    submit_button.click(
        fn=process_syllabus,
        inputs=[course_name_input, syllabus_input],
        outputs=[status_output, schedule_table_output, calendar_file_output]
    )

if __name__ == "__main__":
    # Create the output directories if they don't exist
    os.makedirs("calendars", exist_ok=True)
    print("Launching Gradio UI... Access it at http://127.0.0.1:7860")
    demo.launch()