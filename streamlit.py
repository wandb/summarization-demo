import streamlit as st
import asyncio
import weave
from typing import List
import json
from pathlib import Path
from model import parse_pdf, ChatModel, AuthoringModel, predict
import wandb
from dotenv import load_dotenv
import os

load_dotenv(".env")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
os.environ["WANDB_ENTITY"] = os.getenv("WANDB_ENTITY")

# Initialize Weave
weave.init(f"{os.environ['WANDB_ENTITY']}/summarization-app")

def init_session_state():
    """Initialize session state keys."""
    if "summaries" not in st.session_state:
        st.session_state["summaries"] = []
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = "123abc"

def render_feedback_buttons(call_idx: int):
    """
    Render thumbs up/down and text feedback for the call object at index `call_idx`
    in st.session_state["summaries"].
    """
    col1, col2, col3 = st.columns([1, 1, 4])
    
    # Thumbs up
    with col1:
        if st.button("👍", key=f"thumbs_up_{call_idx}"):
            st.session_state.summaries[call_idx]["call"].feedback.add_reaction("👍")
            st.success("Thanks for the feedback!")
    
    # Thumbs down
    with col2:
        if st.button("👎", key=f"thumbs_down_{call_idx}"):
            st.session_state.summaries[call_idx]["call"].feedback.add_reaction("👎")
            st.success("Thanks for the feedback!")
    
    # Text feedback
    with col3:
        feedback_text = st.text_input("Feedback", key=f"feedback_input_{call_idx}")
        if st.button("Submit Feedback", key=f"submit_feedback_{call_idx}"):
            if feedback_text:
                st.session_state.summaries[call_idx]["call"].feedback.add_note(feedback_text)
                st.success("Feedback submitted!")

def display_summaries():
    """
    Loop through all previously generated summaries in session state
    and display them along with feedback buttons.
    """
    for idx, item in enumerate(st.session_state["summaries"]):
        st.subheader(f"Summary for {item['pdf_name']}")
        st.markdown(item["summary"])
        st.markdown("**Provide Feedback**")
        render_feedback_buttons(idx)
        st.divider()

@weave.op
def process_pdf_files(uploaded_files, chat_model, system_prompt):
    for pdf_file in uploaded_files:
        st.subheader(f"Processing: {pdf_file.name}")

        # Parse text from PDF
        text_content = parse_pdf(pdf_file)
        progress_bar = st.progress(0)

        try:
            # Attach attributes so this call is tracked in Weave
            with weave.attributes({
                "session": st.session_state["session_id"],
                "pdf_name": pdf_file.name,
                "prompt": system_prompt
            }):
                # Call the predict op; returns (summary_text, call_object)
                summary_text, call_obj = predict.call(
                    input=text_content, 
                    chat_model=chat_model, 
                    system_prompt=system_prompt
                )
            
            # Store the summary & call object so we can re-draw them
            st.session_state["summaries"].append({
                "pdf_name": pdf_file.name,
                "summary": summary_text,
                "call": call_obj
            })

            progress_bar.progress(100)
        except Exception as e:
            st.error(f"Error processing {pdf_file.name}: {str(e)}")
        st.divider()

def main():
    st.title("PDF Summarizer for Clinical Trial Data")
    # st.title("Simple PDF Summarizer")
    init_session_state()
    
    # Sidebar: Summarization system prompt
    st.sidebar.header("Prompt")
    default_prompt = (
"""You are a clinical documentation specialist tasked with authoring a detailed and structured clinical report. Use clear and precise medical language, following HL7. Ensure the report is comprehensive yet concise, maintaining a professional and factual tone.
Ensure the report adheres to clinical best practices and is structured logically for easy review by healthcare professionals.
**Clinical Report**

**Patient Name:** [Patient's Name]  
**Age:** [Patient's Age]  
**Diagnosis:** [Primary Diagnosis]  

**Summary:**  
[Brief but structured summary of the patient's condition, treatment, and follow-up plan. Maintain a professional tone and ensure clarity for medical professionals.]
"""  )
    # default_prompt = "You are a helpful assistant that summarizes text clearly and concisely. Focus on capturing the key points, main arguments, and essential context. Avoid repetition or unnecessary details. When summarizing technical, academic, or domain-specific content, preserve important terminology and intent. Use a tone appropriate to the subject matter (e.g., professional for business reports, neutral for news, accessible for general content)."
    system_prompt = st.sidebar.text_area(
        "Enter System Prompt",
        value=default_prompt,
        height=150
    )
    
    # Create the ChatModel instance
    chat_model = ChatModel(
        name="ChatModel",
        chat_model="gpt-4o",
        cm_max_new_tokens=4096,
        cm_temperature=1,
    )
    
    # File uploader for PDF files
    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type="pdf",
        accept_multiple_files=True
    )
    
    # Button to process PDFs
    if uploaded_files and st.button("Process PDFs"):
        process_pdf_files(uploaded_files, chat_model, system_prompt)

    # Display all previously generated summaries + feedback UI
    if st.session_state["summaries"]:
        display_summaries()

if __name__ == "__main__":
    load_dotenv(".env")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
    os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")

    main()