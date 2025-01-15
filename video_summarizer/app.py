import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai

from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import tempfile
import os
import time

_ =load_dotenv(find_dotenv())

# Set GOOGLE API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    raise ValueError("The GOOGLE_API_KEY environment variable is not set.")

# Page configuration
st.set_page_config(
    page_title="Multimodal AI Agent Video Summarizer",
    page_icon="📹",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Video Summarizer - Phidata Multimodal AI Agent 📹💬") 
st.header("Summarize a video using AI Powered by Gemeini 2.0 Flash Expressive AI ✏️🌐🔝🚀")

@st.cache_resource
def initialize_agent():
    # Create a new agent
    video_summarizer = Agent(
        name="Video Summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        instructions=["Summarize the video"],
        show_calls=True,
        markdown=True
    )
    
    return video_summarizer

### initialize the agent
multi_modal_agent = initialize_agent()

# file upload
video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv", "webm"],
                              help="Upload a video file to summarize")

if video_file:
    # Save the uploaded file to a temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video_file.name).suffix) as temp_file:
        temp_file.write(video_file.getvalue())
        video_path = temp_file.name
        
    st.video(video_path, start_time=0, format="video/mp4")

    # Print the video path
    st.info(f"Video uploaded: {video_path}")
    
    user_query = st.text_area(
        "What would you like to know about this video?",
        "Summarize the video",
        placeholder="Ask anything about the video content. The AI agent will analyze the video and gather additional insights.",
        help="Provide specific questions or insights that you want from the video.",
    )
    
    if st.button("Analyze Video", key="analyze_video"):
        if not user_query:
            st.warning("Please provide a question or query to analyze the video.")
        else:
            try:
                with st.spinner("Analyzing the video..."):
                    # upload the video to google
                    processed_video = upload_file(video_path)
                    while processed_video.state.name=="PROCESSING":
                        time.sleep(5)
                        processed_video = get_file(processed_video.name)
                        # prmompt generation for analysis
                        analysis_prompt = (
                            f"""
                            Analyze the uploaded video for the conent and context.
                            Respondtot he following questions using the video content and suppplimentary web search: {user_query}
                            
                            Provide a detailed user friendly and actionable respose to the user query.
                            """
                        )
                        # Get the video summary
                        response = multi_modal_agent.run(analysis_prompt,videos=[processed_video])
                
                # Display the response   
                st.subheader("Video analysis report / Summary")
                st.markdown(response.content)
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
            
            finally:
                # clean up the temporary file
                Path(video_path).unlink(missing_ok=True)
else:
    st.info("upload a video file and click the 'Analyze Video' button to get the video summary.")
    
st.markdown(
        """
        <style>
            .instructions {
                background-color: #f9f9f9;
                padding: 10px;
                border-radius: 5px;
            }
            .highlight {
                color: #ff0000;
                font-weight: bold;
            }
            .center {
                text-align: center;
            }
            .small {
                font-size: 0.8em;
            }
            .italic {
                font-style: italic;
            }
            .bold {
                font-weight: bold;
            }
            .underline {
                text-decoration: underline;
            }
            .code {
                font-family: "Courier New", Courier, monospace;
                font-size: 0.9em;
            }
            .note {
                color: #008000;
                font-style: italic;
            }
            .warning {
                color: #ff0000;
                font-weight: bold;
            }
            .info {
                color: #0000ff;
                font-weight: bold;
            }
            .stTextArea textarea {
                height: 100px;
            }       
        </style>""",
        unsafe_allow_html=True
    )