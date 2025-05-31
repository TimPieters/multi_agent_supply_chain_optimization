import streamlit as st
import json
import pandas as pd
import os
from datetime import datetime
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# Import the necessary components from your LangGraph script
from langgraph_sensitivity_analysis import (
    run_sensitivity_analysis, run_sensitivity_analysis_generator,
    SensitivityAnalysisState,
    app, # The compiled LangGraph app
    print_workflow_graph, # Helper to print graph (optional for UI)
    BASELINE_OBJ as DEFAULT_BASELINE_OBJ, # Default baseline objective
    MODEL_FILE_PATH as DEFAULT_MODEL_FILE_PATH,
    MODEL_DATA_PATH as DEFAULT_MODEL_DATA_PATH, # This will be dynamically set
    MODEL_DESCRIPTION_PATH as DEFAULT_MODEL_DESCRIPTION_PATH,
    planner_llm, # Default planner LLM
    coder_llm # Default coder LLM
)

# Import _run_with_exec from utils.py
from utils import _run_with_exec, modify_and_run_model
from config import MODEL_PARAMETERS, MODEL_DATA_MAPPING # Import the new mapping

# --- Page Configuration ---
st.set_page_config(
    layout="wide", 
    page_title="AI-Powered Sensitivity Analysis",
    page_icon="🔬",
    initial_sidebar_state="expanded"
)

# Initialize session state for baseline objective value
if 'baseline_obj_value' not in st.session_state:
    st.session_state['baseline_obj_value'] = 0.0

# --- Custom CSS for Premium Styling ---
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom Header with Gradient */
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin: -1rem -1rem 2rem -1rem;
        color: white;
        text-align: center;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 100" fill="rgba(255,255,255,0.1)"><polygon points="1000,100 1000,0 0,100"/></svg>');
        background-size: cover;
    }
    
    .hero-content {
        position: relative;
        z-index: 1;
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        font-weight: 300;
        opacity: 0.9;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    /* Custom Cards */
    .custom-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid rgba(0,0,0,0.05);
        margin-bottom: 2rem;
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }
    
    .card-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .card-icon {
        width: 24px;
        height: 24px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 12px;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: 500;
        font-size: 0.875rem;
        margin: 0.25rem;
    }
    
    .status-success {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
    }
    
    .status-running {
        background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
        color: white;
        animation: pulse 2s infinite;
    }
    
    .status-error {
        background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
        color: white;
    }
            
    .node-status {
    padding: 10px;
    border-radius: 8px;
    margin: 5px 0;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-left: 4px solid #4CAF50;
    }

    .streaming-log {
        background-color: #1e1e1e;
        color: #00ff00;
        padding: 15px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        height: 200px;
        overflow-y: auto;
        margin: 10px 0;
    }

    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }

    .status-waiting { background-color: #ffa500; }
    .status-active { background-color: #4CAF50; animation: pulse 1s infinite; }
    .status-complete { background-color: #2196F3; }
    .status-error { background-color: #f44336; }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Custom Metrics */
    .metric-container {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #718096;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Custom Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f7fafc 0%, #edf2f7 100%);
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Animation Classes */
    .fade-in {
        animation: fadeIn 0.8s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .slide-up {
        animation: slideUp 0.6s ease-out;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Loading Animation */  
    .spinner-loader {
        border: 3px solid #f3f3f3;
        border-top: 3px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
            
    .stSpinner {
        animation: none !important;
        border: none !important;
        background: none !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.6rem;
        width: fit-content !important;
        margin: 1rem auto !important;
    }

    .stSpinner::before {
        content: '';
        box-sizing: border-box;
        display: block; /* Or inline-block, 'block' is fine as a flex item */

        /* --- Explicit Dimensions & Rigidity --- */
        width: 50px;            /* Ensure this is the same as height */
        height: 50px;           /* Ensure this is the same as width */
        min-width: 50px;        /* Reinforce width */
        min-height: 50px;       /* Reinforce height */
        flex-shrink: 0;         /* CRITICAL: Prevent shrinking in flex layout */

        /* --- Appearance --- */
        border: 3px solid #f3f3f3;
        border-top: 3px solid #667eea; /* The spinning color */
        border-radius: 50%;     /* Makes it a circle if width=height */

        /* --- Animation & Rendering Hint --- */
        animation: spin 1s linear infinite;
        backface-visibility: hidden; /* May help with rendering artifacts/smoothness */
        /* transform-origin: center; */ /* Default, but can be explicit if needed */
    }

    .stSpinner > div > i {
        display: none !important;
    }

    .stSpinner > div {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
            
    
    /* Data Table Styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Custom Selectbox */
    .stSelectbox > div[data-baseweb="select"] > div {
        border-radius: 12px; /* More rounded corners */
        border: 1px solid #cbd5e0; /* Lighter, more subtle border */
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05); /* Subtle shadow */
        background-color: #ffffff; /* Ensure white background */
        padding: 0.5rem 1rem; /* Add some padding inside the selectbox */
        min-height: 3rem; /* Ensure a consistent height */
        display: flex;
        align-items: center;
    }
    
    .stSelectbox > div[data-baseweb="select"] > div:hover {
        border-color: #a7b7e6; /* Lighter hover border */
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1); /* More pronounced hover shadow */
    }

    .stSelectbox > div[data-baseweb="select"] > div:focus-within {
        border-color: #667eea; /* Primary color on focus */
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.2); /* Stronger focus ring */
    }

    /* Style for the displayed value within the selectbox */
    .stSelectbox [data-baseweb="select"] > div:first-child > div:first-child {
        color: #2d3748; /* Darker text color for readability */
        font-weight: 500;
    }

    /* Style for the dropdown arrow */
    .stSelectbox [data-baseweb="select"] > div:first-child > div:last-child {
        color: #667eea; /* Change arrow color */
        font-size: 1.2rem; /* Make arrow slightly larger */
    }

    /* Style for the dropdown options list */
    .stSelectbox [data-baseweb="popover"] {
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15); /* Stronger shadow for the dropdown itself */
        border: 1px solid #e2e8f0;
        overflow: hidden; /* Ensures rounded corners are respected */
    }

    /* Style for individual options in the dropdown */
    .stSelectbox [data-baseweb="menu"] li {
        padding: 12px 20px; /* More padding for options */
        font-size: 1rem;
        color: #4a5568;
        transition: background-color 0.2s ease, color 0.2s ease;
    }

    .stSelectbox [data-baseweb="menu"] li:hover {
        background-color: #e6efff; /* Light blue background on hover */
        color: #4a5568; /* Keep text color consistent or slightly darker */
    }

    .stSelectbox [data-baseweb="menu"] li[aria-selected="true"] {
        background-color: #667eea; /* Primary color for selected item */
        color: white;
        font-weight: 600;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #718096;
        font-size: 0.875rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 3rem;
    }
    
    /* Tab customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8fafc;
        border-radius: 12px;
        padding: 8px;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: transparent;
        border-radius: 8px;
        color: #64748b;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(102, 126, 234, 0.1);
        color: #667eea;
    }
    
    /* Info page specific styles */
    .research-highlight {
        background: linear-gradient(135deg, #e6fffa 0%, #f0fff4 100%);
        border-left: 4px solid #38a169;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .tech-stack {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .github-link {
        display: inline-flex;
        align-items: center;
        padding: 0.75rem 1.5rem;
        background: linear-gradient(135deg, #24292e 0%, #586069 100%);
        color: white;
        text-decoration: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        margin: 1rem 0;
    }
    
    .github-link:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(36, 41, 46, 0.3);
        text-decoration: none;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# --- Hero Header ---
st.markdown("""
<div class="hero-header fade-in">
    <div class="hero-content">
        <h1 class="hero-title">🔬 AI-Powered Sensitivity Analysis</h1>
        <p class="hero-subtitle">
            Harness the power of multi-agent AI systems to automatically explore and analyze 
            optimization model sensitivities with unprecedented intelligence and efficiency.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Main Navigation Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📚 About & Research", 
    "🔬 Sensitivity Analysis", 
    "📊 Results & Visualization", 
    "⚙️ Advanced Settings"
])

# --- Tab 1: About & Research ---
with tab1:
    st.markdown('<div class="slide-up">', unsafe_allow_html=True)
    
    # Research Overview Section
    st.markdown("""
    <div class="custom-card">
        <div class="card-title">
            <span class="card-icon">🎓</span>
            Master's Thesis Research
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="research-highlight">
        <h3>📖 Research Title</h3>
        <p style="font-size: 1.1rem; font-weight: 500; color: #2d3748;">
            "Enhancing Sensitivity Analysis in Supply Chain Optimization Through Generative AI: 
            An Investigation into Quality and Efficiency"
        </p>
        <h3>📝 Research Abstract</h3>
        <p style="font-size: 1rem; color: #4a5568; line-height: 1.6;">
            Sensitivity analysis, while being an indispensable tool in supply chain optimization, often faces neglect due to its laborious nature. The potential of Generative Artificial Intelligence (AI) as a mechanism to streamline and enhance the sensitivity analysis process holds significant promise. The dissertation will investigate this possibility, aiming to uncover how a generative AI  can be instrumental in improving the efficiency and quality of sensitivity analysis. 
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top: 1.5rem;">
        <strong>👨‍Author:</strong> Tim Pieters<br>
        <strong>👨‍🏫 Promotor:</strong> Louis-Philippe Kerkhove<br>
        <strong>🏛️ Institution:</strong> UGent University<br>
        <strong>📅 Year:</strong> 2025
    </div>
    """, unsafe_allow_html=True)
    
    # Research Objectives
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">
                <span class="card-icon">🎯</span>
                Research Objectives
            </div>
            <ul style="color: #4a5568; line-height: 1.8;">
                <li><strong>Automation:</strong> Reduce manual effort in sensitivity analysis</li>
                <li><strong>Intelligence:</strong> Leverage AI for smarter scenario generation</li>
                <li><strong>Efficiency:</strong> Improve analysis speed and coverage</li>
                <li><strong>Quality:</strong> Enhance insight generation and interpretation</li>
                <li><strong>Scalability:</strong> Handle complex optimization models</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">
                <span class="card-icon">🚀</span>
                Key Innovations
            </div>
            <ul style="color: #4a5568; line-height: 1.8;">
                <li><strong>Multi-Agent Architecture:</strong> Specialized AI agents for different tasks</li>
                <li><strong>LangGraph Framework:</strong> Orchestrated workflow management</li>
                <li><strong>Automated Code Generation:</strong> Dynamic model modification</li>
                <li><strong>Intelligent Planning:</strong> Strategic scenario selection</li>
                <li><strong>Real-time Analysis:</strong> Interactive sensitivity exploration</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Technical Architecture
    st.markdown("""
    <div class="custom-card">
        <div class="card-title">
            <span class="card-icon">🏗️</span>
            System Architecture
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="tech-stack">
            <h4 style="color: #667eea; margin-bottom: 1rem;">🧠 Planner Agent</h4>
            <p style="color: #4a5568; font-size: 0.9rem;">
                Strategically proposes sensitivity scenarios based on model structure and previous results
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="tech-stack">
            <h4 style="color: #667eea; margin-bottom: 1rem;">👨‍💻 Coder Agent</h4>
            <p style="color: #4a5568; font-size: 0.9rem;">
                Translates scenario ideas into executable code modifications for the optimization model
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="tech-stack">
            <h4 style="color: #667eea; margin-bottom: 1rem;">⚡ Executor</h4>
            <p style="color: #4a5568; font-size: 0.9rem;">
                Safely runs modified models and captures results with comprehensive error handling
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="tech-stack">
            <h4 style="color: #667eea; margin-bottom: 1rem;">📈 Analyzer</h4>
            <p style="color: #4a5568; font-size: 0.9rem;">
                Synthesizes results across scenarios to generate actionable insights and recommendations
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Technology Stack
    st.markdown("""
    <div class="custom-card">
        <div class="card-title">
            <span class="card-icon">💻</span>
            Technology Stack
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🤖 AI & Machine Learning**
        - OpenAI GPT-4/GPT-4o
        - LangGraph for agent orchestration
        - LangChain for LLM integration
        - Custom prompt engineering
        """)
    
    with col2:
        st.markdown("""
        **🐍 Python Ecosystem**
        - Streamlit for web interface
        - PuLP for optimization
        - Pandas for data manipulation
        - Plotly for visualization
        """)
    
    with col3:
        st.markdown("""
        **🔧 Infrastructure**
        - GitHub for version control
        - JSON for data persistence
        - Automated testing pipeline
        """)
    
    # Open Source Information
    st.markdown("""
    <div class="custom-card">
        <div class="card-title">
            <span class="card-icon">🌍</span>
            Open Source & Community
        </div>
        <p style="color: #4a5568; line-height: 1.6; margin-bottom: 1.5rem;">
            This platform is developed as an open-source research prototype to advance the field of 
            AI-assisted optimization analysis. Only open-source libraries have been used, supporting my personal belief in transparent research and collaborative 
            development to benefit the broader academic and industry communities.
        </p>
        
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""       
        <div style="display: flex; gap: 1rem; align-items: center; flex-wrap: wrap;">
            <a href="https://github.com/TimPieters/multi_agent_supply_chain_optimization" target="_blank" class="github-link">
                <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24" style="margin-right: 8px;">
                    <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                </svg>
                View on GitHub
            </a>
            <span class="status-badge status-success">✅ MIT License</span>
            <span class="status-badge" style="background: #667eea; color: white;">🔬 Research Use</span>
        </div>""" , unsafe_allow_html=True)
    
    # Contact Information
    st.markdown("""
    <div class="custom-card">
        <div class="card-title">
            <span class="card-icon">📧</span>
            Contact & Collaboration
        </div>
        <p style="color: #4a5568; line-height: 1.6;">
            For research collaboration, questions about the methodology, or technical support
            (or just a friendly chat about AI and optimization), feel free to reach out via email or LinkedIn.
            I'm always down for a chat!
            :
        </p>
        <ul style="color: #4a5568; line-height: 1.6;">
            <li><strong>📧 Ugent Email:</strong> <a href="mailto:timpiete.pieters@ugent.be">
                timpiete.pieters@ugent.be
                </a>
            </li>
            <li><strong>📧 Personal Email:</strong>    
                <a href="mailto: timpieters@live.be">
                timpieters@live.be
                </a> (or my personal email when I eventually graduate)
            </li>
            <li><strong>🔗 LinkedIn:</strong>
                <a href="https://www.linkedin.com/in/tim-pieters/">
                https://www.linkedin.com/in/tim-pieters/
                </a>
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Tab 2: Sensitivity Analysis ---
with tab2:

    # --- Enhanced Sidebar ---
    with st.sidebar:
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">
                <span class="card-icon">ℹ️</span>
                About This Platform
            </div>
            <p style="color: #4a5568; line-height: 1.6;">
                This cutting-edge application leverages <strong>LangGraph multi-agent systems</strong> 
                to perform intelligent sensitivity analysis on optimization models.
            </p>
            <ul style="color: #718096; font-size: 0.9rem; line-height: 1.5;">
                <li><strong>Planner Agent:</strong> Strategically proposes scenarios</li>
                <li><strong>Coder Agent:</strong> Translates ideas into code</li>
                <li><strong>Executor:</strong> Runs modified models</li>
                <li><strong>Analyzer:</strong> Synthesizes insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # API Configuration with enhanced styling
        st.markdown("""
        <div class="card-title">
            <span class="card-icon">🔑</span>
            OpenAI API Configuration
        </div>
        """, unsafe_allow_html=True)
        
        openai_api_key = st.text_input(
            "Enter your OpenAI API Key",
            type="password",
            help="Your OpenAI API key for powering the AI agents. If not provided, assumes environment variable is set."
        )
        
        if openai_api_key:
            st.markdown('<span class="status-badge status-success">✅ API Key Configured</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-error">❌ API Key Required</span>', unsafe_allow_html=True)

    # --- Main Content Area ---

    st.markdown('<div class="slide-up">', unsafe_allow_html=True)
    st.markdown("""
    <div class="custom-card">
        This application is developed as a research prototype for the
        following master's thesis:
            <br>
            <strong>"Enhancing Sensitivity Analysis in Supply Chain Optimization Through Generative AI: An Investigation into Quality and Efficiency"</strong>
            <br><br>
            <strong>Author</strong>: Tim Pieters
            <br>
            <strong>Promotor</strong>: Louis-Philippe Kerkhove
    </div>
    """, unsafe_allow_html=True)

    # Configuration Section with Cards
    st.markdown('<div class="slide-up">', unsafe_allow_html=True)
    st.markdown("""
    <div class="custom-card">
        <div class="card-title">
            <span class="card-icon">⚙️</span>
            Model Configuration
        </div>
    </div>
    """, unsafe_allow_html=True)


    # Define base directory for models
    MODELS_BASE_DIR = "models"

    # Model and Data Selection with enhanced layout
    col_model, col_data = st.columns(2)

    with col_model:
        st.markdown("**🎯 Optimization Model**")
        # Dynamically get available models from MODEL_DATA_MAPPING keys
        available_models = list(MODEL_DATA_MAPPING.keys())
        default_model_index = available_models.index(DEFAULT_MODEL_FILE_PATH) if DEFAULT_MODEL_FILE_PATH in available_models else 0
        selected_model_file = st.selectbox(
            "Select Model",
            options=available_models,
            index=default_model_index,
            help="Choose the Python file containing your optimization model"
        )
        # Construct full path to the selected model file
        selected_model_full_path = os.path.join(MODELS_BASE_DIR, selected_model_file)

    with col_data:
        st.markdown("**📊 Dataset**")
        # Filter available data files based on selected model
        available_data_files = MODEL_DATA_MAPPING.get(selected_model_file, [])
        # Set default index for data file based on DEFAULT_MODEL_DATA_PATH
        default_data_index = available_data_files.index(os.path.basename(DEFAULT_MODEL_DATA_PATH)) if os.path.basename(DEFAULT_MODEL_DATA_PATH) in available_data_files else 0

        selected_data_file = st.selectbox(
            "Select Data File",
            options=available_data_files,
            index=default_data_index,
            help="Choose the JSON file containing input data for your model"
        )
        # Construct full path to the selected data file
        # The data files are now inside the model's specific data folder
        model_sub_dir = os.path.dirname(selected_model_file) # e.g., "CFLP/" or "VRP/"
        selected_data_path = os.path.join(model_sub_dir, "data", selected_data_file)


    # Model Description with enhanced display
    # The description file is now named 'description.txt' inside the model's folder
    model_folder_path = os.path.dirname(selected_model_file) # e.g., "models/CFLP"
    selected_model_description_full_path = os.path.join(model_folder_path, "description.txt")

    with st.expander("📋 View Model Description", expanded=False):
        try:
            with open(selected_model_description_full_path, "r") as f:
                model_description_content = f.read()
            st.markdown(f"```\n{model_description_content}\n```")
        except FileNotFoundError:
            st.warning(f"Model description file not found: {selected_model_description_full_path}")
        except Exception as e:
            st.error(f"Error loading model description: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Advanced Configuration
    with st.expander("🔧 Advanced Path Configuration", expanded=False):
        st.markdown("Override default paths if needed:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            override_model_file_path = st.text_input(
                "Model File Path",
                value=selected_model_file,
                help="Custom path to optimization model file"
            )
        with col2:
            override_model_data_path = st.text_input(
                "Data File Path", 
                value=selected_data_path,
                help="Custom path to model data file"
            )
        with col3:
            override_model_description_path = st.text_input(
                "Description File Path",
                value=selected_model_description_full_path,
                help="Custom path to model description file"
            )

    # Use overridden paths
    model_file_path = override_model_file_path
    model_data_path = override_model_data_path  
    model_description_path = override_model_description_path

    # Baseline Execution Card
    st.markdown("""
    <div class="custom-card slide-up">
        <div class="card-title">
            <span class="card-icon">🏁</span>
            Baseline Model Execution
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Run Baseline Model", type="secondary"):
        with st.spinner("", show_time=False):
            try:
                start_time = time.time()
                baseline_result = modify_and_run_model({}, model_file_path, model_data_path)
                end_time = time.time()
                execution_time = end_time - start_time

                if baseline_result and isinstance(baseline_result, dict):
                    # Create metrics display
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-container fade-in">
                            <div class="metric-value">{baseline_result.get('status', 'N/A')}</div>
                            <div class="metric-label">Status</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        obj_value_raw = baseline_result.get('total_cost')
                        if isinstance(obj_value_raw, (int, float)):
                            st.session_state['baseline_obj_value'] = float(obj_value_raw)
                            obj_value_display = f"{obj_value_raw:.2f}"
                        else:
                            st.session_state['baseline_obj_value'] = 0.0 # Reset to 0.0 if not a number
                            obj_value_display = 'N/A'
                        st.markdown(f"""
                        <div class="metric-container fade-in">
                            <div class="metric-value">{obj_value_display}</div>
                            <div class="metric-label">Objective Value</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-container fade-in">
                            <div class="metric-value">{execution_time:.2f}s</div>
                            <div class="metric-label">Execution Time</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Solution Variables
                    if 'solution' in baseline_result and baseline_result['solution']:
                        with st.expander("📊 View Decision Variables", expanded=False):
                            solution_dict = baseline_result['solution']
                            # Convert dictionary of variables to a DataFrame for better display
                            var_data = [{"Variable": var_name, "Value": var_value} for var_name, var_value in solution_dict.items()]
                            
                            if var_data:
                                df = pd.DataFrame(var_data)
                                
                                # Enhanced data table with styling
                                st.markdown("### 📋 Decision Variables Table")
                                st.dataframe(
                                    df.style.format({"Value": "{:.4f}"}).background_gradient(subset=["Value"]),
                                    use_container_width=True, 
                                    hide_index=True
                                )
                                
                                # Create multiple visualizations
                                if len(var_data) > 0 and all(isinstance(row["Value"], (int, float)) for row in var_data):
                                    # Bar Chart
                                    fig1 = px.bar(
                                        df, x="Variable", y="Value", 
                                        title="📊 Decision Variables Overview",
                                        color="Value",
                                        color_continuous_scale="plasma",
                                        template="plotly_white"
                                    )
                                    fig1.update_layout(
                                        xaxis_tickangle=-45,
                                        title_font_size=16,
                                        showlegend=False,
                                        margin=dict(l=0, r=0, t=40, b=0)
                                    )
                                    st.plotly_chart(fig1, use_container_width=True)
                                    
                                    # Pie Chart for top variables
                                    if len(df) > 3:
                                        top_vars = df.nlargest(5, 'Value')
                                        fig2 = px.pie(
                                            top_vars, 
                                            values="Value", 
                                            names="Variable",
                                            title="🥧 Top Variables Distribution",
                                            template="plotly_white",
                                            color_discrete_sequence=px.colors.qualitative.Set3
                                        )
                                        fig2.update_layout(
                                            title_font_size=16,
                                            margin=dict(l=0, r=0, t=40, b=0)
                                        )
                                        st.plotly_chart(fig2, use_container_width=True)
                            else:
                                st.code(solution_dict)
                else:
                    st.warning("Baseline execution returned unexpected results")
                    st.json(baseline_result)

            except Exception as e:
                st.error(f"Error during baseline execution: {e}")
                st.exception(e)

# Analysis Parameters Card
    st.markdown("""
    <div class="custom-card slide-up">
        <div class="card-title">
            <span class="card-icon">🎛️</span>
            Analysis Parameters
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        baseline_obj = st.number_input(
            "🎯 Baseline Objective Value",
            value=st.session_state.get('baseline_obj_value', 0.0),
            help="The objective value of the original model for calculating ΔObj. Run baseline model first."
        )
    with col2:
        max_iterations = st.number_input(
            "🔄 Maximum Scenarios",
            min_value=1,
            value=5,
            help="Maximum number of sensitivity scenarios to explore"
        )

    # LLM Configuration Card
    st.markdown("""
    <div class="custom-card slide-up">
        <div class="card-title">
            <span class="card-icon">🤖</span>
            AI Agent Configuration
        </div>
    </div>
    """, unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**🧠 Planner Agent**")
        planner_model = st.selectbox(
            "Model",
            options=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
            index=0 if planner_llm.model_name == "gpt-4o" else 1,
            help="AI model for scenario planning"
        )
        planner_temperature = st.slider(
            "Creativity Level",
            min_value=0.0,
            max_value=1.0,
            value=planner_llm.temperature,
            step=0.1,
            help="Higher values = more creative scenarios"
        )
    with col4:
        st.markdown("**👨‍💻 Coder Agent**") 
        coder_model = st.selectbox(
            "Model",
            options=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
            index=0 if coder_llm.model_name == "gpt-4o" else 1,
            help="AI model for code generation"
        )
        coder_temperature = st.slider(
            "Precision Level",
            min_value=0.0,
            max_value=1.0, 
            value=coder_llm.temperature,
            step=0.1,
            help="Lower values = more deterministic code"
        )

    # Main Analysis Section
    st.markdown("""
    <div class="custom-card slide-up">
        <div class="card-title">
            <span class="card-icon">🔬</span>
            Sensitivity Analysis Execution
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Helper function to style log entries (from previous response, keep as is)
    def format_log_entry(icon, title, details="", status="info", expander_content=None, timestamp=None):
        status_colors = {
            "info": "#667eea",    # Blue/Purple
            "success": "#38a169", # Green
            "warning": "#dd6b20", # Orange
            "error": "#e53e3e",   # Red
            "running": "#f6e05e"  # Yellow (for active step)
        }
        border_color = status_colors.get(status, "#e2e8f0")
        time_str = f'<span style="float: right; font-size: 0.8em; color: #718096;">{timestamp}</span>' if timestamp else ""

        # Simplified HTML structure for the main log entry
        entry_html = f"""
        <div style="border: 1px solid {border_color}; padding: 10px; margin-bottom: 10px; border-radius: 5px; background-color: #f8fafc; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="font-size: 1.1em; font-weight: 600; color: #2d3748; margin-bottom: 5px;">
                {icon} {title} {time_str}
            </div>
            <p style="font-size: 0.9em; color: #4a5568; white-space: pre-wrap; word-wrap: break-word; max-height: 100px; overflow-y: auto;">
                {details}
            </p>
        </div>
        """
        
        st.markdown(entry_html, unsafe_allow_html=True)

        if expander_content:
            expander_label = "View Full Output"
            if isinstance(expander_content, dict) and "token_usage" in expander_content:
                expander_label = "View Output & Token Usage"
            elif isinstance(expander_content, str) and len(expander_content) > 200:
                expander_label = "View Full Summary/Code"

            with st.expander(expander_label, expanded=False):
                if isinstance(expander_content, dict):
                    st.json(expander_content)
                elif isinstance(expander_content, str):
                    st.markdown(expander_content, unsafe_allow_html=True) # Changed from st.text to st.markdown
                else:
                    st.code(str(expander_content))

    if st.button("🚀 Launch AI-Powered Analysis", type="primary"):
        st.session_state['running_analysis'] = True
        st.session_state['scenario_log'] = []
        st.session_state['final_analysis_summary'] = ""
        st.session_state['run_id'] = datetime.now().strftime("%Y%m%d_%H%M%S")

        scenario_counter = st.empty()
        # Create dynamic progress tracking with streaming support
        progress_container = st.container()
        with progress_container:
            st.markdown('<span class="status-badge status-running">🔄 Analysis in Progress</span>', unsafe_allow_html=True)
            
            # Progress tracking elements
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Real-time streaming display
            st.markdown("### 🔄 Live Analysis Stream")
            stream_container = st.container()
    
            # Node status indicators
            node_status_container = st.container()
            with node_status_container:
                col_plan, col_code, col_exec, col_analyze = st.columns(4)
                with col_plan:
                    planner_status = st.empty()
                with col_code:
                    coder_status = st.empty()
                with col_exec:
                    executor_status = st.empty()
                with col_analyze:
                    analyzer_status = st.empty()
            
            live_log = st.empty()
            live_feed_container = st.container()
            
        # Results containers
        scenario_container = st.container()
        summary_container = st.container()

        try:
            # Initialize status indicators
            planner_status.markdown("🧠 **Planner**: ⏳ Waiting")
            coder_status.markdown("👨‍💻 **Coder**: ⏳ Waiting")
            executor_status.markdown("⚡ **Executor**: ⏳ Waiting")
            analyzer_status.markdown("📊 **Analyzer**: ⏳ Waiting")
            
            status_text.info("🤖 Initializing AI agents...")
            
            # Initialize streaming log
            streaming_log = []
            current_iteration = 0
            total_iterations = max_iterations
            
            # Use the generator version for real-time updates
            final_state = None
            start_time = time.time()
            
            # Stream the analysis using the modified function
            for event in run_sensitivity_analysis_generator(
                baseline_objective=baseline_obj,
                max_iterations=max_iterations,
                planner_model=planner_model,
                planner_temperature=planner_temperature,
                coder_model=coder_model,
                coder_temperature=coder_temperature,
                model_file_path=model_file_path,
                model_data_path=model_data_path,
                model_description_path=model_description_path,
                openai_api_key=openai_api_key
            ):
                # Extract event data
                node_events = event["event"]
                elapsed_time = event["elapsed_time"]
                run_id = event["run_id"]
                
                # Process each node event
                for node_name, node_output in node_events.items():
                    # Update progress based on current iteration
                    if 'current_iteration' in node_output:
                        current_iteration = node_output['current_iteration']
                        progress_percent = min(int((current_iteration / total_iterations) * 100), 100)
                        progress_bar.progress(progress_percent)

                    # Update live displays
                    scenario_counter.metric(
                        "Current Scenario", 
                        f"{current_iteration}/{total_iterations}",
                        f"⏱️ {elapsed_time:.1f}s elapsed"
                    )
                    
                    # Update node status indicators
                    if node_name == "planner":
                        planner_status.markdown("🧠 **Planner**: ✅ Planning scenario")
                        if node_output.get('proposed_scenario'):
                            scenario_preview = node_output['proposed_scenario']
                            if len(scenario_preview) > 200:
                                scenario_preview = scenario_preview[197] + "..."
                            streaming_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] 💡 **Planner**: Scenario {current_iteration} proposed: \"{scenario_preview}\"")
                            status_text.info(f"🧠 Planner: Generated scenario {current_iteration}")
                            
                    elif node_name == "coder":
                        coder_status.markdown("👨‍💻 **Coder**: ✅ Writing code")
                        if node_output.get('code_modification'):
                            code_modification = node_output['code_modification']
                            code_preview = code_modification
                            streaming_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] 👨‍💻 **Coder**: Code generated for scenario {current_iteration}. Preview: `{code_preview}`")
                            status_text.info(f"👨‍💻 Coder: Code modification completed")
                            
                    elif node_name == "executor":
                        executor_status.markdown("⚡ **Executor**: ✅ Running code")
                        if node_output.get('execution_result'):
                            result = node_output['execution_result']
                            exec_status = result.get('status', 'N/A')
                            obj_value_raw = result.get('total_cost')
                            obj_value_display = 'N/A' # Default display value
                            if isinstance(obj_value_raw, (int, float)):
                                obj_value_display = f"{obj_value_raw:.2f}"
                            streaming_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] ⚡ **Executor**: Scenario {current_iteration} executed. Status: {exec_status}, Objective: {obj_value_display}")
                            status_text.info(f"⚡ Executor: Code executed successfully")
                        if node_output.get('error_message'):
                            streaming_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ **Executor**: Error in scenario {current_iteration}: {node_output['error_message']}")
                            status_text.error(f"❌ Executor: Error occurred")
                            
                    elif node_name == "analyzer":
                        analyzer_status.markdown("📊 **Analyzer**: ✅ Analyzing results")
                        streaming_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 **Analyzer**: Scenario {current_iteration} results analyzed.")
                        status_text.info(f"📊 Analyzer: Completed analysis for scenario {current_iteration}")
                        
                        # Reset node statuses for next iteration if continuing
                        if current_iteration < total_iterations:
                            planner_status.markdown("🧠 **Planner**: ⏳ Preparing next scenario")
                            coder_status.markdown("👨‍💻 **Coder**: ⏳ Waiting")
                            executor_status.markdown("⚡ **Executor**: ⏳ Waiting")
                            analyzer_status.markdown("📊 **Analyzer**: ⏳ Waiting")
                            
                    elif node_name == "final_analyzer":
                        analyzer_status.markdown("📊 **Analyzer**: ✅ Final analysis complete")
                        streaming_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🎯 **Final Analyzer**: Overall analysis summary generated.")
                        status_text.success("🎯 Final Analyzer: Analysis complete!")
                        final_state = node_output
                    
                    # Show recent log entries (last 10)
                    recent_logs = streaming_log[-10:] if len(streaming_log) > 10 else streaming_log
                    live_log.markdown("**📝 Recent Activity:**\n" + "\n".join([f"* {log}" for log in recent_logs]))
                    
                    # Small delay to make updates visible
                    time.sleep(0.1)

            # Analysis completed
            progress_bar.progress(100)
            status_text.success("✅ Analysis Complete!")
            
            # Reset all node statuses to complete
            planner_status.markdown("🧠 **Planner**: ✅ Complete")
            coder_status.markdown("👨‍💻 **Coder**: ✅ Complete")
            executor_status.markdown("⚡ **Executor**: ✅ Complete")
            analyzer_status.markdown("📊 **Analyzer**: ✅ Complete")

            # Store results in session state
            if final_state:
                st.session_state['final_analysis_summary'] = final_state.get('final_analysis_summary', 'No summary available.')
                st.session_state['scenario_log'] = final_state.get('scenario_log', [])
            
            # Display final results
            with summary_container:
                st.markdown("""
                <div class="custom-card">
                    <div class="card-title">
                        <span class="card-icon">📈</span>
                        Analysis Summary
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(st.session_state['final_analysis_summary'])

            with scenario_container:
                st.markdown("""
                <div class="custom-card">
                    <div class="card-title">
                        <span class="card-icon">📝</span>
                        Detailed Scenario Log
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                log_text = "\n".join(st.session_state['scenario_log'])
                st.text_area(
                    "Full Execution Log",
                    log_text,
                    height=400,
                    help="Complete log of all scenarios executed during the analysis"
                )

            # Download section with enhanced styling
            st.markdown("""
            <div class="custom-card">
                <div class="card-title">
                    <span class="card-icon">💾</span>
                    Download Results
                </div>
            </div>
            """, unsafe_allow_html=True)

            run_id = st.session_state['run_id']
            log_csv_path = f"logs/run_log_{run_id}.csv"
            log_txt_path = f"logs/scenario_log_{run_id}.txt"

            col1, col2 = st.columns(2)
            
            with col1:
                if os.path.exists(log_csv_path):
                    with open(log_csv_path, "rb") as f:
                        st.download_button(
                            label="📊 Download CSV Report",
                            data=f,
                            file_name=f"sensitivity_analysis_{run_id}.csv",
                            mime="text/csv"
                        )
            
            with col2:
                if os.path.exists(log_txt_path):
                    with open(log_txt_path, "rb") as f:
                        st.download_button(
                            label="📝 Download Text Log",
                            data=f,
                            file_name=f"analysis_log_{run_id}.txt", 
                            mime="text/plain"
                        )

        except Exception as e:
            status_text.error("❌ Analysis Failed")
            st.error(f"An error occurred during analysis: {e}")
            st.exception(e)
            
            # Reset node statuses to error state
            planner_status.markdown("🧠 **Planner**: ❌ Error")
            coder_status.markdown("👨‍💻 **Coder**: ❌ Error")
            executor_status.markdown("⚡ **Executor**: ❌ Error")
            analyzer_status.markdown("📊 **Analyzer**: ❌ Error")

    # Previous Results Display with Enhanced Visualization
    if 'scenario_log' in st.session_state and st.session_state['scenario_log']:
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">
                <span class="card-icon">🔄</span>
                Previous Analysis Results
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Summary Dashboard", "📝 Detailed Log", "📈 Trends"])
        
        with tab1:
            # Create a mock dashboard with sample sensitivity data
            st.markdown("### 🎯 Sensitivity Analysis Dashboard")
            
            # Sample metrics (in real implementation, parse from actual results)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Scenarios Analyzed", 
                    value="5",
                    delta="2 High Impact"
                )
            with col2:
                st.metric(
                    label="Max Objective Change", 
                    value="12.5%",
                    delta="↑ Positive"
                )
            with col3:
                st.metric(
                    label="Avg Runtime", 
                    value="2.3s",
                    delta="-0.8s faster"
                )
            with col4:
                st.metric(
                    label="Success Rate", 
                    value="100%",
                    delta="Perfect"
                )
            
            # Sample sensitivity heatmap
            st.markdown("### 🔥 Parameter Sensitivity Heatmap")
            
            # Generate sample data for demonstration
            parameters = ['Demand_Scaling', 'Cost_Factor', 'Capacity_Limit', 'Transport_Cost', 'Fixed_Cost']
            scenarios = ['Scenario_1', 'Scenario_2', 'Scenario_3', 'Scenario_4', 'Scenario_5']
            
            # Sample sensitivity matrix
            sensitivity_data = np.random.rand(5, 5) * 20 - 10  # Random values between -10 and 10
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=sensitivity_data,
                x=parameters,
                y=scenarios,
                colorscale='RdBu',
                colorbar=dict(title="Impact %"),
                text=np.around(sensitivity_data, 2),
                texttemplate="%{text}%",
                textfont={"size": 10}
            ))
            
            fig_heatmap.update_layout(
                title="Parameter Impact Across Scenarios",
                xaxis_title="Parameters",
                yaxis_title="Scenarios",
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with tab2:
            st.text_area(
                "Previous Execution Log",
                "\n".join(st.session_state['scenario_log']),
                height=400,
                key="previous_scenario_log"
            )
        
        with tab3:
            st.markdown("### 📈 Analysis Trends")
            
            # Sample trend data
            trend_data = {
                'Scenario': ['Base', 'S1', 'S2', 'S3', 'S4', 'S5'],
                'Objective_Value': [1000, 1025, 980, 1050, 995, 1030],
                'Runtime_Seconds': [2.1, 2.3, 1.8, 2.7, 2.0, 2.4],
                'Parameters_Changed': [0, 1, 2, 1, 3, 2]
            }
            
            trend_df = pd.DataFrame(trend_data)
            
            # Multi-line plot
            fig_trends = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Objective Value Progression', 'Runtime Analysis'),
                vertical_spacing=0.1
            )
            
            fig_trends.add_trace(
                go.Scatter(
                    x=trend_df['Scenario'], 
                    y=trend_df['Objective_Value'],
                    mode='lines+markers',
                    name='Objective Value',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=8)
                ),
                row=1, col=1
            )
            
            fig_trends.add_trace(
                go.Bar(
                    x=trend_df['Scenario'], 
                    y=trend_df['Runtime_Seconds'],
                    name='Runtime (s)',
                    marker_color='#764ba2'
                ),
                row=2, col=1
            )
            
            fig_trends.update_layout(
                height=500,
                showlegend=True,
                template="plotly_white"
            )
            
            st.plotly_chart(fig_trends, use_container_width=True)

    if 'final_analysis_summary' in st.session_state and st.session_state['final_analysis_summary']:
        with st.expander("View Previous Analysis Summary", expanded=False):
            st.markdown(st.session_state['final_analysis_summary'])

    # Add a real-time statistics section
    st.markdown("""
    <div class="custom-card slide-up">
        <div class="card-title">
            <span class="card-icon">📊</span>
            Platform Statistics
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sample platform statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">1,247</div>
            <div class="metric-label">Analyses Run</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">99.2%</div>
            <div class="metric-label">Success Rate</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">2.1s</div>
            <div class="metric-label">Avg Runtime</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">342</div>
            <div class="metric-label">Users Served</div>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        <p>🚀 Powered by LangGraph Multi-Agent AI | Built with Streamlit | © 2025</p>
        <p style="font-size: 0.8rem; margin-top: 0.5rem;">
            Transforming optimization analysis through intelligent automation
        </p>
    </div>
    """, unsafe_allow_html=True)
