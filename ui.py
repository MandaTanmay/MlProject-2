"""
Meta-Learning AI System - ChatGPT-like Interface
Clean, modern chat interface matching ChatGPT's exact layout and functionality.
"""
import streamlit as st
import requests
import json
from datetime import datetime
import uuid

# Page configuration
st.set_page_config(
    page_title="Meta-Learning AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint
API_URL = "http://localhost:8001"

# Clean ChatGPT-style CSS
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Remove Streamlit branding and padding */
    .stApp > header {visibility: hidden;}
    .stApp > div > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) {visibility: hidden;}
    .stDeployButton {display: none;}
    footer {visibility: hidden;}
    .stActionButton {display: none;}
    
    /* Remove default margins and ensure full height */
    html, body {
        margin: 0;
        padding: 0;
        height: 100vh;
        overflow: hidden;
    }
    
    /* Streamlit main container fixes */
    .main {
        padding: 0 !important;
        height: 100vh;
        overflow: hidden;
    }
    
    /* Fix initial scroll position */
    .stApp {
        scroll-behavior: smooth;
        scroll-padding-top: 0;
    }
    
    /* Ensure content starts at top */
    .main-content {
        scroll-snap-align: start;
    }
    
    /* Main app container */
    .main .block-container {
        padding-top: 0rem !important;
        padding-left: 0rem !important;
        padding-right: 0rem !important;
        padding-bottom: 0rem !important;
        margin-top: 0rem !important;
        max-width: none;
        height: 100vh;
        overflow: hidden;
        position: relative;
    }
    
    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Dark theme - Full viewport */
    .stApp {
        background-color: #0D1117;
        color: #E6EDF3;
        height: 100vh;
        overflow: hidden;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #161B22;
        border-right: 1px solid #30363D;
        height: 100vh;
        overflow-y: auto;
    }
    
    /* Main content area - Full height, scrollable */
    .main-content {
        background-color: #0D1117;
        height: 100vh;
        display: flex;
        flex-direction: column;
        overflow: hidden;
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
    }
    
    /* Messages area - Scrollable */
    .messages-area {
        flex: 1;
        overflow-y: auto;
        padding-bottom: 140px; /* Space for input */
        padding-top: 0rem !important; /* Remove top padding */
        margin-top: 0rem !important; /* Ensure no top margin */
        position: relative;
        top: 0;
    }
    
    /* Welcome/Start screen - Fit in viewport */
    .welcome-container {
        max-width: 768px;
        margin: 0 auto;
        padding: 0.5rem 1rem; /* Minimal top padding */
        text-align: center;
    }
    
    .welcome-title {
        font-size: 1.8rem; /* Slightly smaller */
        font-weight: 600;
        margin-bottom: 0.8rem; /* Reduced margin */
        margin-top: 0 !important; /* No top margin */
        background: linear-gradient(135deg, #58A6FF 0%, #79C0FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
    }
    
    .welcome-subtitle {
        font-size: 0.95rem; /* Slightly smaller */
        color: #8B949E;
        margin-bottom: 1.5rem; /* Reduced margin */
        max-width: 600px;
        line-height: 1.5; /* Tighter line height */
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Example buttons styling */
    .stButton > button {
        background: #161B22 !important;
        color: #E6EDF3 !important;
        border: 1px solid #21262D !important;
        border-radius: 0.75rem !important;
        font-weight: 400 !important;
        transition: all 0.2s !important;
        text-align: left !important;
        padding: 0.8rem !important; /* Reduced padding */
        height: auto !important;
        white-space: normal !important;
        min-height: 55px !important; /* Reduced min height */
        margin-bottom: 0.4rem !important; /* Reduced margin */
    }
    
    .stButton > button:hover {
        background: #21262D !important;
        border-color: #30363D !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
    }
    
    /* Primary buttons (New Chat, Send) */
    .stButton[data-baseweb="button"][kind="primary"] > button {
        background: #238636 !important;
        border-color: #238636 !important;
        color: white !important;
    }
    
    .stButton[data-baseweb="button"][kind="primary"] > button:hover {
        background: #2EA043 !important;
        border-color: #2EA043 !important;
    }
    
    /* Message containers */
    .message-container {
        max-width: 768px;
        margin: 0 auto;
        padding: 1.5rem 1rem;
        border-bottom: 1px solid #21262D;
    }
    
    .user-message {
        background-color: #0D1117;
    }
    
    .ai-message {
        background-color: #0D1117;
    }
    
    .message-header {
        display: flex;
        align-items: center;
        margin-bottom: 0.75rem;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .user-avatar {
        background: #238636;
        color: white;
        width: 28px;
        height: 28px;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 0.75rem;
        font-size: 14px;
    }
    
    .ai-avatar {
        background: #58A6FF;
        color: white;
        width: 28px;
        height: 28px;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 0.75rem;
        font-size: 14px;
    }
    
    .message-content {
        margin-left: 36px;
        color: #E6EDF3;
        font-size: 16px;
        line-height: 1.6;
    }
    
    .strategy-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        background: rgba(88, 166, 255, 0.15);
        color: #58A6FF;
        border-radius: 0.375rem;
        font-size: 0.75rem;
        font-weight: 500;
        margin-bottom: 0.75rem;
        border: 1px solid rgba(88, 166, 255, 0.3);
    }
    
    .metadata-box {
        margin-top: 1rem;
        padding: 0.75rem;
        background: #161B22;
        border: 1px solid #21262D;
        border-radius: 0.5rem;
        font-size: 0.875rem;
    }
    
    .confidence-bar {
        background: #21262D;
        height: 4px;
        border-radius: 2px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        background: #238636;
        height: 100%;
        transition: width 0.3s ease;
    }
    
    /* Input section - Fixed at bottom */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 260px; /* Account for sidebar */
        right: 0;
        background: #0D1117;
        border-top: 1px solid #21262D;
        padding: 0.8rem; /* Reduced padding */
        z-index: 1000;
    }
    
    .input-wrapper {
        max-width: 768px;
        margin: 0 auto;
        position: relative;
    }
    
    /* Responsive input on mobile */
    @media (max-width: 768px) {
        .input-container {
            left: 0; /* Full width on mobile */
        }
        
        .messages-area {
            padding-bottom: 120px; /* Less space on mobile */
        }
    }
    
    /* Sidebar buttons */
    .sidebar-button {
        width: 100%;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        background: #21262D;
        border: 1px solid #30363D;
        border-radius: 0.5rem;
        color: #E6EDF3;
        cursor: pointer;
        transition: all 0.2s;
        font-size: 0.875rem;
        text-align: left;
    }
    
    .sidebar-button:hover {
        background: #30363D;
        border-color: #484F58;
    }
    
    .new-chat-button {
        background: #238636;
        border-color: #238636;
        font-weight: 500;
        text-align: center;
    }
    
    .new-chat-button:hover {
        background: #2EA043;
        border-color: #2EA043;
    }
    
    /* Example prompt buttons */
    .example-button {
        width: 100%;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #161B22;
        border: 1px solid #21262D;
        border-radius: 0.75rem;
        color: #E6EDF3;
        cursor: pointer;
        transition: all 0.2s;
        text-align: left;
    }
    
    .example-button:hover {
        background: #21262D;
        border-color: #30363D;
    }
    
    .example-title {
        font-weight: 500;
        margin-bottom: 0.25rem;
        color: #58A6FF;
    }
    
    .example-text {
        font-size: 0.875rem;
        color: #8B949E;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .message-container {
            padding: 1rem 0.5rem;
        }
        
        .welcome-container {
            padding: 1rem;
        }
        
        .welcome-title {
            font-size: 2rem;
        }
    }
    
    /* Hide specific Streamlit elements */
    .stTextArea > label {
        display: none;
    }
    
    /* Apply custom styling to specific buttons */
    div[data-testid=\"column\"]:nth-child(3) .stButton > button {
        background: linear-gradient(135deg, #58A6FF 0%, #1F6FEB 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 12px 20px !important;
        min-width: 80px !important;
        height: 48px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(88, 166, 255, 0.3) !important;
    }
    
    div[data-testid=\"column\"]:nth-child(3) .stButton > button:hover {
        background: linear-gradient(135deg, #79C0FF 0%, #58A6FF 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(88, 166, 255, 0.4) !important;
    }
    
    div[data-testid=\"column\"]:nth-child(1) .stButton > button {
        background: #21262D !important;
        color: #8B949E !important;
        border: 1px solid #30363D !important;
        border-radius: 12px !important;
        padding: 12px !important;
        width: 48px !important;
        height: 48px !important;
        font-size: 16px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    div[data-testid=\"column\"]:nth-child(1) .stButton > button:hover {
        background: #30363D !important;
        color: #E6EDF3 !important;
        transform: translateY(-1px) !important;
    }
    
    /* Hide form styling to prevent auto-submit */
    .stForm {
        border: none !important;
        background: transparent !important;
    }
    
    /* Custom form layout */
    .custom-input-form {
        display: flex !important;
        gap: 0 !important;
        width: 100% !important;
    }
    
    /* Search Engine Style Input Container */
    .search-container {
        max-width: 768px;
        margin: 0 auto 2rem auto;
        padding: 0 1rem;
        position: relative;
    }
    
    .search-card {
        background: #161B22;
        border: 2px solid #30363D;
        border-radius: 16px;
        padding: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        display: flex;
        align-items: flex-end;
        gap: 8px;
    }
    
    .search-card:focus-within {
        border-color: #58A6FF;
        box-shadow: 0 6px 20px rgba(88, 166, 255, 0.2);
        transform: translateY(-1px);
    }
    
    .search-input-wrapper {
        flex: 1;
        position: relative;
    }
    
    /* Enhanced Text Area Styling */
    .stTextArea > div > div > textarea {
        background: transparent !important;
        border: none !important;
        border-radius: 12px !important;
        color: #E6EDF3 !important;
        font-size: 16px !important;
        font-family: 'Inter', sans-serif !important;
        padding: 16px !important;
        resize: none !important;
        min-height: 24px !important;
        max-height: 120px !important;
        line-height: 1.5 !important;
        box-shadow: none !important;
        outline: none !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }
    
    .stTextArea > div > div > textarea::placeholder {
        color: #8B949E !important;
        font-style: italic !important;
    }
    
    /* Enhanced Send Button */
    .search-send-btn {
        background: linear-gradient(135deg, #58A6FF 0%, #1F6FEB 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 12px 20px !important;
        min-width: 80px !important;
        height: 48px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 6px !important;
        box-shadow: 0 2px 8px rgba(88, 166, 255, 0.3) !important;
    }
    
    .search-send-btn:hover {
        background: linear-gradient(135deg, #79C0FF 0%, #58A6FF 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(88, 166, 255, 0.4) !important;
    }
    
    .search-send-btn:active {
        transform: translateY(0) !important;
    }
    
    /* Upload Button */
    .upload-btn {
        background: #21262D !important;
        color: #8B949E !important;
        border: 1px solid #30363D !important;
        border-radius: 12px !important;
        padding: 12px !important;
        width: 48px !important;
        height: 48px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 18px !important;
    }
    
    .upload-btn:hover {
        background: #30363D !important;
        color: #E6EDF3 !important;
        transform: translateY(-1px) !important;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def send_query(query: str):
    """Send query to API."""
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"query": query},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Error {response.status_code}: {response.text}"
    except requests.exceptions.Timeout:
        return None, "Request timeout. The query took too long to process."
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Make sure the FastAPI server is running on port 8001."
    except Exception as e:
        return None, f"Error: {str(e)}"


def get_strategy_emoji(strategy):
    """Get emoji for strategy."""
    emoji_map = {
        "FACTUAL": "📚", "RETRIEVAL": "📚", 
        "NUMERIC": "🔢", "ML": "🔢",
        "EXPLANATION": "💡", "TRANSFORMER": "💡",
        "UNSAFE": "🚫", "RULE": "🚫"
    }
    return emoji_map.get(strategy, "🎯")


def render_welcome_screen():
    """Render direct chat interface with title and examples."""
    st.markdown("""
        <div style="max-width: 768px; margin: 0 auto; padding: 0rem 1rem 0.8rem; position: relative; top: 0;">
            <div class="welcome-title" style="text-align: center; margin-bottom: 1rem; margin-top: 0; padding-top: 0;">
                Meta-Learning AI System
            </div>
            <div style="text-align: center; color: #8B949E; margin-bottom: 1.5rem; font-size: 0.95rem;">
                Advanced AI orchestration that learns which engine should handle your query
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Example prompts in 2x2 grid - styled like ChatGPT
    st.markdown('<div style="max-width: 768px; margin: 0 auto; padding: 0 1rem;">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        if st.button("📚 **Factual Query**\nWhat is the minimum attendance requirement?", key="ex1", use_container_width=True):
            st.session_state.pending_query = "What is the minimum attendance requirement?"
            st.rerun()
            
        if st.button("💡 **Explanation Request**\nExplain how meta-learning works", key="ex3", use_container_width=True):
            st.session_state.pending_query = "Explain how meta-learning works"
            st.rerun()
    
    with col2:
        if st.button("🔢 **Numeric Calculation**\nCalculate 25 * 16 + 144", key="ex2", use_container_width=True):
            st.session_state.pending_query = "Calculate 25 * 16 + 144"
            st.rerun()
            
        if st.button("🎯 **System Inquiry**\nWhat are the benefits of AI orchestration?", key="ex4", use_container_width=True):
            st.session_state.pending_query = "What are the benefits of AI orchestration?"
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_message(msg, msg_type="user"):
    """Safe render without exposing raw HTML."""
    import html

    if msg_type == "user":
        with st.container():
            st.markdown("### 👤 You")
            st.write(msg)

    else:
        if isinstance(msg, dict):
            strategy = msg.get("strategy", "UNKNOWN")
            confidence = msg.get("confidence", 0.0)
            answer = msg.get("answer", "")
            metadata = msg.get("metadata", {})

            active_intents = metadata.get("active_intents", [])
            engine_chain = metadata.get("engine_chain", [])
            intent_scores = metadata.get("intent_scores", {})
            classification_method = metadata.get("classification_method", "")
            classification_time_ms = metadata.get("classification_time_ms", 0)

            st.markdown("### 🧠 Meta-Learning AI")

            # Strategy badge
            st.markdown(f"**Strategy:** {strategy}")
            st.write(answer)

            with st.expander("Orchestration Details"):
                st.write("**Active Intents:**", ", ".join(active_intents) or "N/A")
                st.write("**Execution Chain:**", " → ".join(engine_chain) or "N/A")
                st.write("**Confidence:**", f"{confidence:.1%}")

                if intent_scores:
                    st.write("**Intent Scores:**")
                    for k, v in intent_scores.items():
                        st.write(f"- {k}: {v:.2f}")

                if classification_method:
                    time_str = f" ({round(classification_time_ms)} ms)" if classification_time_ms else ""
                    st.write(f"**Classifier:** {classification_method}{time_str}")

        else:
            st.markdown("### 🧠 Meta-Learning AI")
            st.write(msg)


def main():
    """Main ChatGPT-like interface."""
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}
    if "current_session" not in st.session_state:
        st.session_state.current_session = str(uuid.uuid4())
    
    # Check API health
    api_healthy = check_api_health()
    
    # Sidebar - ChatGPT style
    with st.sidebar:
        st.markdown('<div style="padding: 0.5rem 0;">', unsafe_allow_html=True)
        
        # New Chat Button
        if st.button("➕ New chat", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.current_session = str(uuid.uuid4())
            if "pending_query" in st.session_state:
                del st.session_state.pending_query
            st.rerun()
        
        st.markdown("---")
        
        # Recent chats (if any)
        if st.session_state.messages:
            st.markdown("**Recent chats**")
            # Show last few user messages as conversation starters
            user_messages = [msg for msg in st.session_state.messages if msg["role"] == "user"]
            for i, msg in enumerate(user_messages[-5:]):  # Last 5 user messages
                preview = msg["content"][:35] + ("..." if len(msg["content"]) > 35 else "")
                if st.button(f"💬 {preview}", key=f"hist_{i}", use_container_width=True):
                    # Could implement chat session loading here
                    pass
            st.markdown("---")
        
        # System Info
        with st.expander("ℹ️ System Info"):
            st.markdown(f"""
            **Model:** Meta-Learning AI v1.0  
            **Status:** {'🟢 Online' if api_healthy else '🔴 Offline'}  
            **Engines:** Retrieval, ML, Transformer, Rule  
            **Session:** {st.session_state.current_session[:8]}...
            """)
        
        with st.expander("📋 Query Types"):
            st.markdown("""
            - **📚 Factual** → Retrieval Engine
            - **🔢 Numeric** → ML Engine  
            - **💡 Explanation** → Transformer Engine
            - **🚫 Unsafe** → Rule Engine
            """)
        
        with st.expander("⚙️ Settings"):
            st.markdown("API Endpoint: `localhost:8001`")
            if st.button("🔄 Refresh API Status"):
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    
    
    # Always show title and examples first, then messages
    if not st.session_state.messages:
        # Show welcome/start interface directly in chat area at top
        st.markdown('<div style="position: absolute; top: 0; left: 0; right: 0; z-index: 1;">', unsafe_allow_html=True)
        render_welcome_screen()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Show title at top even when there are messages
        st.markdown("""
            <div style="max-width: 768px; margin: 0 auto; padding: 1rem; text-align: center; border-bottom: 1px solid #21262D;">
                <div class="welcome-title" style="font-size: 1.5rem; margin-bottom: 0.5rem;">
                    Meta-Learning AI System
                </div>
                <div style="color: #8B949E; font-size: 0.9rem;">
                    AI Orchestration Layer
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Render all messages
        for msg in st.session_state.messages:
            render_message(msg["content"], msg["role"])
    
    
    # Input section (fixed at bottom)
    # Handle API status
    if not api_healthy:
        st.error("🚨 **API Server Offline** - Please start the FastAPI server: `python app.py`")
        st.stop()
    
    # Handle pending query from example buttons
    pending_query = st.session_state.get("pending_query", "")
    if "pending_query" in st.session_state:
        del st.session_state.pending_query
    
    # Initialize input field state
    if "input_text" not in st.session_state:
        st.session_state.input_text = pending_query
    
    # Initialize input counter for clearing
    if "input_counter" not in st.session_state:
        st.session_state.input_counter = 0
    
    # Input container - Clean ChatGPT style 
    st.markdown('<div style="max-width: 768px; margin: 0 auto; padding: 0 1rem; display: flex; gap: 8px; align-items: flex-end;">', unsafe_allow_html=True)
    
    # Input with integrated send button - flex layout
    col1, col2 = st.columns([0.9, 0.1], gap="small")
    
    with col1:
        # Main input field with dynamic key for clearing
        user_input = st.text_area(
            "",
            value=st.session_state.input_text,
            height=68,
            placeholder="Ask me anything about your query...",
            label_visibility="collapsed",
            max_chars=2000,
            key=f"input_field_{st.session_state.input_counter}"
        )
    
    with col2:
        # Send button aligned with input bottom
        st.markdown('<div style="display: flex; align-items: flex-end; height: 20px; padding-bottom: 3px;">', unsafe_allow_html=True)
        send_clicked = st.button("↗", key="send_btn", help="Send message", use_container_width=True, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close input container
    
    # Process message ONLY when Send button is clicked
    if send_clicked and user_input and user_input.strip():
        # Immediately add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input.strip()
        })
        
        # Save current message for processing
        st.session_state.pending_ai_query = user_input.strip()
        
        # Clear input field by incrementing counter (forces new widget)
        st.session_state.input_text = ""
        st.session_state.input_counter += 1
        
        # Immediate refresh to show user message and clear input
        st.rerun()
    
    # Process AI response if we have a pending query
    if "pending_ai_query" in st.session_state:
        query = st.session_state.pending_ai_query
        del st.session_state.pending_ai_query
        
        # Get AI response
        with st.spinner("🤔 Thinking..."):
            result, error = send_query(query)
        
        # Add AI response
        if error:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ **Error:** {error}"
            })
        elif result:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result
            })
        
        # Save to session and refresh
        session_id = st.session_state.current_session
        st.session_state.chat_sessions[session_id] = st.session_state.messages.copy()
        
        st.session_state.chat_sessions[session_id] = st.session_state.messages.copy()
        st.rerun()


if __name__ == "__main__":
    main()
