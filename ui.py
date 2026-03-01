
import streamlit as st
st.set_page_config(page_title="Meta-Learning Academic AI", layout="wide", page_icon="🧠")

import requests
import json
from datetime import datetime
import uuid

# API endpoint
API_URL = "http://localhost:8001"

# Custom CSS for ChatGPT/Gemini look
st.markdown("""
<style>
body, html {
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
    background: #18181b;
    color: #e5e7eb;
}
.stApp {
    background: #18181b;
}
.chat-window {
    max-width: 600px;
    margin: 40px auto 0 auto;
    background: #23232a;
    border-radius: 18px;
    box-shadow: 0 4px 32px rgba(0,0,0,0.18);
    padding: 0 0 80px 0;
    min-height: 70vh;
}
.message-bubble {
    display: flex;
    align-items: flex-start;
    margin: 18px 0;
}
.bubble-user {
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);


# --- BACKEND API INTEGRATION ---
import requests
API_BASE = "http://localhost:8001/api"

def get_query_history():
    try:
        return requests.get(f"{API_BASE}/history").json()
    except Exception as e:
        st.error(f"Query history API error: {e}")
        return []

def get_routing_stats():
    try:
        return requests.get(f"{API_BASE}/stats").json()
    except Exception as e:
        st.error(f"Routing stats API error: {e}")
        return {"total":0,"multi_intent":0,"unsafe_blocks":0,"avg_response_ms":0}

def get_model_info():
    try:
        return requests.get(f"{API_BASE}/model_info").json()
    except Exception as e:
        st.error(f"Model info API error: {e}")
        return {"model_version":"-","embedding_model":"-","transformer_model":"-","last_retrained":"-","intent_threshold":0,"unsafe_threshold":0,"system_status":"error"}

def submit_query(query):
    try:
        return requests.post(f"{API_BASE}/query", json={"query": query}).json()
    except Exception as e:
        st.error(f"Query API error: {e}")
        return None

# --- FETCH DYNAMIC DATA ---
model_info = get_model_info()
model_version = model_info.get("model_version", "-")
embedding_model = model_info.get("embedding_model", "-")
transformer_model = model_info.get("transformer_model", "-")
system_status = model_info.get("system_status", "error")
last_retrained = model_info.get("last_retrained", "-")
intent_threshold = model_info.get("intent_threshold", 0)
unsafe_threshold = model_info.get("unsafe_threshold", 0)

query_history = get_query_history()
routing_stats = get_routing_stats()

# --- HEADER ---
with st.container():
    cols = st.columns([1, 3, 2])
    with cols[0]:
        st.image("https://img.icons8.com/fluency/96/brain.png", width=48)
    with cols[1]:
        st.title("Meta-Learning Academic AI")
        st.subheader("Multi-Intent Deterministic Orchestration System")
    with cols[2]:
        status_color = {"ready": "🟢", "retraining": "🟡", "error": "🔴"}[system_status]
        st.markdown(f"**System Status:** {status_color} {system_status.capitalize()}")
        st.markdown(f"**Model Version:** `{model_version}`")
        st.markdown(f"**Embedding Model:** `{embedding_model}`")
        st.markdown(f"**Transformer Model:** `{transformer_model}`")

st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Query History")
    for q in query_history[-10:]:
        chain_icons = " → ".join([f":blue_circle:" if e != "RULE" else ":red_circle:" for e in q["chain"]])
        unsafe_badge = "🚫" if q["unsafe"] else ""
        if st.button(f"{q['query']} {unsafe_badge}", key=q['query']):
            st.session_state.selected_query = q['query']
    st.markdown("---")
    st.header("Routing Statistics")
    st.metric("Total Queries", routing_stats["total"])
    st.metric("Multi-Intent %", f"{routing_stats['multi_intent']}%")
    st.metric("Unsafe Blocks", routing_stats["unsafe_blocks"])
    st.metric("Avg Response Time (ms)", routing_stats["avg_response_ms"])
    st.markdown("---")
    st.header("Model Info")
    st.metric("Intent Threshold", intent_threshold)
    st.metric("Unsafe Threshold", unsafe_threshold)
    st.metric("Last Retrained", last_retrained)

# --- MAIN INPUT SECTION ---

with st.container():
    st.markdown("### Academic Query")
    query = st.text_area("Enter your academic question", value=st.session_state.get("selected_query", ""), key="query_input")
    cols = st.columns([1, 1])
    submit = cols[0].button("🔍 Submit")
    clear = cols[1].button("❌ Clear")
    if clear:
        st.session_state.query_input = ""
        st.session_state.selected_query = ""
        st.session_state.last_result = None
        st.experimental_rerun()
    if submit:
        with st.spinner("Processing..."):
            result_struct = submit_query(query)
            if result_struct:
                st.session_state.last_result = result_struct
                st.session_state.selected_query = query
                st.experimental_rerun()

# --- RESPONSE DISPLAY SECTION ---
if "last_result" in st.session_state:
    result = st.session_state.last_result

    # --- PANEL A: FINAL ANSWER ---
    with st.container():
        st.markdown("## 🧠 Final Answer")
        if "UNSAFE" in result["intents"]["active_intents"]:
            st.error("Query blocked by Rule Engine.")
        elif result["result"]["confidence"] < 0.5:
            st.info("Low confidence result.")
        elif len(result["intents"]["active_intents"]) > 1:
            st.warning("Ambiguous multi-intent routing.")
        else:
            st.success(result["result"]["answer"])
        st.markdown(f"**Explanation:** {result['execution_plan']['chain_reasoning']}")
        if isinstance(result["result"]["answer"], (int, float)):
            st.metric("Numeric Result", result["result"]["answer"])
        st.markdown("---")

    # --- PANEL B: INTENT ANALYSIS ---
    with st.expander("Intent Analysis"):
        scores = result["intents"]["all_scores"]
        active_intents = set(result["intents"].get("active_intents", []))
        # Only show scores for active intents
        filtered_scores = [(intent, score) for intent, score in scores.items() if intent in active_intents]
        if filtered_scores:
            df = pd.DataFrame(filtered_scores, columns=["Intent", "Score"])
            st.dataframe(df.style.highlight_max(axis=0, subset=["Score"], color="lightgreen"), use_container_width=True)
        else:
            st.info("No relevant intent scores for this query.")
        st.markdown(f"**Active Intents:** {', '.join(result['intents']['active_intents'])}")
        st.markdown(f"**Threshold Used:** {result['intents']['threshold_used']}")
        st.markdown(f"**Primary Intent:** {result['intents']['sorted_intents'][0]}")

    # --- PANEL C: EXECUTION PLAN ---
    with st.expander("Execution Plan"):
        chain = result["execution_plan"]["engine_chain"]
        chain_str = " → ".join(chain)
        st.markdown(f"**Engine Chain:** {chain_str}")
        st.markdown(f"**Chain Reasoning:** {result['execution_plan']['chain_reasoning']}")
        st.markdown(f"**Number of Engines Used:** {result['execution_plan']['num_engines']}")

    # --- PANEL D: PERFORMANCE METRICS ---
    with st.expander("Performance Metrics"):
        st.metric("Classification Time (ms)", result["metadata"]["classification_time_ms"])
        st.metric("Confidence", result["result"]["confidence"])
        st.metric("Model Used", model_version)
        st.metric("Timestamp", result["metadata"]["timestamp"])

    # --- PANEL E: SAFETY BLOCK (If Applicable) ---
    if "UNSAFE" in result["intents"]["active_intents"]:
        with st.container():
            st.error("🚨 Query blocked by Rule Engine.")
            st.markdown("**Category:** UNSAFE")
            st.markdown(f"**Confidence:** {scores['UNSAFE']}")
            st.markdown("**Logged:** Yes")

    # --- EXECUTION FLOW VISUALIZATION ---
    st.markdown("### Execution Flow")
    flow_steps = ["Semantic Classification", "Execution Planner"] + chain + ["Validation"]
    flow_cols = st.columns(len(flow_steps))
    for i, step in enumerate(flow_steps):
        with flow_cols[i]:
            st.markdown(f"**{step}**")
            st.markdown(":arrow_down:" if i < len(flow_steps)-1 else "")

    # --- RETRAINING NOTIFICATION ---
    if result["status"] == "ready":
        st.markdown("<span style='color:green; font-size:0.9rem;'>Query stored for automatic retraining.</span>", unsafe_allow_html=True)
    # Optional: Admin mode toggle
    if st.checkbox("Admin Mode"):
        st.markdown("#### Retraining Logs")
        st.info("Model retrained on 2026-02-28. No errors detected.")
    # Minimal sidebar (collapsed by default)
    with st.sidebar:
        st.markdown("<div style='padding: 1rem 0; text-align:center;'><span style='font-size:2rem;'>🧠</span><br><b>Meta-Learning AI</b></div>", unsafe_allow_html=True)
        if st.button("➕ New chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_session = str(uuid.uuid4())
            st.rerun()
        st.markdown("---")
        st.markdown("<small style='color:#8B949E;'>Inspired by ChatGPT & Gemini</small>", unsafe_allow_html=True)

    # Chat header
    st.markdown("<div class='chat-header'>Meta-Learning AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='chat-subtitle'>Advanced AI orchestration. Ask anything!</div>", unsafe_allow_html=True)

    # Example questions (like ChatGPT/Gemini)
    if not st.session_state.messages:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📚 **Factual Query** What is the minimum attendance requirement?", use_container_width=True):
                # Set pending AI query so it is processed immediately
                st.session_state.pending_ai_query = "What is the minimum attendance requirement?"
                st.experimental_rerun()
        with col2:
            if st.button("🔢 **Numeric Calculation** Calculate 25 * 16 + 144", use_container_width=True):
                st.session_state.pending_ai_query = "Calculate 25 * 16 + 144"
                st.experimental_rerun()
        col3, col4 = st.columns(2)
        with col3:
            if st.button("💡 **Explanation Request** Explain how meta-learning works", use_container_width=True):
                st.session_state.pending_ai_query = "Explain how meta-learning works"
                st.experimental_rerun()
        with col4:
            if st.button("🎯 **System Inquiry** What are the benefits of AI orchestration?", use_container_width=True):
                st.session_state.pending_ai_query = "What are the benefits of AI orchestration?"
                st.experimental_rerun()
        st.markdown("<div style='text-align:center; color:#8B949E; margin-top:60px;'>Start a new conversation. Your messages will appear here.</div>", unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                # User messages on the LEFT (avatar then bubble)
                st.markdown(
                    f"<div class='message-bubble' style='justify-content:flex-start;'><div class='avatar avatar-user'>U</div><div class='bubble-user'>{content}</div></div>",
                    unsafe_allow_html=True
                )
            else:
                # AI / system messages on the RIGHT (bubble then avatar)
                st.markdown(
                    f"<div class='message-bubble' style='justify-content:flex-end;'><div class='bubble-ai'>{content}</div><div class='avatar avatar-ai'>AI</div></div>",
                    unsafe_allow_html=True
                )
    st.markdown("</div>", unsafe_allow_html=True)

    # Floating input area - hide while a query is pending/processing
    if "pending_ai_query" not in st.session_state:
        st.markdown("<div class='input-floating'>", unsafe_allow_html=True)
        st.markdown("<div class='input-inner'>", unsafe_allow_html=True)
        user_input = st.text_input("", value=st.session_state.get("input_text", ""), max_chars=2000, placeholder="Type your message and press Enter...", key="input_box", label_visibility="collapsed")
        send_clicked = st.button("Send", key="send_btn", help="Send message", use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Ensure variables exist when input is hidden
        user_input = ""
        send_clicked = False

    # Handle API status
    api_healthy = check_api_health()
    if not api_healthy:
        st.error("🚨 **API Server Offline** - Please start the FastAPI server: `python app.py`")
        st.stop()

    # Handle pending query from example buttons
    pending_query = st.session_state.get("pending_query", "")
    if "pending_query" in st.session_state:
        del st.session_state.pending_query
    if user_input:
        st.session_state.input_text = user_input

    # Process message ONLY when Send button is clicked
    if send_clicked and user_input and user_input.strip():
        st.session_state.messages.append({
            "role": "user",
            "content": user_input.strip()
        })
        st.session_state.pending_ai_query = user_input.strip()
        st.session_state.input_text = ""
        st.rerun()

    # Process AI response if we have a pending query
    if "pending_ai_query" in st.session_state:
        query = st.session_state.pending_ai_query
        del st.session_state.pending_ai_query
        with st.spinner("🤔 Thinking..."):
            result, error = send_query(query)
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
        session_id = st.session_state.current_session
        st.session_state.chat_sessions[session_id] = st.session_state.messages.copy()
        st.rerun()
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

            # Strategy badge (show emoji + pretty label)
            pretty_strategy = strategy.replace("_", " ").title()
            st.markdown(f"**Strategy:** {get_strategy_emoji(strategy)} {pretty_strategy}")
            st.write(answer)

            with st.expander("Orchestration Details"):
                active_str = ", ".join(active_intents) if active_intents else "N/A"
                chain_str = " → ".join(engine_chain) if engine_chain else "N/A"
                st.write(f"**Active Intents:** {active_str}")
                st.write(f"**Execution Chain:** {chain_str}")
                st.write(f"**Confidence:** {confidence:.1%}")

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

                # Retrain Models from Feedback Button
                st.markdown("---")
                if st.button("🧠 Retrain Models from Feedback", help="Retrain domain and engine-selector models using collected user feedback."):
                    import subprocess
                    with st.spinner("Retraining models from feedback..."):
                        try:
                            result = subprocess.run([
                                "python", "training/retrain_from_feedback.py"
                            ], capture_output=True, text=True, timeout=120)
                            if result.returncode == 0:
                                st.success("✅ Models retrained successfully from feedback!")
                                st.text(result.stdout)
                            else:
                                st.error(f"❌ Retraining failed: {result.stderr}")
                        except Exception as e:
                            st.error(f"❌ Error during retraining: {str(e)}")
        
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
