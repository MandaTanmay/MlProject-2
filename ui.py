"""
Meta-Learning AI System - Streamlit UI
User-friendly web interface for interacting with the Meta-Learning AI system.
"""
import streamlit as st
import requests
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Meta-Learning AI System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint
API_URL = "http://localhost:8001"

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .strategy-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-weight: bold;
    }
    .retrieval { background-color: #e3f2fd; color: #1976d2; }
    .ml { background-color: #f3e5f5; color: #7b1fa2; }
    .transformer { background-color: #fff3e0; color: #f57c00; }
    .rule { background-color: #ffebee; color: #c62828; }
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
        return None, "Cannot connect to API. Make sure the FastAPI server is running."
    except Exception as e:
        return None, f"Error: {str(e)}"


def send_feedback(query: str, strategy: str, answer: str, feedback: int, comment: str = ""):
    """Send user feedback to API."""
    try:
        response = requests.post(
            f"{API_URL}/feedback",
            json={
                "query": query,
                "strategy": strategy,
                "answer": answer,
                "feedback": feedback,
                "comment": comment
            },
            timeout=5
        )
        return response.status_code == 200
    except:
        return False


def get_stats():
    """Get system statistics."""
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_strategy_color(strategy: str):
    """Get color class for strategy."""
    colors = {
        "RETRIEVAL": "retrieval",
        "ML": "ml",
        "TRANSFORMER": "transformer",
        "RULE": "rule"
    }
    return colors.get(strategy, "retrieval")


# Main UI
def main():
    # Header
    st.markdown('<div class="main-header">🧠 Meta-Learning AI System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Intelligent AI Orchestration Layer - Not a Chatbot</div>',
        unsafe_allow_html=True
    )
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ **API Server Not Running**")
        st.info("Please start the FastAPI server first:")
        st.code("cd meta_learning_ai\npython app.py", language="bash")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This system **learns which engine should answer a query**, not facts themselves.
        
        **Query Types:**
        - 📚 **Factual** → Retrieval Engine
        - 🔢 **Numeric** → ML Engine
        - 💡 **Explanation** → Transformer Engine
        - 🚫 **Unsafe** → Rule Engine
        """)
        
        st.header("📊 System Stats")
        if st.button("Refresh Stats"):
            stats = get_stats()
            if stats:
                st.json(stats)
            else:
                st.warning("Could not fetch stats")
        
        st.header("🎯 Example Queries")
        examples = {
            "Factual": "What is the minimum attendance requirement?",
            "Numeric": "20 multiplied by 8",
            "Explanation": "Explain meta-learning",
            "Unsafe": "Hack the exam system"
        }
        
        for query_type, example in examples.items():
            if st.button(example, key=f"example_{query_type}"):
                st.session_state.example_query = example
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "example_query" in st.session_state:
        query_input = st.session_state.example_query
        del st.session_state.example_query
    else:
        query_input = ""
    
    # Main query interface
    st.header("💬 Ask Your Query")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "Enter your query:",
            value=query_input,
            placeholder="e.g., What is the minimum attendance requirement?",
            label_visibility="collapsed"
        )
    
    with col2:
        submit_button = st.button("🚀 Submit", type="primary", use_container_width=True)
    
    # Process query
    if submit_button and query:
        with st.spinner("🤔 Processing query..."):
            result, error = send_query(query)
            
            if error:
                st.error(f"❌ {error}")
            elif result:
                # Store in history
                st.session_state.chat_history.insert(0, {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "query": query,
                    "result": result
                })
                
                # Display result
                st.success("✅ Query processed successfully!")
                
                # Strategy badge
                strategy = result["strategy"]
                strategy_class = get_strategy_color(strategy)
                st.markdown(
                    f'<div class="strategy-box {strategy_class}">🎯 Strategy: {strategy}</div>',
                    unsafe_allow_html=True
                )
                
                # Answer
                st.subheader("📝 Answer")
                st.write(result["answer"])
                
                # Metadata
                with st.expander("🔍 Details"):
                    st.write("**Confidence:**", f"{result['confidence']:.2%}")
                    st.write("**Reason:**", result["reason"])
                    
                    if result.get("metadata"):
                        st.write("**Metadata:**")
                        st.json(result["metadata"])
                
                # Feedback
                st.subheader("💬 Was this helpful?")
                col1, col2, col3 = st.columns([1, 1, 3])
                
                with col1:
                    if st.button("👍 Yes", key=f"feedback_pos_{len(st.session_state.chat_history)}"):
                        if send_feedback(query, strategy, result["answer"], 1):
                            st.success("Thanks for your feedback!")
                
                with col2:
                    if st.button("👎 No", key=f"feedback_neg_{len(st.session_state.chat_history)}"):
                        if send_feedback(query, strategy, result["answer"], -1):
                            st.info("Feedback recorded. We'll improve!")
    
    # Chat history
    if st.session_state.chat_history:
        st.header("📜 Query History")
        
        for idx, entry in enumerate(st.session_state.chat_history[:10]):
            with st.expander(f"🕐 {entry['timestamp']} - {entry['query'][:50]}..."):
                st.write("**Query:**", entry["query"])
                st.write("**Strategy:**", entry["result"]["strategy"])
                st.write("**Answer:**", entry["result"]["answer"])
                st.write("**Confidence:**", f"{entry['result']['confidence']:.2%}")
        
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p><strong>Meta-Learning AI System v1.0</strong></p>
        <p>AI Orchestration Layer | Intent-Based Routing | Production-Grade Architecture</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
