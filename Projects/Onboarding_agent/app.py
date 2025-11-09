import streamlit as st
import autogen
from autogen import AssistantAgent, UserProxyAgent
import os
import json
from datetime import datetime
import asyncio
from typing import Dict, List
import groq

# Set page configuration
st.set_page_config(
    page_title="AI Onboarding Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #ff4b4b;
    }
    .assistant-message {
        background-color: #e6f3ff;
        border-left-color: #1f77b4;
    }
    .system-message {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }
    .message-header {
        font-weight: bold;
        margin-bottom: 5px;
        color: #333;
    }
    .message-content {
        color: #555;
        line-height: 1.5;
    }
    .stButton button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
    .agent-status {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        background-color: #f8f9fa;
        border-left: 3px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

class GroqClient:
    def __init__(self, api_key: str):
        self.client = groq.Groq(api_key=api_key)
    
    def get_completion(self, message: str) -> str:
        """Get completion from Groq API"""
        try:
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert onboarding specialist. Your role is to guide new users through the onboarding process.
                        You help with:
                        - Account setup and configuration
                        - Platform navigation and features
                        - Team introductions and collaboration
                        - Best practices and tips
                        - Troubleshooting common issues
                        
                        Be friendly, patient, and thorough in your explanations. Ask clarifying questions when needed.
                        Provide step-by-step guidance and check for understanding.
                        Always maintain a professional yet approachable tone."""
                    },
                    {
                        "role": "user",
                        "content": message
                    }
                ],
                temperature=0.7,
                max_tokens=1024,
                top_p=1,
                stream=False
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Please try again."

class OnboardingAIAgent:
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.groq_client = GroqClient(groq_api_key)
        
        # Configure for AutoGen with correct base URL
        self.config_list = [
            {
                "model": "llama-3.3-70b-versatile", 
                "api_key": groq_api_key,
                "base_url": "https://api.groq.com/openai/v1",
                "api_type": "openai"  # Use 'openai' type for Groq's OpenAI-compatible API
            }
        ]
        
        self.llm_config = {
            "config_list": self.config_list,
            "temperature": 0.7,
            "timeout": 120,
            "cache_seed": 42
        }
        
        self.initialize_agents()
    
    def initialize_agents(self):
        """Initialize AutoGen agents for onboarding assistance"""
        
        # Onboarding Specialist Agent
        self.onboarding_specialist = AssistantAgent(
            name="Onboarding_Specialist",
            system_message="""
            You are an expert onboarding specialist. Your role is to guide new users through the onboarding process.
            You help with:
            - Account setup and configuration
            - Platform navigation and features
            - Team introductions and collaboration
            - Best practices and tips
            - Troubleshooting common issues
            
            Be friendly, patient, and thorough in your explanations. Ask clarifying questions when needed.
            Provide step-by-step guidance and check for understanding.
            Always maintain a professional yet approachable tone.
            """,
            llm_config=self.llm_config,
            human_input_mode="NEVER"
        )
        
        # Technical Support Agent
        self.technical_support = AssistantAgent(
            name="Technical_Support",
            system_message="""
            You are a technical support specialist focusing on technical aspects of onboarding.
            You help with:
            - Software installation and setup
            - Technical configuration
            - Integration with other tools
            - API and developer documentation
            - Troubleshooting technical issues
            
            Provide clear, technical instructions. Include code examples when relevant.
            Explain technical concepts in an accessible way.
            """,
            llm_config=self.llm_config,
            human_input_mode="NEVER"
        )
        
        # User Proxy Agent
        self.user_proxy = UserProxyAgent(
            name="User_Proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=5,
            code_execution_config=False,
            llm_config=self.llm_config,
            system_message="""You represent the user in conversations with onboarding agents.
            Always be clear about user needs and provide context when necessary."""
        )
    
    def process_message_simple(self, message: str) -> str:
        """Process user message using direct Groq API (fallback)"""
        return self.groq_client.get_completion(message)
    
    def process_message_autogen(self, message: str) -> str:
        """Process user message through AutoGen agent system"""
        try:
            # Reset the chat to avoid context issues
            self.user_proxy.reset()
            self.onboarding_specialist.reset()
            
            # Initiate chat with the onboarding specialist
            chat_result = self.user_proxy.initiate_chat(
                self.onboarding_specialist,
                message=message,
                max_turns=2,
                summary_method="last_msg"
            )
            
            # Extract the last message
            if hasattr(chat_result, 'chat_history') and chat_result.chat_history:
                for msg in reversed(chat_result.chat_history):
                    if hasattr(msg, 'content') and msg.content:
                        return str(msg.content)
            
            return "I apologize, but I didn't receive a proper response. Please try again."
            
        except Exception as e:
            # Fallback to direct Groq API
            st.warning(f"AutoGen encountered an issue, using direct API: {str(e)}")
            return self.process_message_simple(message)

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent_initialized" not in st.session_state:
        st.session_state.agent_initialized = False
    
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = "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"
    
    if "onboarding_agent" not in st.session_state:
        st.session_state.onboarding_agent = None
    
    if "use_autogen" not in st.session_state:
        st.session_state.use_autogen = True

def initialize_agent():
    """Initialize the AI agent"""
    try:
        st.session_state.onboarding_agent = OnboardingAIAgent(st.session_state.groq_api_key)
        st.session_state.agent_initialized = True
        return True
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        return False

def display_chat_message(role: str, content: str, timestamp: str):
    """Display a chat message with appropriate styling"""
    if role == "user":
        css_class = "user-message"
        header = "👤 You"
    elif role == "assistant":
        css_class = "assistant-message"
        header = "🤖 Onboarding Assistant"
    else:
        css_class = "system-message"
        header = "⚙️ System"
    
    st.markdown(f"""
    <div class="agent-message {css_class}">
        <div class="message-header">{header} - {timestamp}</div>
        <div class="message-content">{content}</div>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.markdown('<div class="main-header">🤖 AI Onboarding Assistant</div>', unsafe_allow_html=True)
    st.markdown("Powered by AutoGen + Llama-3.3-70b-versatile (Groq)")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key configuration
        api_key = st.text_input(
            "Groq API Key",
            value=st.session_state.groq_api_key,
            type="password",
            help="Your Groq API key for accessing Llama-3.3-70b-versatile"
        )
        
        if api_key != st.session_state.groq_api_key:
            st.session_state.groq_api_key = api_key
            st.session_state.agent_initialized = False
        
        # Mode selection
        st.session_state.use_autogen = st.checkbox(
            "Use AutoGen Multi-Agent System", 
            value=st.session_state.use_autogen,
            help="Use multiple specialized agents (may be slower but more comprehensive)"
        )
        
        # Initialize agent button
        if not st.session_state.agent_initialized:
            if st.button("🚀 Initialize AI Agent", use_container_width=True):
                with st.spinner("Initializing AI agents..."):
                    if initialize_agent():
                        st.success("AI agents initialized successfully!")
                        st.rerun()
        
        st.markdown("---")
        st.header("🎯 Quick Actions")
        
        # Suggested questions
        suggested_questions = [
            "Help me set up my account",
            "What are the main features of this platform?",
            "How do I connect with my team?",
            "Can you guide me through the profile setup?",
            "What are the best practices for getting started?"
        ]
        
        for question in suggested_questions:
            if st.button(question, key=f"quick_{question}"):
                if st.session_state.agent_initialized:
                    # Add user message
                    st.session_state.messages.append({
                        "role": "user",
                        "content": question,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    
                    # Process with agent
                    with st.spinner("🤖 Processing..."):
                        if st.session_state.use_autogen:
                            response = st.session_state.onboarding_agent.process_message_autogen(question)
                        else:
                            response = st.session_state.onboarding_agent.process_message_simple(question)
                    
                    # Add assistant response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    st.rerun()
                else:
                    st.warning("Please initialize the AI agent first!")
        
        st.markdown("---")
        st.header("🛠️ Agent Status")
        
        if st.session_state.agent_initialized:
            mode = "AutoGen Multi-Agent" if st.session_state.use_autogen else "Direct API"
            st.markdown(f"""
            <div class="agent-status">
                ✅ Agents Initialized
                <br>• Mode: {mode}
                <br>• Model: Llama-3.3-70b-versatile
                <br>• API: Groq
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="padding: 10px; border-radius: 5px; background-color: #fff3cd; border-left: 3px solid #ffc107;">
                ⚠️ Agents Not Initialized
                <br>Click the button above to initialize
            </div>
            """, unsafe_allow_html=True)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Onboarding Chat")
        
        # Display chat messages
        for message in st.session_state.messages:
            display_chat_message(
                message["role"],
                message["content"],
                message["timestamp"]
            )
        
        # Chat input
        if st.session_state.agent_initialized:
            user_input = st.chat_input("Type your onboarding question here...")
            
            if user_input:
                # Add user message
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                
                # Process with agent
                with st.spinner("🤖 Processing your request..."):
                    if st.session_state.use_autogen:
                        response = st.session_state.onboarding_agent.process_message_autogen(user_input)
                    else:
                        response = st.session_state.onboarding_agent.process_message_simple(user_input)
                
                # Add assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                
                st.rerun()
        else:
            st.info("👆 Please initialize the AI agent in the sidebar to start chatting!")
    
    with col2:
        st.subheader("📊 Onboarding Progress")
        
        # Progress tracking
        progress_stages = {
            "Account Setup": 0,
            "Profile Completion": 0,
            "Platform Orientation": 0,
            "Team Integration": 0,
            "First Project": 0
        }
        
        # Analyze conversation to determine progress
        if st.session_state.messages:
            conversation_text = " ".join([msg["content"] for msg in st.session_state.messages])
            
            # Simple keyword-based progress detection
            if any(word in conversation_text.lower() for word in ["account", "setup", "create"]):
                progress_stages["Account Setup"] = 80
            if any(word in conversation_text.lower() for word in ["profile", "bio", "complete"]):
                progress_stages["Profile Completion"] = 60
            if any(word in conversation_text.lower() for word in ["feature", "platform", "navigate"]):
                progress_stages["Platform Orientation"] = 70
            if any(word in conversation_text.lower() for word in ["team", "colleague", "connect"]):
                progress_stages["Team Integration"] = 40
            if any(word in conversation_text.lower() for word in ["project", "task", "work"]):
                progress_stages["First Project"] = 20
        
        # Display progress bars
        for stage, progress in progress_stages.items():
            st.write(f"**{stage}**")
            st.progress(progress / 100)
            st.write(f"{progress}% complete")
            st.markdown("---")
        
        # Agent activity monitor
        st.subheader("🔄 Agent Activity")
        if st.session_state.agent_initialized:
            if st.session_state.use_autogen:
                st.success("✅ Onboarding Specialist: Active")
                st.success("✅ Technical Support: Ready")
                st.success("✅ User Proxy: Active")
                st.info("🤝 Agents are collaborating in real-time")
            else:
                st.success("✅ Direct API Mode: Active")
                st.info("⚡ Using direct Groq API for faster responses")
        else:
            st.warning("Agents are offline")

if __name__ == "__main__":
    main()
