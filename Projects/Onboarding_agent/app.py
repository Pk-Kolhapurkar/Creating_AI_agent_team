import streamlit as st
import json
import time
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Onboarding Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b3137;
        border-left: 5px solid #ff4b4b;
    }
    .chat-message.assistant {
        background-color: #262730;
        border-left: 5px solid #00d4aa;
    }
    .chat-message.system {
        background-color: #1e1e1e;
        border-left: 5px solid #ffd700;
    }
    .message-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .message-sender {
        font-weight: bold;
        font-size: 0.9rem;
    }
    .message-time {
        font-size: 0.8rem;
        color: #888;
    }
    .message-content {
        font-size: 1rem;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

class OnboardingChat:
    def __init__(self):
        self.responses = {
            "welcome": "Hello! I'm your onboarding assistant. I'm here to help you get started with our platform. What would you like to know about?",
            "getting_started": "To get started, you'll need to:\n1. Set up your account\n2. Complete your profile\n3. Explore the main features\n4. Connect with team members\n\nWhich step would you like help with?",
            "account_setup": "For account setup:\n• Go to Settings → Account\n• Verify your email\n• Set up two-factor authentication\n• Configure your preferences",
            "profile_completion": "Complete your profile by:\n• Adding a professional photo\n• Filling out your bio\n• Listing your skills\n• Setting your availability",
            "features": "Our main features include:\n• Project management tools\n• Team collaboration\n• File sharing\n• Analytics dashboard\n• Custom workflows",
            "team": "To connect with team members:\n• Browse the team directory\n• Join relevant channels\n• Schedule introductory meetings\n• Participate in team activities",
            "help": "I can help you with:\n• Getting started\n• Account setup\n• Profile completion\n• Platform features\n• Team connections\n• Technical issues",
            "unknown": "I'm not sure I understand. Could you rephrase your question? I can help with onboarding topics like account setup, profile completion, or platform features."
        }
        
        self.suggested_questions = [
            "How do I get started?",
            "Help with account setup",
            "What features are available?",
            "How to connect with team?",
            "Complete my profile"
        ]
    
    def get_response(self, user_input):
        user_input = user_input.lower().strip()
        
        if any(word in user_input for word in ["hello", "hi", "hey", "start", "begin"]):
            return self.responses["welcome"]
        elif any(word in user_input for word in ["get started", "begin", "start"]):
            return self.responses["getting_started"]
        elif any(word in user_input for word in ["account", "setup", "create account"]):
            return self.responses["account_setup"]
        elif any(word in user_input for word in ["profile", "complete profile", "bio"]):
            return self.responses["profile_completion"]
        elif any(word in user_input for word in ["feature", "what can", "tools", "function"]):
            return self.responses["features"]
        elif any(word in user_input for word in ["team", "connect", "colleague", "member"]):
            return self.responses["team"]
        elif any(word in user_input for word in ["help", "support", "assist"]):
            return self.responses["help"]
        else:
            return self.responses["unknown"]

def initialize_chat():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_bot" not in st.session_state:
        st.session_state.chat_bot = OnboardingChat()
    
    # Add welcome message if no messages exist
    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant",
            "content": st.session_state.chat_bot.responses["welcome"],
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })

def display_chat_message(role, content, timestamp):
    if role == "user":
        css_class = "user"
        sender = "You"
    elif role == "assistant":
        css_class = "assistant"
        sender = "Onboarding Assistant"
    else:
        css_class = "system"
        sender = "System"
    
    st.markdown(f"""
    <div class="chat-message {css_class}">
        <div class="message-header">
            <span class="message-sender">{sender}</span>
            <span class="message-time">{timestamp}</span>
        </div>
        <div class="message-content">{content}</div>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.title("🤖 Onboarding Chat Assistant")
    st.markdown("Welcome to your personalized onboarding experience! I'm here to help you get started.")
    
    # Initialize chat
    initialize_chat()
    
    # Sidebar with information and controls
    with st.sidebar:
        st.header("About This Assistant")
        st.markdown("""
        This onboarding assistant helps you:
        - Set up your account
        - Complete your profile
        - Learn platform features
        - Connect with your team
        - Get answers to common questions
        """)
        
        st.header("Quick Actions")
        if st.button("🔄 Clear Chat History"):
            st.session_state.messages = []
            st.session_state.messages.append({
                "role": "assistant",
                "content": st.session_state.chat_bot.responses["welcome"],
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            st.rerun()
        
        st.header("Suggested Questions")
        for question in st.session_state.chat_bot.suggested_questions:
            if st.button(question, key=f"btn_{question}"):
                # Add user message
                st.session_state.messages.append({
                    "role": "user",
                    "content": question,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                
                # Get and add assistant response
                response = st.session_state.chat_bot.get_response(question)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                st.rerun()
    
    # Main chat area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Chat")
        
        # Display chat messages
        for message in st.session_state.messages:
            display_chat_message(
                message["role"], 
                message["content"], 
                message["timestamp"]
            )
    
    with col2:
        st.subheader("Onboarding Progress")
        
        # Simulated progress tracker
        progress_data = {
            "Account Setup": 85,
            "Profile Completion": 60,
            "Feature Training": 45,
            "Team Integration": 30
        }
        
        for task, progress in progress_data.items():
            st.write(f"**{task}**")
            st.progress(progress / 100)
            st.write(f"{progress}% complete")
            st.markdown("---")
    
    # Chat input at the bottom
    st.markdown("---")
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        # Get assistant response
        with st.spinner("Thinking..."):
            time.sleep(0.5)  # Simulate processing time
            response = st.session_state.chat_bot.get_response(user_input)
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
        
        st.rerun()

if __name__ == "__main__":
    main()
