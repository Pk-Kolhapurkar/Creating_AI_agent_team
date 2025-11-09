import streamlit as st
from autogen import ConversableAgent
import groq

# Cheatsheet: https://cheat-sheet.streamlit.app/

# Groq configuration
groq_client = groq.Groq(api_key="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")

llm_config = {
    "model": "llama-3.3-70b-versatile",
    "api_key": "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",
    "base_url": "https://api.groq.com/openai/v1",
    "api_type": "openai"
}

jack = ConversableAgent(
    "Jack",
    llm_config=llm_config,
    system_message="Your name is Jack and you are a stand-up comedian in a two-person comedy show.",
)
rose = ConversableAgent(
    "Rose",
    llm_config=llm_config,
    system_message="Your name is Rose and you are a stand-up comedian in a two-person comedy show.",
)

st.title("🎭 Comedy Show with Groq AI")
st.markdown("Two AI comedians having a conversation using Groq's Llama-3.3-70b model")

initial_msg = st.text_input("How should Jack initiate the exchange with Rose?", 
                           value="Hey Rose, why don't scientists trust atoms?")
hit_button = st.button('🎤 Jokes ON')

if hit_button and initial_msg:
    with st.spinner("Setting up the comedy stage..."):
        try:
            chat_result = jack.initiate_chat(
                rose, 
                message=initial_msg, 
                max_turns=2
            )
            
            st.markdown("---")
            st.subheader("🎭 Comedy Exchange:")
            
            for i, msg in enumerate(chat_result.chat_history):
                speaker = "Jack" if msg["role"] == "assistant" else "Rose"
                color = ":orange[" if msg["role"] == "assistant" else ":red["
                
                st.markdown(f"{color}{speaker}]")
                st.markdown(msg["content"] + "\n")
                
                # Add a separator between exchanges
                if i < len(chat_result.chat_history) - 1:
                    st.markdown("---")
                    
        except Exception as e:
            st.error(f"Error in the comedy show: {str(e)}")
            st.info("Make sure you have the required dependencies installed and a valid Groq API key.")
else:
    st.info("👆 Enter a starting line and click 'Jokes ON' to begin the comedy show!")

# Add some styling
st.markdown("""
<style>
    .stButton button {
        background-color: #FF6B35;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #E55A2B;
    }
</style>
""", unsafe_allow_html=True)
