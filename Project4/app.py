import streamlit as st
import autogen
from groq import Groq
import asyncio
import threading
from queue import Queue
import time

# Set page configuration
st.set_page_config(
    page_title="AI Stock Report Generator",
    page_icon="📈",
    layout="wide"
)

# Initialize Groq client
def get_groq_client():
    return Groq(api_key="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")

# Custom LLM configuration for Groq
class GroqLLMConfig:
    def __init__(self):
        self.client = get_groq_client()
    
    def create_chat_completion(self, messages, **kwargs):
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1024),
                top_p=kwargs.get('top_p', 1),
                stream=False,
                stop=kwargs.get('stop', None)
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

# Initialize AutoGen agents with Groq
def initialize_agents():
    groq_llm = GroqLLMConfig()
    
    llm_config = {
        "config_list": [{
            "type": "groq",
            "model": "llama-3.3-70b-versatile",
            "api_key": "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"
        }],
        "timeout": 120,
        "temperature": 0.7
    }

    # User Proxy Agent
    user_proxy = autogen.ConversableAgent(
        name="Admin",
        system_message="Give the task, and send instructions to writer to refine the blog post.",
        code_execution_config=False,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # Planner Agent
    planner = autogen.ConversableAgent(
        name="Planner",
        system_message="Given a task, please determine what information is needed to complete the task. "
                      "Please note that the information will all be retrieved using Python code. "
                      "Please only suggest information that can be retrieved using Python code. "
                      "After each step is done by others, check the progress and instruct the remaining steps. "
                      "If a step fails, try to workaround",
        description="Planner. Given a task, determine what information is needed to complete the task. "
                   "After each step is done by others, check the progress and instruct the remaining steps",
        llm_config=llm_config,
    )

    # Engineer Agent
    engineer = autogen.AssistantAgent(
        name="Engineer",
        llm_config=llm_config,
        description="An engineer that writes code based on the plan provided by the planner.",
    )

    # Executor Agent
    executor = autogen.ConversableAgent(
        name="Executor",
        system_message="Execute the code written by the engineer and report the result.",
        human_input_mode="NEVER",
        code_execution_config={
            "last_n_messages": 3,
            "work_dir": "coding",
            "use_docker": False,
        },
    )

    # Writer Agent
    writer = autogen.ConversableAgent(
        name="Writer",
        llm_config=llm_config,
        system_message="Writer. Please write blogs in markdown format (with relevant titles) "
                      "and put the content in pseudo ```md``` code block. "
                      "You take feedback from the admin and refine your blog.",
        description="Writer. Write blogs based on the code execution results and take "
                   "feedback from the admin to refine the blog."
    )

    return user_proxy, planner, engineer, executor, writer

# Custom group chat manager to capture messages
class StreamlitGroupChatManager(autogen.GroupChatManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message_queue = Queue()
        
    def send(self, message, recipient, request_reply=None, silent=False):
        # Capture messages for Streamlit display
        if hasattr(self, 'message_queue'):
            self.message_queue.put({
                'sender': self.name,
                'recipient': recipient.name,
                'message': message
            })
        return super().send(message, recipient, request_reply=request_reply, silent=silent)

def run_group_chat(task, message_container, progress_bar):
    """Run the group chat and update Streamlit UI with progress"""
    try:
        # Initialize agents
        user_proxy, planner, engineer, executor, writer = initialize_agents()
        
        # Create group chat
        groupchat = autogen.GroupChat(
            agents=[user_proxy, engineer, writer, executor, planner],
            messages=[],
            max_round=20,
        )
        
        # Create custom manager
        manager = StreamlitGroupChatManager(
            groupchat=groupchat, 
            llm_config=initialize_agents()[0].llm_config  # Get llm_config from any agent
        )
        
        # Start the chat
        message_container.info("🚀 Starting AI team collaboration...")
        
        # Run the chat
        chat_result = user_proxy.initiate_chat(
            manager,
            message=task,
            max_turns=20
        )
        
        return chat_result
        
    except Exception as e:
        message_container.error(f"Error in group chat: {str(e)}")
        return None

def main():
    st.title("🤖 AI Stock Report Generator")
    st.markdown("Using Multi-Agent System with AutoGen and Groq LLaMA 3.3 70B")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("📊 Stock Analysis Setup")
        
        stock_symbol = st.text_input(
            "Stock Symbol", 
            value="NVDA",
            help="Enter the stock symbol (e.g., NVDA, AAPL, TSLA)"
        )
        
        analysis_period = st.selectbox(
            "Analysis Period",
            ["Past Month", "Past 3 Months", "Past 6 Months", "Past Year", "YTD"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### 🎯 AI Team Members")
        st.markdown("""
        - **Planner**: Determines data requirements
        - **Engineer**: Writes analysis code
        - **Executor**: Runs the code
        - **Writer**: Creates final report
        """)
        
        generate_btn = st.button("Generate Stock Report", type="primary", use_container_width=True)
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if generate_btn:
            # Create task string
            task = f"Write a blogpost about the stock price performance of {stock_symbol} in the {analysis_period.lower()}. Today's date is 2024-07-26."
            
            st.subheader("📋 Task")
            st.info(task)
            
            # Create containers for progress and results
            progress_container = st.container()
            message_container = st.empty()
            result_container = st.container()
            
            with progress_container:
                st.subheader("👥 AI Team Activity")
                
                # Initialize progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate progress updates
                for i in range(5):
                    status_text.text(f"🔄 Step {i+1}/5: AI agents collaborating...")
                    progress_bar.progress((i + 1) * 20)
                    time.sleep(1)
                
                status_text.text("✅ Generating final report...")
                progress_bar.progress(100)
            
            # Run the group chat
            with st.spinner("AI agents are working together to generate your report..."):
                try:
                    # Initialize agents and run chat
                    user_proxy, planner, engineer, executor, writer = initialize_agents()
                    
                    # Create group chat
                    groupchat = autogen.GroupChat(
                        agents=[user_proxy, engineer, writer, executor, planner],
                        messages=[],
                        max_round=15,
                    )
                    
                    # Create manager
                    manager = autogen.GroupChatManager(
                        groupchat=groupchat, 
                        llm_config=user_proxy.llm_config
                    )
                    
                    # Run the chat
                    chat_result = user_proxy.initiate_chat(
                        manager,
                        message=task,
                        max_turns=15
                    )
                    
                    # Display results
                    with result_container:
                        st.subheader("📄 Generated Stock Report")
                        
                        if hasattr(chat_result, 'summary'):
                            st.markdown(chat_result.summary)
                        else:
                            # Extract the last message which should be the report
                            if chat_result and hasattr(chat_result, 'chat_history'):
                                last_message = chat_result.chat_history[-1] if chat_result.chat_history else "No report generated"
                                if hasattr(last_message, 'content'):
                                    st.markdown(last_message.content)
                                else:
                                    st.markdown(str(last_message))
                            else:
                                st.error("No report was generated. Please try again.")
                    
                    message_container.success("🎉 Stock report generated successfully!")
                    
                except Exception as e:
                    message_container.error(f"❌ Error: {str(e)}")
                    st.error("Please check your API key and try again.")
        
        else:
            # Show instructions when no report is generated
            st.subheader("Welcome to AI Stock Report Generator")
            st.markdown("""
            ### How it works:
            
            1. **Enter a stock symbol** in the sidebar (e.g., NVDA, AAPL, TSLA)
            2. **Select analysis period** for the report
            3. **Click 'Generate Stock Report'** to start the AI team
            
            ### What happens behind the scenes:
            
            🤖 **Planner Agent**: Analyzes what data is needed
            💻 **Engineer Agent**: Writes Python code to fetch and analyze stock data  
            📊 **Executor Agent**: Runs the code and processes the data
            ✍️ **Writer Agent**: Creates a comprehensive blog post with analysis
            
            The agents collaborate using the **Groq LLaMA 3.3 70B model** to generate professional stock reports.
            """)
    
    with col2:
        st.subheader("🔍 Live Agent Activity")
        
        if generate_btn:
            # Simulate agent activity
            agent_activities = [
                "🔄 Planner: Determining data requirements...",
                "💻 Engineer: Writing analysis code...",
                "📊 Executor: Fetching stock data...",
                "📈 Executor: Calculating statistics...",
                "✍️ Writer: Compiling final report...",
                "✅ Finalizing report...",
            ]
            
            activity_container = st.container()
            
            with activity_container:
                for i, activity in enumerate(agent_activities):
                    st.write(f"{activity}")
                    time.sleep(1.5)
        
        st.markdown("---")
        st.subheader("📋 Supported Stocks")
        st.markdown("""
        - **NVDA**: NVIDIA
        - **AAPL**: Apple
        - **TSLA**: Tesla
        - **MSFT**: Microsoft
        - **GOOGL**: Google
        - **AMZN**: Amazon
        - **META**: Meta
        - **And many more...**
        """)

if __name__ == "__main__":
    main()
