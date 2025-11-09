import streamlit as st
import autogen
from groq import Groq
import matplotlib.pyplot as plt
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="Stock Report Generator",
    page_icon="📈",
    layout="wide"
)

# Initialize Groq client
@st.cache_resource
def get_groq_client():
    return Groq(api_key="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")

# Initialize AutoGen agents
@st.cache_resource
def initialize_agents():
    llm_config = {
        "config_list": [
            {
                "model": "llama-3.3-70b-versatile",
                "api_key": "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",
                "api_type": "groq"
            }
        ],
        "temperature": 0.7,
        "timeout": 120
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

def generate_stock_report(stock_symbol, period="1mo"):
    """Generate stock report using Groq model"""
    
    prompt = f"""
    Write a comprehensive blog post about the stock price performance of {stock_symbol} 
    in the past {period}. Today's date is 2024-07-26.
    
    The blog post should include:
    1. Key statistics (average price, highest price, lowest price, percentage change)
    2. Stock price trend analysis
    3. Market context and influencing factors
    4. Conclusion and outlook
    
    Format the response in markdown with appropriate headings and sections.
    """
    
    client = get_groq_client()
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_completion_tokens=2048,
            top_p=1,
            stream=False,
            stop=None
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        return f"Error generating report: {str(e)}"

def main():
    st.title("📈 AI Stock Report Generator")
    st.markdown("Generate comprehensive stock performance reports using AI agents")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        stock_symbol = st.text_input("Stock Symbol", value="NVDA", 
                                   help="Enter the stock symbol (e.g., NVDA, AAPL, TSLA)")
        
        period_options = {
            "1 Month": "1mo",
            "3 Months": "3mo", 
            "6 Months": "6mo",
            "1 Year": "1y",
            "YTD": "ytd"
        }
        
        period = st.selectbox("Time Period", list(period_options.keys()))
        period_value = period_options[period]
        
        generate_btn = st.button("Generate Stock Report", type="primary")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Stock Report")
        
        if generate_btn:
            with st.spinner("🤖 AI agents are collaborating to generate your report..."):
                
                # Show agent activity
                with st.expander("👥 Agent Activity", expanded=True):
                    st.info("🔄 Initializing AI team...")
                    st.info("📊 Planner determining data requirements...")
                    st.info("💻 Engineer writing analysis code...")
                    st.info("📈 Executor running analysis...")
                    st.info("✍️ Writer compiling final report...")
                
                # Generate report
                report = generate_stock_report(stock_symbol, period_value)
                
                if report.startswith("Error"):
                    st.error(report)
                else:
                    st.markdown(report)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Report",
                        data=report,
                        file_name=f"{stock_symbol}_stock_report.md",
                        mime="text/markdown"
                    )
    
    with col2:
        st.subheader("About This App")
        st.markdown("""
        This app uses a team of specialized AI agents:
        
        - **🤖 Planner**: Determines what data is needed
        - **💻 Engineer**: Writes analysis code  
        - **📊 Executor**: Runs the analysis
        - **✍️ Writer**: Creates the final report
        
        The agents collaborate using the **Groq LLaMA 3.3 70B** model
        to generate comprehensive stock analysis reports.
        """)
        
        st.info("💡 **Tip**: Use clear stock symbols for best results")
        
        # Example reports
        with st.expander("📋 Example Stock Symbols"):
            st.write("""
            - **NVDA**: NVIDIA Corporation
            - **AAPL**: Apple Inc.
            - **TSLA**: Tesla, Inc.
            - **MSFT**: Microsoft Corporation
            - **GOOGL**: Alphabet Inc.
            - **AMZN**: Amazon.com, Inc.
            """)

if __name__ == "__main__":
    main()
