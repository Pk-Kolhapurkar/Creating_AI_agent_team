import streamlit as st
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import os
import time
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="AI Stock Report Generator",
    page_icon="📈",
    layout="wide"
)

# Custom configuration for Groq
class GroqConfig:
    def __init__(self):
        self.api_key = "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"
        self.base_url = "https://api.groq.com/openai/v1"
        self.model = "llama-3.3-70b-versatile"

def initialize_agents():
    """Initialize AutoGen agents with Groq configuration"""
    
    groq_config = GroqConfig()
    
    # LLM configuration for Groq
    llm_config = {
        "config_list": [
            {
                "model": groq_config.model,
                "api_key": groq_config.api_key,
                "api_type": "openai",
                "base_url": groq_config.base_url,
            }
        ],
        "temperature": 0.7,
        "timeout": 120,
        "max_tokens": 2048,
    }

    # User Proxy Agent
    user_proxy = UserProxyAgent(
        name="Admin",
        system_message="""You are the Admin. Give the task to the team and provide feedback on the final report.
        You can ask the writer to refine the blog post if needed.""",
        code_execution_config=False,
        human_input_mode="NEVER",
        llm_config=llm_config,
    )

    # Planner Agent
    planner = AssistantAgent(
        name="Planner",
        system_message="""You are the Planner. Given a task, determine what information is needed to complete it.
        Focus on information that can be retrieved using Python code.
        After each step is done by others, check progress and instruct remaining steps.
        If a step fails, try to find workarounds.""",
        llm_config=llm_config,
    )

    # Engineer Agent
    engineer = AssistantAgent(
        name="Engineer",
        system_message="""You are the Engineer. Write Python code to accomplish the tasks specified by the Planner.
        You can use libraries like yfinance, pandas, and matplotlib for stock analysis.
        Make sure your code is efficient and well-documented.""",
        llm_config=llm_config,
    )

    # Executor Agent
    executor = UserProxyAgent(
        name="Executor",
        system_message="""You are the Executor. Execute the Python code written by the Engineer.
        Report the results and any errors encountered during execution.""",
        human_input_mode="NEVER",
        code_execution_config={
            "last_n_messages": 3,
            "work_dir": "coding",
            "use_docker": False,
        },
        llm_config=False,  # Executor doesn't need LLM for code execution
    )

    # Writer Agent
    writer = AssistantAgent(
        name="Writer",
        system_message="""You are the Writer. Create comprehensive blog posts in markdown format.
        Include relevant titles, sections for key statistics, trend analysis, market context, and conclusions.
        Format the content professionally and ensure it's engaging to read.
        Take feedback from the Admin to refine your blog.""",
        llm_config=llm_config,
    )

    return user_proxy, planner, engineer, executor, writer

def create_stock_chart(stock_data, symbol):
    """Create a stock price chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(stock_data.index, stock_data['Close'], label=f'{symbol} Closing Price', linewidth=2)
    ax.set_title(f'{symbol} Stock Price - Past 3 Months', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Convert plot to base64 for display in Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return base64.b64encode(buf.read()).decode()

def get_stock_data(symbol, period="3mo"):
    """Get stock data using yfinance"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        return hist
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return None

def main():
    st.title("🤖 AI Stock Report Generator")
    st.markdown("Multi-Agent System with AutoGen & Groq LLaMA 3.3 70B")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("📊 Configuration")
        
        stock_symbol = st.text_input(
            "Stock Symbol", 
            value="TTM",
            help="Enter the stock symbol (e.g., TTM for Tata Motors, NVDA, AAPL)"
        )
        
        analysis_period = st.selectbox(
            "Analysis Period",
            ["1mo", "3mo", "6mo", "1y", "ytd"],
            index=1,
            format_func=lambda x: {
                "1mo": "Past Month",
                "3mo": "Past 3 Months", 
                "6mo": "Past 6 Months",
                "1y": "Past Year",
                "ytd": "Year to Date"
            }[x]
        )
        
        st.markdown("---")
        st.markdown("### 🎯 AI Team")
        st.markdown("""
        - **Planner**: Data requirements
        - **Engineer**: Code writing
        - **Executor**: Code execution  
        - **Writer**: Report creation
        """)
        
        if st.button("🔄 Clear Chat", type="secondary"):
            st.rerun()
            
        generate_btn = st.button("🚀 Generate Stock Report", type="primary", use_container_width=True)
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if generate_btn and stock_symbol:
            # Create task
            period_display = {
                "1mo": "past month",
                "3mo": "past 3 months", 
                "6mo": "past 6 months",
                "1y": "past year",
                "ytd": "year to date"
            }[analysis_period]
            
            task = f"Write a comprehensive blogpost about the stock price performance of {stock_symbol} in the {period_display}. Today's date is 2024-07-26. Include key statistics, trend analysis, and market context."
            
            st.subheader("📋 Task")
            st.info(task)
            
            # Show stock data first
            with st.spinner("📊 Fetching stock data..."):
                stock_data = get_stock_data(stock_symbol, analysis_period)
                
                if stock_data is not None and not stock_data.empty:
                    # Display basic stock info
                    col1a, col2a, col3a, col4a = st.columns(4)
                    
                    with col1a:
                        st.metric(
                            "Current Price", 
                            f"${stock_data['Close'].iloc[-1]:.2f}",
                            f"${stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[0]:.2f}"
                        )
                    
                    with col2a:
                        st.metric(
                            "High", 
                            f"${stock_data['High'].max():.2f}"
                        )
                    
                    with col3a:
                        st.metric(
                            "Low", 
                            f"${stock_data['Low'].min():.2f}"
                        )
                    
                    with col4a:
                        pct_change = ((stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[0]) / stock_data['Close'].iloc[0]) * 100
                        st.metric(
                            "Total Change", 
                            f"{pct_change:.2f}%"
                        )
                    
                    # Display chart
                    st.subheader("📈 Price Chart")
                    chart_base64 = create_stock_chart(stock_data, stock_symbol)
                    st.image(f"data:image/png;base64,{chart_base64}")
            
            # Initialize progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            agent_activity = st.empty()
            
            # Simulate agent collaboration
            steps = [
                "🔄 Initializing AI team...",
                "📋 Planner analyzing requirements...",
                "💻 Engineer writing analysis code...",
                "📊 Executor running analysis...",
                "✍️ Writer compiling final report...",
                "✅ Finalizing report..."
            ]
            
            for i, step in enumerate(steps):
                status_text.text(step)
                agent_activity.info(f"Current step: {step}")
                progress_bar.progress((i + 1) * (100 // len(steps)))
                time.sleep(2)
            
            # Generate report using the agents
            try:
                with st.spinner("🤖 AI agents are collaborating to generate your report..."):
                    # Initialize agents
                    user_proxy, planner, engineer, executor, writer = initialize_agents()
                    
                    # Create group chat
                    groupchat = GroupChat(
                        agents=[user_proxy, planner, engineer, executor, writer],
                        messages=[],
                        max_round=12,
                    )
                    
                    # Create manager
                    manager = GroupChatManager(
                        groupchat=groupchat,
                        llm_config=user_proxy.llm_config
                    )
                    
                    # Start the chat
                    chat_result = user_proxy.initiate_chat(
                        manager,
                        message=task,
                        max_turns=10
                    )
                
                # Display results
                st.subheader("📄 Generated Stock Report")
                
                if hasattr(chat_result, 'summary') and chat_result.summary:
                    st.markdown(chat_result.summary)
                elif hasattr(chat_result, 'chat_history') and chat_result.chat_history:
                    # Extract the final report from chat history
                    for msg in reversed(chat_result.chat_history):
                        if hasattr(msg, 'content') and msg.content:
                            if any(keyword in msg.content.lower() for keyword in ['blog', 'report', 'analysis', 'conclusion']):
                                st.markdown(msg.content)
                                break
                    else:
                        # Fallback: show last message
                        last_msg = chat_result.chat_history[-1]
                        if hasattr(last_msg, 'content'):
                            st.markdown(last_msg.content)
                        else:
                            st.markdown(str(last_msg))
                else:
                    # Fallback: generate a simple report
                    st.warning("Using fallback report generation...")
                    fallback_report = generate_fallback_report(stock_symbol, stock_data, period_display)
                    st.markdown(fallback_report)
                
                status_text.success("🎉 Stock report generated successfully!")
                
                # Download button
                report_text = st.session_state.get('current_report', '')
                if report_text:
                    st.download_button(
                        label="📥 Download Report",
                        data=report_text,
                        file_name=f"{stock_symbol}_stock_report.md",
                        mime="text/markdown"
                    )
                
            except Exception as e:
                st.error(f"❌ Error in AI collaboration: {str(e)}")
                st.info("Generating fallback report...")
                if stock_data is not None:
                    fallback_report = generate_fallback_report(stock_symbol, stock_data, period_display)
                    st.markdown(fallback_report)
        
        elif not stock_symbol:
            st.warning("⚠️ Please enter a stock symbol")
        
        else:
            # Welcome message
            st.subheader("🚀 Welcome to AI Stock Report Generator")
            st.markdown("""
            ### How to use:
            1. **Enter a stock symbol** in the sidebar (e.g., TTM, NVDA, AAPL)
            2. **Select analysis period** for your report
            3. **Click 'Generate Stock Report'** to start the AI team
            
            ### Example Stock Symbols:
            - **TTM**: Tata Motors
            - **NVDA**: NVIDIA
            - **AAPL**: Apple
            - **TSLA**: Tesla
            - **RELIANCE.NS**: Reliance Industries (NSE)
            
            The AI team will collaborate to fetch data, analyze trends, and create a comprehensive report.
            """)
    
    with col2:
        st.subheader("👥 Live Agent Activity")
        
        if generate_btn and stock_symbol:
            activities = [
                "🤝 Team initialized",
                "📋 Planning data requirements",
                "💻 Writing analysis code", 
                "📊 Executing stock analysis",
                "📈 Processing results",
                "✍️ Writing final report",
                "✅ Quality check",
                "🎯 Report finalized"
            ]
            
            for activity in activities:
                st.write(f"• {activity}")
                time.sleep(1.5)

def generate_fallback_report(symbol, stock_data, period):
    """Generate a fallback report if AI collaboration fails"""
    if stock_data is None or stock_data.empty:
        return "Unable to fetch stock data. Please check the symbol and try again."
    
    current_price = stock_data['Close'].iloc[-1]
    high_price = stock_data['High'].max()
    low_price = stock_data['Low'].min()
    start_price = stock_data['Close'].iloc[0]
    pct_change = ((current_price - start_price) / start_price) * 100
    
    report = f"""
# {symbol} Stock Performance Analysis - {period.capitalize()}

## Key Statistics

- **Current Price**: ${current_price:.2f}
- **Highest Price**: ${high_price:.2f}
- **Lowest Price**: ${low_price:.2f}
- **Price Change**: ${current_price - start_price:.2f} ({pct_change:.2f}%)
- **Analysis Period**: {period}

## Performance Summary

The stock of {symbol} has shown {'positive' if pct_change > 0 else 'negative'} performance during the {period}, 
with a total change of {pct_change:.2f}%.

## Market Context

This analysis covers the period up to July 26, 2024. The stock demonstrated volatility typical of equity markets, 
with fluctuations influenced by market conditions, company performance, and broader economic factors.

## Conclusion

Based on the {period} performance, {symbol} has {'outperformed' if pct_change > 0 else 'underperformed'} relative to its starting price. 
Investors should consider this performance in the context of their investment strategy and market conditions.

*Note: This is an automated analysis. Please conduct additional research before making investment decisions.*
"""
    
    return report

if __name__ == "__main__":
    main()
