import streamlit as st
import os
import json
from datetime import datetime
from typing import Dict, List
import groq
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import io
import base64

# Try to import AutoGen
try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    AUTOGEN_AVAILABLE = True
except ImportError as e:
    st.warning(f"AutoGen not available: {str(e)}. Using direct Groq API only.")
    AUTOGEN_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="AI Stock Report Planning System",
    page_icon="📈",
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
    .planner-message {
        background-color: #fff0f0;
        border-left-color: #dc3545;
    }
    .engineer-message {
        background-color: #f0fff0;
        border-left-color: #28a745;
    }
    .writer-message {
        background-color: #fff8e1;
        border-left-color: #ffc107;
    }
    .executor-message {
        background-color: #e8f4fd;
        border-left-color: #17a2b8;
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
    .stock-chart {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
    
    def get_completion(self, message: str, system_message: str = None) -> str:
        """Get completion from Groq API"""
        try:
            messages = []
            
            if system_message:
                messages.append({
                    "role": "system",
                    "content": system_message
                })
            
            messages.append({
                "role": "user",
                "content": message
            })
            
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                top_p=1,
                stream=False
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Please try again."

class StockDataFetcher:
    def __init__(self):
        pass
    
    def fetch_stock_data(self, symbol: str, period: str = "1mo"):
        """Fetch stock data using yfinance"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)
            
            if hist.empty:
                return None, f"No data found for symbol {symbol}"
            
            # Get additional info
            info = stock.info
            company_name = info.get('longName', symbol)
            
            return {
                'history': hist,
                'company_name': company_name,
                'symbol': symbol,
                'info': info
            }, None
        except Exception as e:
            return None, f"Error fetching data: {str(e)}"
    
    def create_stock_chart(self, hist_data):
        """Create a stock price chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(hist_data.index, hist_data['Close'], linewidth=2, label='Close Price')
        ax.set_title('Stock Price Performance', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig

class StockReportAISystem:
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.groq_client = GroqClient(groq_api_key)
        self.stock_fetcher = StockDataFetcher()
        
        # System messages for different agent roles
        self.system_messages = {
            "planner": """
            You are a Stock Report Planner. Your role is to:
            - Analyze the user's stock report request
            - Determine what information is needed (price data, metrics, analysis period)
            - Create a step-by-step plan for generating the report
            - Coordinate with other agents (Engineer, Writer, Executor)
            - Ensure all necessary data is collected and analyzed
            
            Focus on:
            - Defining the scope of analysis
            - Identifying key metrics to include
            - Planning the report structure
            - Validating data completeness
            
            Be thorough and methodical in your planning.
            """,
            
            "engineer": """
            You are a Data Engineer specializing in stock analysis. Your role is to:
            - Write Python code to fetch and process stock data
            - Calculate key financial metrics and indicators
            - Generate charts and visualizations
            - Handle data cleaning and transformation
            - Ensure code efficiency and accuracy
            
            You work with:
            - yfinance for stock data
            - pandas for data manipulation
            - matplotlib for visualization
            - Technical indicators (SMA, RSI, etc.)
            
            Always write clean, well-documented code.
            """,
            
            "writer": """
            You are a Financial Writer specializing in stock reports. Your role is to:
            - Write comprehensive, professional stock analysis reports
            - Structure reports with clear sections (executive summary, analysis, conclusion)
            - Include key metrics and insights in markdown format
            - Provide actionable investment insights
            - Maintain professional tone with appropriate disclaimers
            
            Report structure should include:
            - Executive Summary
            - Performance Analysis
            - Key Metrics
            - Technical Analysis
            - Investment Outlook
            - Risk Factors
            - Conclusion
            
            Use markdown formatting with appropriate headers and sections.
            """,
            
            "executor": """
            You are a Code Executor. Your role is to:
            - Execute Python code written by the Engineer
            - Report execution results and any errors
            - Validate data outputs
            - Ensure code runs successfully
            - Provide feedback on execution issues
            
            Always verify code safety before execution.
            Report detailed results including any charts or data outputs.
            """
        }
        
        self.agents_initialized = False
        
        # Initialize AutoGen agents if available
        if AUTOGEN_AVAILABLE:
            try:
                self.initialize_autogen_agents()
                self.agents_initialized = True
            except Exception as e:
                st.warning(f"AutoGen initialization failed: {str(e)}. Using direct API mode.")
                self.agents_initialized = False
    
    def initialize_autogen_agents(self):
        """Initialize AutoGen agents for stock report planning"""
        if not AUTOGEN_AVAILABLE:
            return
        
        # Configure for AutoGen
        self.config_list = [
            {
                "model": "llama-3.3-70b-versatile", 
                "api_key": self.groq_api_key,
                "base_url": "https://api.groq.com/openai/v1",
                "api_type": "openai"
            }
        ]
        
        self.llm_config = {
            "config_list": self.config_list,
            "temperature": 0.7,
            "timeout": 120,
            "cache_seed": 42
        }
        
        # Create agents
        self.planner = AssistantAgent(
            name="Planner",
            system_message=self.system_messages["planner"],
            llm_config=self.llm_config,
            human_input_mode="NEVER"
        )
        
        self.engineer = AssistantAgent(
            name="Engineer",
            system_message=self.system_messages["engineer"],
            llm_config=self.llm_config,
            human_input_mode="NEVER"
        )
        
        self.writer = AssistantAgent(
            name="Writer",
            system_message=self.system_messages["writer"],
            llm_config=self.llm_config,
            human_input_mode="NEVER"
        )
        
        self.executor = AssistantAgent(
            name="Executor",
            system_message=self.system_messages["executor"],
            llm_config=self.llm_config,
            human_input_mode="NEVER"
        )
        
        # User Proxy Agent
        self.user_proxy = UserProxyAgent(
            name="User_Proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3,
            code_execution_config=False,
            llm_config=self.llm_config,
            system_message="You represent the user requesting stock analysis reports."
        )
        
        # Create group chat
        self.group_chat = GroupChat(
            agents=[self.user_proxy, self.planner, self.engineer, self.writer, self.executor],
            messages=[],
            max_round=15
        )
        
        self.manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.llm_config
        )
    
    def process_stock_request_simple(self, request: str) -> Dict:
        """Process stock request using direct API calls"""
        # Extract stock symbol from request
        symbols = self.extract_symbols_from_request(request)
        
        if not symbols:
            return {
                "success": False,
                "response": "Please specify a stock symbol (e.g., AAPL, TSLA, NVDA) in your request."
            }
        
        symbol = symbols[0]
        
        # Fetch stock data
        stock_data, error = self.stock_fetcher.fetch_stock_data(symbol)
        
        if error:
            return {
                "success": False,
                "response": f"Error fetching data for {symbol}: {error}"
            }
        
        # Generate analysis using AI
        analysis_prompt = f"""
        Analyze the stock {symbol} ({stock_data['company_name']}) and provide a comprehensive report.
        
        Available data: {len(stock_data['history'])} days of price history.
        Current request: {request}
        
        Please provide:
        1. Executive Summary
        2. Performance Analysis
        3. Key Metrics
        4. Technical Insights
        5. Investment Outlook
        6. Risk Factors
        
        Format the response in markdown with clear sections.
        """
        
        analysis = self.groq_client.get_completion(
            analysis_prompt,
            self.system_messages["writer"]
        )
        
        # Create chart
        chart_fig = self.stock_fetcher.create_stock_chart(stock_data['history'])
        
        return {
            "success": True,
            "symbol": symbol,
            "company_name": stock_data['company_name'],
            "analysis": analysis,
            "chart": chart_fig,
            "data": stock_data['history'],
            "raw_data": stock_data
        }
    
    def extract_symbols_from_request(self, request: str) -> List[str]:
        """Extract stock symbols from user request"""
        # Common stock symbols to look for
        common_symbols = {
            'apple': 'AAPL', 'aapl': 'AAPL',
            'microsoft': 'MSFT', 'msft': 'MSFT',
            'google': 'GOOGL', 'googl': 'GOOGL',
            'amazon': 'AMZN', 'amzn': 'AMZN',
            'tesla': 'TSLA', 'tsla': 'TSLA',
            'nvidia': 'NVDA', 'nvda': 'NVDA',
            'meta': 'META', 'meta': 'META',
            'netflix': 'NFLX', 'nflx': 'NFLX'
        }
        
        symbols = []
        words = request.upper().split()
        
        # Look for explicit symbols (3-5 letter uppercase)
        for word in words:
            if 3 <= len(word) <= 5 and word.isalpha() and word.isupper():
                symbols.append(word)
        
        # Look for company names
        request_lower = request.lower()
        for name, symbol in common_symbols.items():
            if name in request_lower and symbol not in symbols:
                symbols.append(symbol)
        
        return symbols
    
    def generate_technical_analysis(self, stock_data: Dict) -> str:
        """Generate technical analysis using AI"""
        hist = stock_data['history']
        
        # Calculate basic metrics
        current_price = hist['Close'].iloc[-1]
        prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100
        
        # Simple moving averages
        sma_20 = hist['Close'].tail(20).mean() if len(hist) >= 20 else current_price
        sma_50 = hist['Close'].tail(50).mean() if len(hist) >= 50 else current_price
        
        analysis_prompt = f"""
        Provide technical analysis for {stock_data['symbol']} ({stock_data['company_name']}).
        
        Current Metrics:
        - Current Price: ${current_price:.2f}
        - Daily Change: ${price_change:.2f} ({price_change_pct:+.2f}%)
        - 20-day SMA: ${sma_20:.2f}
        - 50-day SMA: ${sma_50:.2f}
        - Price vs SMA20: {'Above' if current_price > sma_20 else 'Below'}
        - Price vs SMA50: {'Above' if current_price > sma_50 else 'Below'}
        
        Data Period: {len(hist)} trading days
        
        Provide insights on:
        - Trend direction
        - Support/resistance levels
        - Momentum indicators
        - Volume analysis (if available)
        - Short-term outlook
        """
        
        return self.groq_client.get_completion(
            analysis_prompt,
            "You are a technical analysis expert. Provide clear, actionable technical insights based on the provided metrics."
        )

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent_initialized" not in st.session_state:
        st.session_state.agent_initialized = False
    
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = "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"
    
    if "stock_system" not in st.session_state:
        st.session_state.stock_system = None
    
    if "use_autogen" not in st.session_state:
        st.session_state.use_autogen = False
    
    if "current_report" not in st.session_state:
        st.session_state.current_report = None

def initialize_system():
    """Initialize the stock report system"""
    try:
        st.session_state.stock_system = StockReportAISystem(st.session_state.groq_api_key)
        st.session_state.agent_initialized = True
        return True
    except Exception as e:
        st.error(f"Failed to initialize system: {str(e)}")
        return False

def display_chat_message(role: str, content: str, timestamp: str, agent_type: str = "assistant"):
    """Display a chat message with appropriate styling"""
    css_classes = {
        "user": "user-message",
        "planner": "planner-message",
        "engineer": "engineer-message",
        "writer": "writer-message",
        "executor": "executor-message",
        "assistant": "assistant-message",
        "system": "system-message"
    }
    
    icons = {
        "user": "👤",
        "planner": "📋",
        "engineer": "⚙️",
        "writer": "✍️",
        "executor": "🖥️",
        "assistant": "🤖",
        "system": "⚙️"
    }
    
    headers = {
        "user": "You",
        "planner": "Report Planner",
        "engineer": "Data Engineer",
        "writer": "Financial Writer",
        "executor": "Code Executor",
        "assistant": "Stock Analyst",
        "system": "System"
    }
    
    css_class = css_classes.get(agent_type, "assistant-message")
    icon = icons.get(agent_type, "🤖")
    header = headers.get(agent_type, "Assistant")
    
    st.markdown(f"""
    <div class="agent-message {css_class}">
        <div class="message-header">{icon} {header} - {timestamp}</div>
        <div class="message-content">{content}</div>
    </div>
    """, unsafe_allow_html=True)

def display_stock_report(report_data: Dict):
    """Display the generated stock report"""
    if not report_data.get("success", False):
        st.error(f"Report generation failed: {report_data.get('response', 'Unknown error')}")
        return
    
    st.markdown("---")
    st.markdown("## 📊 Generated Stock Report")
    
    # Company header
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"{report_data['company_name']} ({report_data['symbol']})")
    with col2:
        if 'data' in report_data and not report_data['data'].empty:
            current_price = report_data['data']['Close'].iloc[-1]
            st.metric("Current Price", f"${current_price:.2f}")
    
    # Display chart
    if 'chart' in report_data:
        st.markdown("### Price Chart")
        st.pyplot(report_data['chart'])
    
    # Display analysis
    if 'analysis' in report_data:
        st.markdown("### Analysis Report")
        st.markdown(report_data['analysis'])
    
    # Technical analysis
    if 'raw_data' in report_data:
        st.markdown("### Technical Analysis")
        technical_analysis = st.session_state.stock_system.generate_technical_analysis(report_data['raw_data'])
        st.markdown(technical_analysis)
    
    # Raw data preview
    if 'data' in report_data and not report_data['data'].empty:
        with st.expander("View Raw Data"):
            st.dataframe(report_data['data'].tail(10))

def main():
    st.markdown('<div class="main-header">📈 AI Stock Report Planning System</div>', unsafe_allow_html=True)
    st.markdown("Multi-Agent System for Comprehensive Stock Analysis")
    
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
        if AUTOGEN_AVAILABLE:
            st.session_state.use_autogen = st.checkbox(
                "Use AutoGen Multi-Agent System", 
                value=st.session_state.use_autogen,
                help="Use specialized agents (Planner, Engineer, Writer, Executor)"
            )
        else:
            st.session_state.use_autogen = False
            st.info("🔧 AutoGen not available. Using direct API mode.")
        
        # Initialize system button
        if not st.session_state.agent_initialized:
            if st.button("🚀 Initialize AI System", use_container_width=True):
                with st.spinner("Initializing AI agents..."):
                    if initialize_system():
                        st.success("AI system initialized successfully!")
                        st.rerun()
        
        st.markdown("---")
        st.header("🎯 Quick Analysis Requests")
        
        # Suggested stock analysis requests
        suggested_requests = [
            "Analyze NVDA stock performance over the past month",
            "Generate a technical report for AAPL",
            "Comprehensive analysis of TSLA stock",
            "MSFT stock report with price predictions",
            "GOOGL technical analysis and outlook",
            "AMZN performance review and investment insights"
        ]
        
        for request in suggested_requests:
            if st.button(request, key=f"quick_{request}"):
                if st.session_state.agent_initialized:
                    # Add user message
                    st.session_state.messages.append({
                        "role": "user",
                        "content": request,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "agent_type": "user"
                    })
                    
                    # Process request
                    with st.spinner("🤖 Planning and generating stock report..."):
                        report_data = st.session_state.stock_system.process_stock_request_simple(request)
                    
                    # Add system response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Generated stock analysis report for {report_data.get('symbol', 'unknown')}",
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "agent_type": "assistant",
                        "report_data": report_data
                    })
                    
                    st.session_state.current_report = report_data
                    st.rerun()
                else:
                    st.warning("Please initialize the AI system first!")
        
        st.markdown("---")
        st.header("🛠️ System Status")
        
        if st.session_state.agent_initialized:
            mode = "AutoGen Multi-Agent" if st.session_state.use_autogen else "Direct API"
            autogen_status = "✅ Available" if AUTOGEN_AVAILABLE else "❌ Not Available"
            
            st.markdown(f"""
            <div class="agent-status">
                ✅ System Initialized
                <br>• Mode: {mode}
                <br>• Model: Llama-3.3-70b-versatile
                <br>• Agents: Planner, Engineer, Writer, Executor
                <br>• AutoGen: {autogen_status}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="padding: 10px; border-radius: 5px; background-color: #fff3cd; border-left: 3px solid #ffc107;">
                ⚠️ System Not Initialized
                <br>Click the button above to initialize
            </div>
            """, unsafe_allow_html=True)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_report = None
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("💬 Stock Analysis Chat")
        
        # Display chat messages
        for message in st.session_state.messages:
            display_chat_message(
                message["role"],
                message["content"],
                message["timestamp"],
                message.get("agent_type", "assistant")
            )
            
            # Display report if available
            if "report_data" in message and message["report_data"]:
                display_stock_report(message["report_data"])
        
        # Chat input
        if st.session_state.agent_initialized:
            user_input = st.chat_input("Enter your stock analysis request (e.g., 'Analyze AAPL stock')...")
            
            if user_input:
                # Add user message
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "agent_type": "user"
                })
                
                # Process request
                with st.spinner("🤖 Multi-agent system processing your request..."):
                    report_data = st.session_state.stock_system.process_stock_request_simple(user_input)
                
                # Add system response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Generated comprehensive stock analysis report for {report_data.get('symbol', 'the requested stock')}",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "agent_type": "assistant",
                    "report_data": report_data
                })
                
                st.session_state.current_report = report_data
                st.rerun()
        else:
            st.info("👆 Please initialize the AI system in the sidebar to start analyzing stocks!")
    
    with col2:
        st.subheader("📈 Agent Activity")
        
        if st.session_state.agent_initialized:
            if st.session_state.use_autogen and AUTOGEN_AVAILABLE:
                st.success("✅ Planner: Analyzing requests")
                st.success("✅ Engineer: Processing data")
                st.success("✅ Writer: Generating reports")
                st.success("✅ Executor: Running analysis")
                st.info("🤝 Multi-agent collaboration enabled")
            else:
                st.success("✅ Direct API Mode: Active")
                st.success("✅ Data Fetcher: Ready")
                st.success("✅ Analyst: Processing")
                st.success("✅ Reporter: Generating")
                st.info("⚡ Using optimized analysis pipeline")
            
            st.markdown("---")
            st.subheader("🔍 Recent Analysis")
            
            # Show recent analyzed symbols
            recent_symbols = set()
            for msg in st.session_state.messages:
                if "report_data" in msg and msg["report_data"].get("success"):
                    symbol = msg["report_data"].get("symbol")
                    if symbol:
                        recent_symbols.add(symbol)
            
            if recent_symbols:
                for symbol in list(recent_symbols)[-5:]:
                    st.write(f"• {symbol}")
            else:
                st.write("No recent analyses")
        else:
            st.warning("System is offline")
            
        st.markdown("---")
        st.subheader("📊 Analysis Metrics")
        
        if st.session_state.current_report and st.session_state.current_report.get("success"):
            report = st.session_state.current_report
            if 'data' in report and not report['data'].empty:
                data = report['data']
                
                # Calculate metrics
                current_price = data['Close'].iloc[-1]
                high_price = data['High'].max()
                low_price = data['Low'].min()
                avg_price = data['Close'].mean()
                volatility = data['Close'].std()
                
                st.metric("Current Price", f"${current_price:.2f}")
                st.metric("Period High", f"${high_price:.2f}")
                st.metric("Period Low", f"${low_price:.2f}")
                st.metric("Average Price", f"${avg_price:.2f}")
                st.metric("Volatility", f"${volatility:.2f}")

if __name__ == "__main__":
    main()
