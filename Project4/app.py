import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
from groq import Groq

# Set page configuration
st.set_page_config(
    page_title="AI Stock Report Generator",
    page_icon="📈",
    layout="wide"
)

# Initialize Groq client
@st.cache_resource
def get_groq_client():
    return Groq(api_key="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")

def get_stock_data(symbol, period="3mo"):
    """Get stock data with caching"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        return hist
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return None

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

def generate_stock_report(symbol, stock_data, period_display):
    """Generate stock report using Groq with pre-computed data"""
    
    # Calculate key statistics
    current_price = stock_data['Close'].iloc[-1]
    start_price = stock_data['Close'].iloc[0]
    high_price = stock_data['High'].max()
    low_price = stock_data['Low'].min()
    pct_change = ((current_price - start_price) / start_price) * 100
    avg_volume = stock_data['Volume'].mean()
    
    # Calculate moving averages
    stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
    
    # Prepare the prompt with actual data
    prompt = f"""
    Write a comprehensive blog post about the stock price performance of {symbol} in the {period_display}. 
    Today's date is 2024-07-26.
    
    Here are the actual stock statistics for {symbol}:
    - Current Price: ${current_price:.2f}
    - Starting Price (3 months ago): ${start_price:.2f}
    - Highest Price: ${high_price:.2f}
    - Lowest Price: ${low_price:.2f}
    - Percentage Change: {pct_change:.2f}%
    - Average Volume: {avg_volume:,.0f}
    
    The stock has shown a {'positive' if pct_change > 0 else 'negative'} trend over the period.
    
    Please write a professional blog post that includes:
    1. Introduction to the company and its industry
    2. Key statistics and performance overview
    3. Trend analysis with technical insights
    4. Market context and industry comparison
    5. Conclusion and future outlook
    6. Investment considerations
    
    Format the response in markdown with appropriate headings and sections.
    Make it engaging and informative for investors.
    """
    
    client = get_groq_client()
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2048,
            top_p=1,
            stream=False,
            stop=None
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        return f"Error generating report: {str(e)}"

def generate_fallback_report(symbol, stock_data, period_display):
    """Generate a fallback report if API fails"""
    current_price = stock_data['Close'].iloc[-1]
    start_price = stock_data['Close'].iloc[0]
    high_price = stock_data['High'].max()
    low_price = stock_data['Low'].min()
    pct_change = ((current_price - start_price) / start_price) * 100
    avg_volume = stock_data['Volume'].mean()
    
    report = f"""
# {symbol} Stock Performance Analysis - {period_display}

## 📊 Key Statistics

| Metric | Value |
|--------|-------|
| **Current Price** | ${current_price:.2f} |
| **Starting Price** | ${start_price:.2f} |
| **Highest Price** | ${high_price:.2f} |
| **Lowest Price** | ${low_price:.2f} |
| **Total Change** | {pct_change:.2f}% |
| **Average Volume** | {avg_volume:,.0f} |

## 📈 Performance Summary

The stock of {symbol} has shown **{'positive' if pct_change > 0 else 'negative'} performance** during the {period_display}, with a total change of **{pct_change:.2f}%**.

### Trend Analysis
- **Overall Trend**: {'Bullish' if pct_change > 0 else 'Bearish'}
- **Price Range**: ${low_price:.2f} - ${high_price:.2f}
- **Volatility**: Moderate to High

## 🏢 Company Overview
{symbol} operates in the automotive/technology sector with a focus on innovation and market expansion.

## 🌐 Market Context
The stock performance should be considered in the context of:
- Overall market conditions
- Industry-specific trends
- Economic factors affecting the sector

## 💡 Investment Considerations
- **Strengths**: Market position, innovation pipeline
- **Risks**: Market volatility, competition
- **Opportunities**: Industry growth, expansion plans

## 🎯 Conclusion
Based on the {period_display} performance analysis, {symbol} demonstrates {'strong potential' if pct_change > 0 else 'areas for improvement'}. 

*Note: This analysis is automated. Consult with financial advisors before making investment decisions.*
"""
    return report

def main():
    st.title("🚀 Quick Stock Report Generator")
    st.markdown("Fast AI-powered stock analysis with minimal API calls")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        stock_symbol = st.text_input(
            "Stock Symbol", 
            value="TTM",
            help="Enter the stock symbol (e.g., TTM for Tata Motors, NVDA, AAPL)"
        )
        
        period_options = {
            "1mo": "Past Month",
            "3mo": "Past 3 Months", 
            "6mo": "Past 6 Months",
            "1y": "Past Year"
        }
        
        period = st.selectbox(
            "Analysis Period",
            list(period_options.keys()),
            index=1,
            format_func=lambda x: period_options[x]
        )
        
        st.markdown("---")
        st.markdown("### 🎯 Features")
        st.markdown("""
        - 📊 Real stock data
        - 🤖 AI analysis
        - 📈 Interactive charts
        - ⚡ Fast generation
        """)
        
        generate_btn = st.button("🚀 Generate Report", type="primary", use_container_width=True)
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if generate_btn and stock_symbol:
            # Get period display name
            period_display = period_options[period]
            
            # Fetch stock data first
            with st.spinner("📊 Fetching stock data..."):
                stock_data = get_stock_data(stock_symbol, period)
                
                if stock_data is None or stock_data.empty:
                    st.error(f"❌ Could not fetch data for {stock_symbol}. Please check the symbol and try again.")
                    return
            
            # Display stock metrics and chart
            st.subheader("📊 Stock Overview")
            
            # Key metrics
            col1a, col2a, col3a, col4a = st.columns(4)
            current_price = stock_data['Close'].iloc[-1]
            start_price = stock_data['Close'].iloc[0]
            pct_change = ((current_price - start_price) / start_price) * 100
            
            with col1a:
                st.metric(
                    "Current Price", 
                    f"${current_price:.2f}",
                    f"{pct_change:.2f}%"
                )
            
            with col2a:
                st.metric("High", f"${stock_data['High'].max():.2f}")
            
            with col3a:
                st.metric("Low", f"${stock_data['Low'].min():.2f}")
            
            with col4a:
                st.metric("Avg Volume", f"{stock_data['Volume'].mean():,.0f}")
            
            # Display chart
            st.subheader("📈 Price Chart")
            chart_base64 = create_stock_chart(stock_data, stock_symbol)
            st.image(f"data:image/png;base64,{chart_base64}")
            
            # Generate report
            with st.spinner("🤖 Generating AI analysis..."):
                try:
                    report = generate_stock_report(stock_symbol, stock_data, period_display)
                    
                    if report.startswith("Error"):
                        st.warning("⚠️ Using fallback report due to API limits")
                        report = generate_fallback_report(stock_symbol, stock_data, period_display)
                    
                    st.subheader("📄 AI Analysis Report")
                    st.markdown(report)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Report",
                        data=report,
                        file_name=f"{stock_symbol}_stock_report.md",
                        mime="text/markdown"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Generating fallback report...")
                    report = generate_fallback_report(stock_symbol, stock_data, period_display)
                    st.markdown(report)
        
        elif not stock_symbol:
            st.warning("⚠️ Please enter a stock symbol")
        
        else:
            # Welcome message
            st.subheader("🎯 Quick Stock Analysis")
            st.markdown("""
            ### Get instant stock reports with AI!
            
            **How it works:**
            1. Enter a stock symbol
            2. Select analysis period
            3. Get instant AI analysis
            
            **Why this is better:**
            - ⚡ **Fast**: Single API call
            - 💰 **Efficient**: No rate limit issues
            - 📊 **Accurate**: Real stock data
            - 🤖 **Smart**: AI-powered insights
            
            **Popular Symbols:**
            - **TTM**: Tata Motors
            - **NVDA**: NVIDIA
            - **AAPL**: Apple
            - **TSLA**: Tesla
            - **RELIANCE.NS**: Reliance (NSE)
            """)
    
    with col2:
        st.subheader("⚡ Quick Stats")
        
        if generate_btn and stock_symbol:
            if stock_data is not None and not stock_data.empty:
                stats = {
                    "Data Points": len(stock_data),
                    "Period": f"{period_display}",
                    "Volatility": f"{(stock_data['High'] - stock_data['Low']).mean():.2f}",
                    "Trend": "📈 Bullish" if pct_change > 0 else "📉 Bearish"
                }
                
                for key, value in stats.items():
                    st.metric(key, value)
                
                st.markdown("---")
                st.markdown("### 💡 Tips")
                st.markdown("""
                - Check symbol format
                - Use major exchanges
                - Consider market hours
                """)

if __name__ == "__main__":
    main()
