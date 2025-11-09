import streamlit as st
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple
import groq

# Try to import AutoGen, but provide fallback if it fails
try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent
    AUTOGEN_AVAILABLE = True
except ImportError as e:
    st.warning(f"AutoGen not available: {str(e)}. Using direct Groq API only.")
    AUTOGEN_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="AI Blog Writing Assistant",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #2E8B57, #3CB371);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .agent-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #f0f8ff;
        border-left-color: #4169E1;
    }
    .writer-message {
        background-color: #f0fff0;
        border-left-color: #2E8B57;
    }
    .researcher-message {
        background-color: #fffaf0;
        border-left-color: #FF8C00;
    }
    .editor-message {
        background-color: #f5f0ff;
        border-left-color: #9370DB;
    }
    .seo-message {
        background-color: #fff0f5;
        border-left-color: #FF69B4;
    }
    .message-header {
        font-weight: bold;
        margin-bottom: 5px;
        color: #333;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .message-content {
        color: #555;
        line-height: 1.6;
        white-space: pre-wrap;
    }
    .stButton button {
        width: 100%;
        background-color: #2E8B57;
        color: white;
    }
    .agent-status {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        background-color: #f8f9fa;
        border-left: 3px solid #28a745;
    }
    .blog-preview {
        border: 2px solid #2E8B57;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .progress-section {
        background-color: #f0f8f0;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class GroqClient:
    def __init__(self, api_key: str):
        self.client = groq.Groq(api_key=api_key)
    
    def get_completion(self, message: str, system_message: str = None, model: str = "llama-3.3-70b-versatile") -> str:
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
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
                top_p=1,
                stream=False
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Please try again."

class BlogWritingAIAgent:
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.groq_client = GroqClient(groq_api_key)
        
        # System messages for different blog writing roles
        self.system_messages = {
            "content_researcher": """
            You are an expert content researcher and strategist. Your role is to:
            - Research topics thoroughly
            - Identify key points and angles for blog posts
            - Suggest relevant subtopics and structure
            - Provide factual information and data points
            - Identify target audience and their interests
            
            Be thorough, factual, and strategic in your approach.
            Always cite reliable sources and provide well-researched information.
            """,
            
            "blog_writer": """
            You are a professional blog writer with expertise in creating engaging, well-structured content.
            Your responsibilities include:
            - Writing compelling introductions that hook readers
            - Creating well-organized, readable content
            - Using appropriate tone and style for the target audience
            - Incorporating storytelling elements when relevant
            - Ensuring content flows logically from section to section
            - Using clear, concise language
            
            Write in an engaging, professional tone. Use headings, subheadings, and bullet points appropriately.
            Make complex topics accessible to your target audience.
            """,
            
            "seo_specialist": """
            You are an SEO expert specializing in blog content optimization.
            Your role is to:
            - Suggest relevant keywords and keyphrases
            - Optimize meta descriptions and titles
            - Improve content structure for search engines
            - Suggest internal and external linking strategies
            - Analyze competitor content and identify opportunities
            - Ensure content meets search intent
            
            Provide specific, actionable SEO recommendations.
            Focus on both on-page and technical SEO aspects.
            """,
            
            "editor": """
            You are a professional editor with sharp attention to detail.
            Your responsibilities include:
            - Checking grammar, spelling, and punctuation
            - Improving sentence structure and readability
            - Ensuring consistent tone and style
            - Verifying factual accuracy
            - Improving flow and transitions
            - Eliminating redundancy and wordiness
            - Ensuring the content meets quality standards
            
            Be thorough but constructive in your feedback.
            Suggest specific improvements with explanations.
            """
        }
        
        self.agents_initialized = False
        self.current_blog_state = {
            "topic": "",
            "target_audience": "",
            "tone": "professional",
            "word_count": 1000,
            "keywords": [],
            "outline": "",
            "content": "",
            "seo_analysis": "",
            "editor_feedback": ""
        }
        
        # Initialize AutoGen agents if available
        if AUTOGEN_AVAILABLE:
            try:
                self.initialize_autogen_agents()
                self.agents_initialized = True
            except Exception as e:
                st.warning(f"AutoGen initialization failed: {str(e)}. Using direct API mode.")
                self.agents_initialized = False
    
    def initialize_autogen_agents(self):
        """Initialize AutoGen agents for blog writing"""
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
        
        # Initialize agents
        self.content_researcher = AssistantAgent(
            name="Content_Researcher",
            system_message=self.system_messages["content_researcher"],
            llm_config=self.llm_config,
            human_input_mode="NEVER"
        )
        
        self.blog_writer = AssistantAgent(
            name="Blog_Writer",
            system_message=self.system_messages["blog_writer"],
            llm_config=self.llm_config,
            human_input_mode="NEVER"
        )
        
        self.seo_specialist = AssistantAgent(
            name="SEO_Specialist",
            system_message=self.system_messages["seo_specialist"],
            llm_config=self.llm_config,
            human_input_mode="NEVER"
        )
        
        self.editor = AssistantAgent(
            name="Editor",
            system_message=self.system_messages["editor"],
            llm_config=self.llm_config,
            human_input_mode="NEVER"
        )
        
        self.user_proxy = UserProxyAgent(
            name="User_Proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3,
            code_execution_config=False,
            llm_config=self.llm_config,
            system_message="""You represent the user in conversations with blog writing agents.
            Coordinate between different specialists to create high-quality blog content."""
        )
    
    def research_topic(self, topic: str, target_audience: str = "general") -> Dict:
        """Research a blog topic and provide strategic insights"""
        research_prompt = f"""
        Please research the following blog topic: "{topic}"
        
        Target Audience: {target_audience}
        
        Provide a comprehensive research report including:
        1. Key angles and perspectives
        2. Main subtopics to cover
        3. Important facts and data points
        4. Target audience interests and pain points
        5. Potential sources and references
        6. Current trends related to this topic
        
        Format your response as a structured research report.
        """
        
        research = self.groq_client.get_completion(
            research_prompt,
            self.system_messages["content_researcher"]
        )
        
        return {
            "research": research,
            "topic": topic,
            "target_audience": target_audience,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_outline(self, research_data: Dict, tone: str = "professional", word_count: int = 1000) -> str:
        """Generate a blog outline based on research"""
        outline_prompt = f"""
        Based on the following research, create a detailed blog outline:
        
        Topic: {research_data['topic']}
        Target Audience: {research_data['target_audience']}
        Tone: {tone}
        Estimated Word Count: {word_count}
        
        Research Insights:
        {research_data['research']}
        
        Create a comprehensive outline with:
        - Compelling introduction
        - Main sections with subpoints
        - Logical flow
        - Key takeaways
        - Strong conclusion
        
        Format the outline clearly with headings and bullet points.
        """
        
        return self.groq_client.get_completion(
            outline_prompt,
            self.system_messages["blog_writer"]
        )
    
    def write_blog_content(self, outline: str, research_data: Dict, keywords: List[str] = None) -> str:
        """Write full blog content based on outline and research"""
        keywords_text = ", ".join(keywords) if keywords else "Not specified"
        
        content_prompt = f"""
        Write a complete blog post based on this outline and research:
        
        TOPIC: {research_data['topic']}
        TARGET AUDIENCE: {research_data['target_audience']}
        KEYWORDS: {keywords_text}
        
        OUTLINE:
        {outline}
        
        RESEARCH INSIGHTS:
        {research_data['research']}
        
        Please write the full blog post with:
        - Engaging introduction
        - Well-structured body paragraphs
        - Clear headings and subheadings
        - Bullet points or numbered lists where appropriate
        - Compelling conclusion
        - Natural incorporation of keywords
        
        Write in a {research_data.get('tone', 'professional')} tone.
        """
        
        return self.groq_client.get_completion(
            content_prompt,
            self.system_messages["blog_writer"]
        )
    
    def analyze_seo(self, content: str, topic: str, target_keywords: List[str] = None) -> str:
        """Analyze and optimize content for SEO"""
        seo_prompt = f"""
        Analyze the following blog content for SEO and provide optimization recommendations:
        
        TOPIC: {topic}
        TARGET KEYWORDS: {', '.join(target_keywords) if target_keywords else 'Not specified'}
        
        CONTENT:
        {content}
        
        Provide a comprehensive SEO analysis including:
        1. Keyword optimization suggestions
        2. Meta description recommendations
        3. Title tag optimization
        4. Internal linking opportunities
        5. Readability improvements
        6. Competitor analysis insights
        7. Technical SEO considerations
        
        Be specific and provide actionable recommendations.
        """
        
        return self.groq_client.get_completion(
            seo_prompt,
            self.system_messages["seo_specialist"]
        )
    
    def edit_content(self, content: str, topic: str, tone: str = "professional") -> Tuple[str, str]:
        """Edit and improve blog content"""
        edit_prompt = f"""
        Please edit and improve the following blog content:
        
        TOPIC: {topic}
        DESIRED TONE: {tone}
        
        CONTENT TO EDIT:
        {content}
        
        Provide:
        1. The improved version of the content
        2. Specific feedback on what was changed and why
        
        Focus on:
        - Grammar and spelling
        - Sentence structure and flow
        - Tone consistency
        - Readability improvements
        - Eliminating redundancy
        - Enhancing clarity
        
        Separate your response into two clear sections: "EDITED CONTENT" and "EDITING FEEDBACK".
        """
        
        editing_result = self.groq_client.get_completion(
            edit_prompt,
            self.system_messages["editor"]
        )
        
        # Simple parsing to separate content from feedback
        if "EDITED CONTENT" in editing_result and "EDITING FEEDBACK" in editing_result:
            parts = editing_result.split("EDITING FEEDBACK")
            edited_content = parts[0].replace("EDITED CONTENT", "").strip()
            feedback = "EDITING FEEDBACK" + parts[1]
        else:
            edited_content = editing_result
            feedback = "Comprehensive editing completed. Review the changes above."
        
        return edited_content, feedback
    
    def generate_keywords(self, topic: str, target_audience: str) -> List[str]:
        """Generate relevant keywords for the blog topic"""
        keyword_prompt = f"""
        Generate 10-15 relevant keywords and keyphrases for a blog about:
        
        TOPIC: {topic}
        TARGET AUDIENCE: {target_audience}
        
        Include:
        - Primary keywords (1-2 words)
        - Long-tail keywords (3-5 words)
        - Question-based keywords
        - LSI (Latent Semantic Indexing) keywords
        
        Format as a simple comma-separated list.
        """
        
        keywords_text = self.groq_client.get_completion(
            keyword_prompt,
            self.system_messages["seo_specialist"]
        )
        
        # Parse comma-separated keywords
        keywords = [kw.strip() for kw in keywords_text.split(",") if kw.strip()]
        return keywords[:15]  # Limit to 15 keywords

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent_initialized" not in st.session_state:
        st.session_state.agent_initialized = False
    
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = "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"
    
    if "blog_agent" not in st.session_state:
        st.session_state.blog_agent = None
    
    if "use_autogen" not in st.session_state:
        st.session_state.use_autogen = False
    
    if "current_blog" not in st.session_state:
        st.session_state.current_blog = {
            "topic": "",
            "target_audience": "general",
            "tone": "professional",
            "word_count": 1000,
            "keywords": [],
            "research": "",
            "outline": "",
            "content": "",
            "seo_analysis": "",
            "edited_content": "",
            "editor_feedback": "",
            "progress": 0
        }
    
    if "writing_stage" not in st.session_state:
        st.session_state.writing_stage = "setup"  # setup, research, outline, writing, seo, editing, complete

def initialize_agent():
    """Initialize the AI agent"""
    try:
        st.session_state.blog_agent = BlogWritingAIAgent(st.session_state.groq_api_key)
        st.session_state.agent_initialized = True
        return True
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        return False

def display_agent_message(role: str, content: str, timestamp: str, agent_type: str = "assistant"):
    """Display a message from an agent with appropriate styling"""
    if role == "user":
        css_class = "user-message"
        icon = "👤"
        header = "You"
    elif agent_type == "researcher":
        css_class = "researcher-message"
        icon = "🔍"
        header = "Content Researcher"
    elif agent_type == "writer":
        css_class = "writer-message"
        icon = "✍️"
        header = "Blog Writer"
    elif agent_type == "seo":
        css_class = "seo-message"
        icon = "📈"
        header = "SEO Specialist"
    elif agent_type == "editor":
        css_class = "editor-message"
        icon = "📝"
        header = "Editor"
    else:
        css_class = "assistant-message"
        icon = "🤖"
        header = "Assistant"
    
    st.markdown(f"""
    <div class="agent-message {css_class}">
        <div class="message-header">{icon} {header} - {timestamp}</div>
        <div class="message-content">{content}</div>
    </div>
    """, unsafe_allow_html=True)

def update_progress():
    """Update blog writing progress based on current stage"""
    stage_weights = {
        "setup": 10,
        "research": 20,
        "outline": 30,
        "writing": 60,
        "seo": 80,
        "editing": 95,
        "complete": 100
    }
    st.session_state.current_blog["progress"] = stage_weights.get(st.session_state.writing_stage, 0)

def main():
    st.markdown('<div class="main-header">🤖 AI Blog Writing Assistant</div>', unsafe_allow_html=True)
    st.markdown("Powered by Llama-3.3-70b-versatile (Groq) - Multi-Agent System")
    
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
                "Use AutoGen Multi-Agent System (Experimental)", 
                value=st.session_state.use_autogen,
                help="Use multiple specialized agents for blog writing"
            )
        else:
            st.session_state.use_autogen = False
            st.info("🔧 AutoGen not available. Using direct API mode.")
        
        # Initialize agent button
        if not st.session_state.agent_initialized:
            if st.button("🚀 Initialize Blog Writing Agents", use_container_width=True):
                with st.spinner("Initializing AI agents..."):
                    if initialize_agent():
                        st.success("Blog writing agents initialized successfully!")
                        st.rerun()
        
        st.markdown("---")
        st.header("📝 Blog Setup")
        
        # Blog configuration
        st.session_state.current_blog["topic"] = st.text_input(
            "Blog Topic",
            value=st.session_state.current_blog["topic"],
            placeholder="Enter your blog topic here..."
        )
        
        st.session_state.current_blog["target_audience"] = st.selectbox(
            "Target Audience",
            ["general", "technical", "business", "beginners", "experts", "academic"],
            index=0
        )
        
        st.session_state.current_blog["tone"] = st.selectbox(
            "Writing Tone",
            ["professional", "conversational", "formal", "friendly", "authoritative", "humorous"],
            index=0
        )
        
        st.session_state.current_blog["word_count"] = st.slider(
            "Target Word Count",
            min_value=500,
            max_value=3000,
            value=1000,
            step=100
        )
        
        # Quick start buttons
        st.markdown("---")
        st.header("🎯 Quick Actions")
        
        if st.session_state.agent_initialized and st.session_state.current_blog["topic"]:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔍 Research Topic", use_container_width=True):
                    with st.spinner("Researching topic..."):
                        research_data = st.session_state.blog_agent.research_topic(
                            st.session_state.current_blog["topic"],
                            st.session_state.current_blog["target_audience"]
                        )
                        st.session_state.current_blog["research"] = research_data["research"]
                        st.session_state.current_blog["keywords"] = st.session_state.blog_agent.generate_keywords(
                            st.session_state.current_blog["topic"],
                            st.session_state.current_blog["target_audience"]
                        )
                        st.session_state.writing_stage = "research"
                        update_progress()
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Research completed for topic: '{st.session_state.current_blog['topic']}'\n\n{research_data['research']}",
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "agent_type": "researcher"
                        })
                        st.rerun()
            
            with col2:
                if st.button("📋 Generate Outline", use_container_width=True) and st.session_state.current_blog["research"]:
                    with st.spinner("Creating outline..."):
                        outline = st.session_state.blog_agent.generate_outline(
                            {"topic": st.session_state.current_blog["topic"], 
                             "target_audience": st.session_state.current_blog["target_audience"],
                             "research": st.session_state.current_blog["research"]},
                            st.session_state.current_blog["tone"],
                            st.session_state.current_blog["word_count"]
                        )
                        st.session_state.current_blog["outline"] = outline
                        st.session_state.writing_stage = "outline"
                        update_progress()
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Outline generated:\n\n{outline}",
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "agent_type": "writer"
                        })
                        st.rerun()
            
            if st.button("✍️ Write Full Blog", use_container_width=True) and st.session_state.current_blog["outline"]:
                with st.spinner("Writing blog content..."):
                    content = st.session_state.blog_agent.write_blog_content(
                        st.session_state.current_blog["outline"],
                        {
                            "topic": st.session_state.current_blog["topic"],
                            "target_audience": st.session_state.current_blog["target_audience"],
                            "research": st.session_state.current_blog["research"],
                            "tone": st.session_state.current_blog["tone"]
                        },
                        st.session_state.current_blog["keywords"]
                    )
                    st.session_state.current_blog["content"] = content
                    st.session_state.writing_stage = "writing"
                    update_progress()
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Blog content written:\n\n{content}",
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "agent_type": "writer"
                    })
                    st.rerun()
            
            col3, col4 = st.columns(2)
            
            with col3:
                if st.button("📈 SEO Analysis", use_container_width=True) and st.session_state.current_blog["content"]:
                    with st.spinner("Analyzing SEO..."):
                        seo_analysis = st.session_state.blog_agent.analyze_seo(
                            st.session_state.current_blog["content"],
                            st.session_state.current_blog["topic"],
                            st.session_state.current_blog["keywords"]
                        )
                        st.session_state.current_blog["seo_analysis"] = seo_analysis
                        st.session_state.writing_stage = "seo"
                        update_progress()
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"SEO Analysis:\n\n{seo_analysis}",
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "agent_type": "seo"
                        })
                        st.rerun()
            
            with col4:
                if st.button("📝 Edit Content", use_container_width=True) and st.session_state.current_blog["content"]:
                    with st.spinner("Editing content..."):
                        edited_content, feedback = st.session_state.blog_agent.edit_content(
                            st.session_state.current_blog["content"],
                            st.session_state.current_blog["topic"],
                            st.session_state.current_blog["tone"]
                        )
                        st.session_state.current_blog["edited_content"] = edited_content
                        st.session_state.current_blog["editor_feedback"] = feedback
                        st.session_state.writing_stage = "editing"
                        update_progress()
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Editing completed:\n\n{feedback}\n\n---\n\nEDITED CONTENT:\n\n{edited_content}",
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "agent_type": "editor"
                        })
                        st.rerun()
        
        st.markdown("---")
        st.header("🛠️ Agent Status")
        
        if st.session_state.agent_initialized:
            mode = "AutoGen Multi-Agent" if st.session_state.use_autogen else "Direct API"
            
            st.markdown(f"""
            <div class="agent-status">
                ✅ Agents Initialized
                <br>• Mode: {mode}
                <br>• Model: Llama-3.3-70b-versatile
                <br>• Current Stage: {st.session_state.writing_stage.title()}
            </div>
            """, unsafe_allow_html=True)
            
            # Agent roles status
            st.subheader("👥 Active Agents")
            st.success("🔍 Content Researcher: Ready")
            st.success("✍️ Blog Writer: Ready")
            st.success("📈 SEO Specialist: Ready")
            st.success("📝 Editor: Ready")
        else:
            st.warning("⚠️ Agents Not Initialized")
        
        # Clear buttons
        col5, col6 = st.columns(2)
        with col5:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col6:
            if st.button("🔄 Reset Blog", use_container_width=True):
                st.session_state.current_blog = {
                    "topic": st.session_state.current_blog["topic"],
                    "target_audience": "general",
                    "tone": "professional",
                    "word_count": 1000,
                    "keywords": [],
                    "research": "",
                    "outline": "",
                    "content": "",
                    "seo_analysis": "",
                    "edited_content": "",
                    "editor_feedback": "",
                    "progress": 0
                }
                st.session_state.writing_stage = "setup"
                st.rerun()
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("💬 Blog Writing Process")
        
        # Display progress
        st.markdown(f"""
        <div class="progress-section">
            <h4>📊 Writing Progress: {st.session_state.current_blog['progress']}%</h4>
            <div style="margin: 10px 0;">
        """, unsafe_allow_html=True)
        st.progress(st.session_state.current_blog["progress"] / 100)
        st.markdown(f"Current Stage: **{st.session_state.writing_stage.title()}**</div>", unsafe_allow_html=True)
        
        # Display chat messages
        for message in st.session_state.messages:
            display_agent_message(
                message["role"],
                message["content"],
                message["timestamp"],
                message.get("agent_type", "assistant")
            )
        
        # Chat input for custom requests
        if st.session_state.agent_initialized:
            user_input = st.chat_input("Ask about blog writing or request specific help...")
            
            if user_input:
                # Add user message
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "agent_type": "user"
                })
                
                # Process with appropriate agent based on content
                with st.spinner("🤖 Processing your request..."):
                    if any(word in user_input.lower() for word in ["research", "data", "facts"]):
                        response = st.session_state.blog_agent.groq_client.get_completion(
                            user_input,
                            st.session_state.blog_agent.system_messages["content_researcher"]
                        )
                        agent_type = "researcher"
                    elif any(word in user_input.lower() for word in ["write", "content", "blog", "article"]):
                        response = st.session_state.blog_agent.groq_client.get_completion(
                            user_input,
                            st.session_state.blog_agent.system_messages["blog_writer"]
                        )
                        agent_type = "writer"
                    elif any(word in user_input.lower() for word in ["seo", "keyword", "optimize", "search"]):
                        response = st.session_state.blog_agent.groq_client.get_completion(
                            user_input,
                            st.session_state.blog_agent.system_messages["seo_specialist"]
                        )
                        agent_type = "seo"
                    elif any(word in user_input.lower() for word in ["edit", "grammar", "improve", "feedback"]):
                        response = st.session_state.blog_agent.groq_client.get_completion(
                            user_input,
                            st.session_state.blog_agent.system_messages["editor"]
                        )
                        agent_type = "editor"
                    else:
                        # Default to blog writer
                        response = st.session_state.blog_agent.groq_client.get_completion(
                            user_input,
                            st.session_state.blog_agent.system_messages["blog_writer"]
                        )
                        agent_type = "writer"
                
                # Add assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "agent_type": agent_type
                })
                
                st.rerun()
        else:
            st.info("👆 Please initialize the blog writing agents in the sidebar to get started!")
    
    with col2:
        st.subheader("📄 Blog Preview")
        
        if st.session_state.current_blog["edited_content"]:
            content_to_show = st.session_state.current_blog["edited_content"]
            st.success("✅ Showing edited version")
        elif st.session_state.current_blog["content"]:
            content_to_show = st.session_state.current_blog["content"]
            st.info("📝 Showing original version")
        else:
            content_to_show = "Your blog content will appear here as you progress through the writing stages."
            st.warning("⏳ Blog not started yet")
        
        st.markdown(f'<div class="blog-preview">{content_to_show}</div>', unsafe_allow_html=True)
        
        # Download button for final content
        if st.session_state.current_blog["content"]:
            st.download_button(
                label="📥 Download Blog",
                data=content_to_show,
                file_name=f"blog_{st.session_state.current_blog['topic'].replace(' ', '_').lower()}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        # Blog metrics
        st.subheader("📊 Blog Metrics")
        if st.session_state.current_blog["content"]:
            word_count = len(content_to_show.split())
            reading_time = max(1, word_count // 200)  # Assuming 200 words per minute
            
            st.metric("Word Count", word_count)
            st.metric("Estimated Reading Time", f"{reading_time} min")
            
            if st.session_state.current_blog["keywords"]:
                st.write("**Keywords:**")
                for keyword in st.session_state.current_blog["keywords"][:5]:
                    st.write(f"• {keyword}")
        
        # Quick tips
        st.subheader("💡 Writing Tips")
        tips = [
            "Start with thorough research to build authority",
            "Use your target audience's language and address their pain points",
            "Incorporate keywords naturally throughout the content",
            "Break up long paragraphs for better readability",
            "Use headings and subheadings to structure your content",
            "Include a clear call-to-action in your conclusion"
        ]
        
        for tip in tips:
            st.write(f"• {tip}")

if __name__ == "__main__":
    main()
