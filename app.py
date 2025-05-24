import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
import time
from groq import Groq
import os
from typing import Optional, Tuple, Dict, Any
import logging
from dotenv import load_dotenv
from firecrawl import FirecrawlApp

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Blog Post Summarizer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

class BlogSummarizer:
    def __init__(self, groq_api_key: str, firecrawl_api_key: Optional[str] = None):
        """Initialize the BlogSummarizer with API keys."""
        self.groq_client = Groq(api_key=groq_api_key)
        self.firecrawl_client = FirecrawlApp(api_key=firecrawl_api_key) if firecrawl_api_key else None
        
        # Fallback session for basic scraping
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def is_valid_url(self, url: str) -> bool:
        """Validate URL format."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def extract_with_firecrawl(self, url: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract content using Firecrawl API."""
        try:
            # Scrape the URL with Firecrawl
            scrape_result = self.firecrawl_client.scrape_url(
                url,
                params={
                    'formats': ['markdown', 'html'],
                    'onlyMainContent': True,
                    'includeTags': ['title', 'h1', 'h2', 'h3', 'p', 'article'],
                    'excludeTags': ['nav', 'footer', 'header', 'aside', 'script', 'style'],
                    'waitFor': 3000,  # Wait for dynamic content
                    'timeout': 30000  # 30 second timeout
                }
            )
            
            if not scrape_result.get('success', False):
                return None, None, f"Firecrawl failed to scrape the URL: {scrape_result.get('error', 'Unknown error')}"
            
            data = scrape_result.get('data', {})
            
            # Extract title and content
            title = data.get('title') or data.get('metadata', {}).get('title')
            
            # Prefer markdown content, fallback to cleaned HTML
            content = data.get('markdown')
            if not content:
                content = data.get('content')
            
            if not content:
                return None, None, "No content could be extracted from the page"
            
            # Clean markdown content if needed
            if content:
                # Remove excessive whitespace and newlines
                content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
                content = re.sub(r'\s+', ' ', content)
                content = content.strip()
            
            return title, content, None
            
        except Exception as e:
            logger.error(f"Extraction error for {url}: {str(e)}")
            return None, None, f"Firecrawl error: {str(e)}"
    
    def extract_with_fallback(self, url: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Fallback content extraction using BeautifulSoup."""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
                element.decompose()
            
            # Extract title
            title = None
            title_selectors = [
                'title', 'h1', '.post-title', '.entry-title', '.article-title',
                '[property="og:title"]', '.headline', '.page-title'
            ]
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    title = title_elem.get_text().strip() if hasattr(title_elem, 'get_text') else title_elem.get('content', '').strip()
                    if title:
                        break
            
            # Extract main content
            content = None
            content_selectors = [
                'article', '.post-content', '.entry-content', '.article-content',
                '.content', 'main', '.post', '.blog-post', '[role="main"]',
                '.story-body', '.article-body', '.post-body'
            ]
            
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content = content_elem.get_text(separator=' ', strip=True)
                    if len(content) > 200:  # Ensure substantial content
                        break
            
            # Final fallback to body content
            if not content or len(content) < 200:
                body = soup.find('body')
                if body:
                    content = body.get_text(separator=' ', strip=True)
            
            # Clean up content
            if content:
                content = re.sub(r'\s+', ' ', content)
                content = re.sub(r'(Skip to content|Copyright|All rights reserved|Privacy Policy|Terms of Service).*', '', content, flags=re.IGNORECASE)
                content = content.strip()
            
            return title, content, None
            
        except requests.exceptions.Timeout:
            return None, None, "Request timeout - the website took too long to respond"
        except requests.exceptions.ConnectionError:
            return None, None, "Connection error - unable to reach the website"
        except requests.exceptions.HTTPError as e:
            return None, None, f"HTTP error {e.response.status_code} - {e.response.reason}"
        except Exception as e:
            logger.error(f"Fallback extraction error for {url}: {str(e)}")
            return None, None, f"Error extracting content: {str(e)}"
    
    def extract_content(self, url: str) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
        """Extract content using the best available method."""
        method_used = "unknown"
        
        # Try Firecrawl first if available
        if self.firecrawl_client:
            method_used = "Firecrawl"
            title, content, error = self.extract_with_firecrawl(url)
            if not error and content and len(content.strip()) > 100:
                return title, content, None, method_used
        
        # Fallback to basic scraping
        method_used = "Basic Scraping"
        title, content, error = self.extract_with_fallback(url)
        return title, content, error, method_used
    
    def summarize_content(self, title: str, content: str, summary_length: str = "medium") -> Tuple[Optional[str], Optional[str]]:
        """Summarize the extracted content using Groq."""
        try:
            # Truncate content if too long
            max_content_length = 15000  # More generous limit for better summaries
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            
            # Define summary length instructions
            length_instructions = {
                "short": "Provide a concise 2-3 sentence summary highlighting only the most important points.",
                "medium": "Provide a comprehensive summary in 1-2 paragraphs (4-6 sentences) covering the main points and key insights.",
                "long": "Provide a detailed summary in 2-3 paragraphs (6-10 sentences) covering main points, supporting details, and key takeaways."
            }
            
            length_instruction = length_instructions.get(summary_length, length_instructions["medium"])
            
            prompt = f"""Please summarize the following blog post/article. {length_instruction}

Title: {title or "Article"}

Content: {content}

Focus on:
- Main arguments or points
- Key insights or findings  
- Important conclusions or takeaways
- Actionable information if present

Summary:"""
            
            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert content summarizer. Create clear, informative summaries that capture the essence of articles while being engaging and easy to understand. Focus on the most valuable information for readers."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama3-8b-8192",
                temperature=0.5
            )
            
            summary = response.choices[0].message.content.strip()
            return summary, None
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return None, f"Error generating summary: {str(e)}"

def main():
    st.title("📚 Advanced Blog Post Summarizer")
    st.markdown("Transform lengthy blog posts into concise, informative summaries using AI-powered content extraction.")
    
    # Get API keys from environment
    groq_api_key = os.getenv("GROQ_API_KEY")
    firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
    
    # Check if required API keys are configured
    if not groq_api_key:
        st.error("🔑 Groq API key not configured. Please contact the administrator.")
        st.info("Administrator: Add your Groq API key to the `.env` file as `GROQ_API_KEY=your_key_here`")
        return
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Settings")
        
        # Show extraction method status
        
        
        # Summary length selection
        summary_length = st.selectbox(
            "Summary Length",
            ["short", "medium", "long"],
            index=1,
            help="Choose how detailed you want the summary to be"
        )
        
        st.markdown("---")
        st.markdown("### 📖 How to Use")
        st.markdown("1. Paste any blog post or article URL")
        st.markdown("2. Choose your preferred summary length")
        st.markdown("3. Click 'Summarize' and wait")
        st.markdown("4. Get your AI-powered summary!")
        
        st.markdown("---")
        st.markdown("### 💡 Works Great With")
        st.markdown("• Blog posts & articles")
        st.markdown("• News websites")
        st.markdown("• Technical documentation")
        st.markdown("• Medium & Substack posts")
        if firecrawl_api_key:
            st.markdown("• JavaScript-heavy sites")
            st.markdown("• Protected content")
    
    # Initialize summarizer
    try:
        summarizer = BlogSummarizer(groq_api_key, firecrawl_api_key)
    except Exception as e:
        st.error(f"Error initializing summarizer: {str(e)}")
        return
    
    # URL input
    url = st.text_input(
        "📝 Blog Post or Article URL",
        placeholder="https://example.com/blog-post-to-summarize",
        help="Enter the full URL of the blog post or article you want to summarize"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        summarize_button = st.button("🚀 Summarize Article", type="primary", use_container_width=True)
    
    if summarize_button and url:
        if not summarizer.is_valid_url(url):
            st.error("❌ Please enter a valid URL (including http:// or https://)")
            return
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Extract content
        status_text.text("🔍 Extracting content from the article...")
        progress_bar.progress(25)
        
        title, content, error, method = summarizer.extract_content(url)
        
        if error:
            st.error(f"❌ {error}")
            if not firecrawl_api_key and "timeout" not in error.lower():
                st.info("💡 **Tip**: Many extraction issues can be resolved with Firecrawl integration. Contact your administrator for premium extraction capabilities.")
            return
        
        if not content or len(content.strip()) < 100:
            st.error("❌ Unable to extract sufficient content from this URL. The page might be behind a paywall, require JavaScript, or have content protection.")
            if not firecrawl_api_key:
                st.info("💡 **Tip**: Firecrawl can often handle protected and JavaScript-heavy sites. Contact your administrator about upgrading.")
            return
        
        progress_bar.progress(60)
        status_text.text("🤖 Generating AI summary...")
        
        # Generate summary
        summary, error = summarizer.summarize_content(title or "Untitled Article", content, summary_length)
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        
        if error:
            st.error(f"❌ {error}")
            return
        
        # Display results
        st.success("✅ Summary generated successfully!")
        
        # Display title if available
        if title:
            st.subheader(f"📄 {title}")
        
        # Display summary
        st.markdown("### 📝 Summary")
        st.markdown(summary)
        
        # Display additional info
        with st.expander("ℹ️ Processing Details"):
            st.write(f"**Source URL:** {url}")
            st.write(f"**Extraction Method:** {method}")
            st.write(f"**Content Length:** {len(content):,} characters")
            st.write(f"**Summary Length Setting:** {summary_length.title()}")
            
            if title:
                st.write(f"**Original Title:** {title}")
        
        # Copy summary section
        st.markdown("### 📋 Copy Summary")
        st.code(summary, language=None)
    
    elif summarize_button and not url:
        st.warning("⚠️ Please enter a blog post URL to summarize.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Powered by Groq AI" + (" + Firecrawl" if firecrawl_api_key else "") + " | Built with ❤️ using Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()