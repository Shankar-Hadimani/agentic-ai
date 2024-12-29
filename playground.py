from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.playground import Playground, serve_playground_app
from dotenv import load_dotenv, find_dotenv
import openai
import os
import phi

# Load environment variables
dotenv_path = find_dotenv()
if dotenv_path:
    print(f"Loading .env file from: {dotenv_path}")
    load_dotenv(dotenv_path)
else:
    print("No .env file found.")

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

# Set PHI API key
phi.api = os.getenv("PHI_API_KEY")
if not phi.api:
    raise ValueError("The PHI_API_KEY environment variable is not set.")

# WebSearch Agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for relevant information and provide concise, well-sourced responses.",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=[
        "Search the web for reliable information.",
        "Always provide at least one credible source for your findings.",
        "Summarize the findings clearly and concisely."
    ],
    show_calls=True,
    markdown=True
)

# Finance Agent
finance_agent = Agent(
    name="Finance Agent",
    role="Analyze financial data and provide investment insights.",
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True,
            company_info=True,
            historical_prices=True,
            technical_indicators=True
        )
    ],
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    show_tool_calls=True,
    description="You are an investment analyst specializing in stock analysis and financial research.",
    instructions=[
        "Analyze stock prices, fundamentals, and analyst recommendations.",
        "Use tables to present data where applicable for clarity.",
        "Provide recent news highlights relevant to the stock."
    ],
    markdown=True
)

# Create Playground
app = Playground(agents=[web_search_agent, finance_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)
