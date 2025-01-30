from utility.bigquery_config import setup_bigquery_config
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.sql_database import SQLDatabase
from langchain.tools import Tool
from sqlalchemy import create_engine
from google.cloud import bigquery
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

config = setup_bigquery_config()
project_id = config["project_id"]
dataset_id = config["dataset_id"]


# Create a SQLAlchemy engine for BigQuery
engine = create_engine(f"bigquery://{project_id}/{dataset_id}")


# Initialize LangChain's database wrapper
db = SQLDatabase(engine)


def get_top_reviewed_products():
    query = "SELECT title, rating_number FROM Products ORDER BY rating_number DESC LIMIT 5"
    result = db.run(query)
    return result


def get_most_common_sentiments():
    query = "SELECT sentiment, COUNT(*) as count FROM Sentiments GROUP BY sentiment ORDER BY count DESC"
    result = db.run(query)
    return result


# Define tools
top_products_tool = Tool(
    name="Top Reviewed Products",
    func=get_top_reviewed_products,
    description="Get the top 5 most reviewed products."
)

common_sentiments_tool = Tool(
    name="Most Common Sentiments",
    func=get_most_common_sentiments,
    description="Get the most common sentiment categories in reviews."
)

tools = [top_products_tool, common_sentiments_tool]

# Load OpenAI API Key
OPENAI_API_KEY = "your-openai-key"

# Initialize the LLM
# llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
llm = ChatGoogleGenerativeAI(
    model="gemini-pro", google_api_key="your-google-api-key")


# Create an agent that can use predefined tools or generate SQL dynamically
agent = initialize_agent(
    tools=tools,  # Predefined tools
    llm=llm,
    # Uses reasoning to pick the best tool
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)


def chat():
    print("Welcome to the AI Review Assistant. Type 'exit' to quit.")

    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        response = agent.run(query)  # The agent decides how to answer
        print("\nBot:", response)


if __name__ == "__main__":
    chat()
