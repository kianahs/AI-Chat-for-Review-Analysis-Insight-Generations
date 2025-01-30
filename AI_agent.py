from sqlalchemy import text
from langchain.tools import Tool
from collections import Counter
from dotenv import load_dotenv
from google.cloud import bigquery
import os
from langchain.agents import initialize_agent, Tool
from langchain_community.utilities import SQLDatabase
from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import re
import ast


from utility.bigquery_config import setup_bigquery_config
config = setup_bigquery_config()
project_id = config["project_id"]
dataset_id = config["dataset_id"]
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")


bigquery_uri = f"bigquery://{project_id}/{dataset_id}"

engine = create_engine(bigquery_uri)
Session = sessionmaker(bind=engine)
db = SQLDatabase.from_uri(bigquery_uri)

# sql_prompt = """
# You are an expert SQL assistant working with a BigQuery database.

# ### Database Schema:
# - **Reviews**
#   - (id, rating, title, text, asin, parent_asin,
#      helpful_vote, timestamp, verified_purchase)
#   - `id` is the primary key.
#   - `parent_asin` is a foreign key referencing `Products.parent_asin`.

# - **Products**
#   - (id, main_category, title, features, price, description,
#      parent_asin, average_rating, rating_number)
#   - `parent_asin` is the primary key, referenced by `Reviews.parent_asin`.
#   - `id` is the primary key.

# - **Sentiments**
#   - (id, summary, sentiment)
#   - `id` is a foreign key referencing `Reviews.id`.

# ### Guidelines:
# - **Use correct table and column names.**
# - **Use `JOIN` when necessary**
#   - Example: `Reviews.id = Sentiments.id` for sentiment analysis.
#   - Example: `Reviews.parent_asin = Products.parent_asin` to get product details.
# - **Use aggregation functions (`COUNT()`, `AVG()`, `SUM()`) where appropriate.**
# - **Use `GROUP BY` for sentiment distribution or category-based stats.**
# - **Filter with `WHERE` when needed**
#   - Example: `WHERE verified_purchase = TRUE` for verified purchases only.
#   - Example: `WHERE timestamp >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 YEAR)` for last year's reviews.

#   Your task is to answer natural language queries using this database schema.
#   Please respond with SQL queries that will extract the necessary data from these tables. If the query is complex, try to break it down into multiple steps.
# """
# prompt_template = PromptTemplate(
#     input_variables=["user_input"], template=sql_prompt)


def get_overall_sentiment_trends_over_time(query):

    print(f'query is {query}')
    session = Session()

    try:

        stored_procedure_query = f"""
        CALL `{project_id}.{dataset_id}.GetOverallSentimentTrendsData`();
        """
        stored_procedure_query = text(
            stored_procedure_query)

        result = session.execute(stored_procedure_query)
        results_set = result.fetchall()

        results_list = []
        for row in results_set:

            results_list.append(
                {"id": row[0], "timestamp": row[1], "sentiment": row[2]})

        df = pd.DataFrame(results_list)

        sentiment_mapping = {'POSITIVE': 1, 'NEUTRAL': 0, 'NEGATIVE': -1}
        df['sentiment_score'] = df['sentiment'].map(sentiment_mapping)

        # Convert timestamp to datetime and extract monthly periods (timestamps are in milliseconds)
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['month'] = df['date'].dt.to_period('M')

        # Group by month and calculate average sentiment score
        monthly_sentiment = df.groupby(
            'month')['sentiment_score'].mean().reset_index()

        # Convert the Period data to string for plotting
        monthly_sentiment['month'] = monthly_sentiment['month'].astype(str)

        # print(monthly_sentiment)

        return monthly_sentiment
    except Exception as e:
        return f"Error calling stored procedure: {e}"

    finally:
        session.close()


def get_sentiment_distribution(query):
    print(f'query is {query}')
    session = Session()

    try:

        stored_procedure_query = f"""
        CALL `{project_id}.{dataset_id}.GetOverallSentimentDistribution`();
        """
        stored_procedure_query = text(
            stored_procedure_query)

        result = session.execute(stored_procedure_query)
        results_set = result.fetchall()

        results_list = []
        for row in results_set:

            results_list.append(
                {"value": row[0], "sentiment": row[1]})

        sentiment_distribution = pd.DataFrame(results_list)
        return sentiment_distribution
    except Exception as e:
        return f"Error calling stored procedure: {e}"

    finally:
        session.close()


def get_sentiment_distribution_for_product(query=None):
    session = Session()
    print(f'query is {query}')
    query_dict = ast.literal_eval(query)
    product_id = query_dict.get('product_id')

    try:
        print(f'Product id is {product_id}')

        stored_procedure_query = f"""
        CALL `{project_id}.{dataset_id}.GetOverallSentimentDistributionPerProductID`('{product_id}');
        """
        stored_procedure_query = text(
            stored_procedure_query)

        result = session.execute(stored_procedure_query)
        results_set = result.fetchall()

        results_list = []
        for row in results_set:

            results_list.append(
                {"value": row[0], "sentiment": row[1]})

        sentiment_distribution = pd.DataFrame(results_list)
        return sentiment_distribution
    except Exception as e:
        return f"Error calling stored procedure: {e}"

    finally:
        session.close()


sentiment_tool = Tool(
    name="SentimentAnalysisTool",
    func=get_sentiment_distribution,
    description="Calculates sentiment distribution from the Sentiments table."
)
trend_sentiment_tool = Tool(
    name="TrendSentimentAnalysisTool",
    func=get_overall_sentiment_trends_over_time,
    description="Calculates overall sentiment trend over time like month from the Sentiments table."
)
product_sentiment_tool = Tool(
    name="ProductSentimentAnalysisTool",
    func=get_sentiment_distribution_for_product,
    description="Calculates sentiment distribution from the Sentiments table. Optionally, pass a product ID to filter results."
)

sql_tool = Tool(
    name="BigQueryTool",
    func=SQLDatabaseChain.from_llm(
        llm=ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", temperature=0, api_key=google_api_key),
        db=db,
        # prompt=prompt_template
    ).invoke,
    description='''Use this tool to query the BigQuery database. Input should be a natural language question about the data.
        ### Database Schema:
        - **Reviews**
        - (id, rating, title, text, asin, parent_asin,
            helpful_vote, timestamp, verified_purchase)
        - `id` is the primary key.
        - `parent_asin` is a foreign key referencing `Products.parent_asin`.

        - **Products**
        - (id, main_category, title, features, price, description,
            parent_asin, average_rating, rating_number)
        - `parent_asin` is the primary key, referenced by `Reviews.parent_asin`.
        - `id` is the primary key.

        - **Sentiments**
        - (id, summary, sentiment)
        - `id` is a foreign key referencing `Reviews.id`.
        
        ### Guidelines:
        - **Use correct table and column names.**
        - **Use `JOIN` when necessary**
        - Example: `Reviews.id = Sentiments.id` for sentiment analysis.
        - Example: `Reviews.parent_asin = Products.parent_asin` to get product details.
        - **Use aggregation functions (`COUNT()`, `AVG()`, `SUM()`) where appropriate.**
        - **Use `GROUP BY` for sentiment distribution or category-based stats.**
        - **Filter with `WHERE` when needed**
        - Example: `WHERE verified_purchase = TRUE` for verified purchases only.
        - Example: `WHERE timestamp >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 YEAR)` for last year's reviews.
        '''
)


agent = initialize_agent(
    tools=[sql_tool, sentiment_tool,
           product_sentiment_tool, trend_sentiment_tool],
    llm=ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", temperature=0, api_key=google_api_key),
    agent="zero-shot-react-description",
    verbose=True
)


# print('''Welcome to the BigQuery conversational assistant!\n
# You can ask me questions about your data, like 'What is the total revenue in 2024?'
# or 'Show me sales from last month. \n
# Type 'exit' to end the conversation.\n''')

# while True:

#     user_input = input("You: ")

#     if user_input.lower() == "exit":
#         print("Goodbye!")
#         break

#     try:
#         # response = agent.run(user_input)
#         response = agent.invoke(user_input)

#         print(f"Agent: {response}\n")
#     except Exception as e:
#         print(f"Agent: Sorry, I encountered an error: {e}\n")


# Streamlit UI
# Streamlit UI
st.set_page_config(page_title="Chatbot", page_icon="💬", layout="wide")
st.title("💬 Assistant Chatbot")
st.markdown("Chat with your BigQuery database using AI-powered insights.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display message history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Ask something about your database...")
if user_input:
    st.session_state["messages"].append(
        {"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response from the agent
    try:
        response = agent.invoke(user_input)
    except Exception as e:
        response = f"Error: {e}"

    # Display response
    st.session_state["messages"].append(
        {"role": "assistant", "content": response['output']})
    with st.chat_message("assistant"):
        st.markdown(response['output'])
