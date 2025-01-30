from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool, Tool
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from analyze import Calculate_Overall_Sentiment_Distribution
import re
from pydantic import BaseModel, Field
import asyncio

load_dotenv()


def greet_user(name: str) -> str:
    """Greets the user by name."""
    return f"Hello, {name}!"

# kheili tool mikeshe


def get_overall_sentiment_distribution(res):
    print("response is {}".format(res))
    """Calculates the overall sentiment distribution."""
    print("Sentiment Distribution tool invoked.")  # Debugging log

    if res:

        answer = Calculate_Overall_Sentiment_Distribution(res)
    else:
        answer = Calculate_Overall_Sentiment_Distribution()
    return answer[0]


def get_current_time(query):
    print("query is {}".format(query))
    """Returns the current time in H:MM AM/PM format. The 'unused_input' can be safely ignored."""
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")


def search_wikipedia(query):
    print("query is {}".format(query))
    from wikipedia import summary

    try:

        return summary(query, sentences=2)
    except:
        return "I couldn't find any information on that."


# Pydantic model for tool arguments


# class SentimentDistributionArgs(BaseModel):
#     query: str = Field(description="alphanumeric string")


tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Use this tool to get the current time. The input is not necessary, you can pass anything.",
    ),
    Tool(
        name="Greeter",
        func=greet_user,
        description="Use this tool to greet given username",
    ),
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="Useful for when you need to know information about a topic.",
    ),

    Tool(
        name="Sentiment Distribution",
        func=get_overall_sentiment_distribution,
        # description="This tool automatically calls the database and return overall sentiment distribution for a givenproduct ID as input. Input should be an alphanumeric string representing the product ID. Just get the product ID from user input and pass it to this tool",
        description="Useful for when you need sentiment information related to given product ID about kiana store.",

    )
    # StructuredTool.from_function(
    #     func=get_overall_sentiment_distribution,  # Function to execute
    #     name="Sentiment Distribution",  # Name of the tool
    #     description="Useful for when you need to know the info for a given product ID as input.",
    #     args_schema=SentimentDistributionArgs,  # Pass the class directly
    # )


]

# Define the Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Respond to the human as helpfully and accurately as possible."
     "You have access to the following tools:"
     "\n\n{tools}\n\n"
     "Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input)."
     "\n\nValid \"action\" values: \"Final Answer\" or {tool_names}\n\n"
     "Provide only ONE action per $JSON_BLOB, as shown:"
     "\n\n```\n{{\n\"action\": $TOOL_NAME,\n\"action_input\": $INPUT\n}}\n```\n\n"
     "Follow this format:"
     "\n\nQuestion: input question to answer\n"
     "Thought: consider previous and subsequent steps\n"
     "Action:\n```\n$JSON_BLOB\n```\n"
     "Observation: action result\n... (repeat Thought/Action/Observation N times)\n"
     "Thought: I know what to respond\n"
     "Action:\n```\n{{\n\"action\": \"Final Answer\",\n\"action_input\": \"Final response to human\"\n}}\n\n"
     "Begin! Reminder to ALWAYS respond with a valid json blob of a single action. "
     "Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation"),

    ("placeholder", "{chat_history}"),
    ("human", "{input}\n\n{agent_scratchpad}\n(reminder to respond in a JSON blob no matter what)"),
])


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)


memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True
)

agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
)


while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    memory.chat_memory.add_message(HumanMessage(content=user_input))

    response = agent_executor.invoke({"input": user_input})
    print("Bot:", response["output"])

    memory.chat_memory.add_message(AIMessage(content=response["output"]))
