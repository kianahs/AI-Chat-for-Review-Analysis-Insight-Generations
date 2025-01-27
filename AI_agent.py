from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from analyze import Calculate_Overall_Sentiment_Distribution

# Load environment variables from .env file
load_dotenv()


def get_overall_sentiment_distribution(query):
    """Calculates the overall sentiment distribution."""
    print("Sentiment Distribution tool invoked.")  # Debugging log
    res = Calculate_Overall_Sentiment_Distribution()
    return res


def get_current_time(query):
    """Returns the current time in H:MM AM/PM format. The 'unused_input' can be safely ignored."""
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")


def search_wikipedia(query):

    from wikipedia import summary

    try:

        return summary(query, sentences=2)
    except:
        return "I couldn't find any information on that."


tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Use this tool to get the current time. The input is not necessary, you can pass anything.",
    ),
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="Useful for when you need to know information about a topic.",
    ),

    Tool(
        name="Sentiment Distribution",
        func=get_overall_sentiment_distribution,
        description="Calculates the overall sentiment distribution from the pre-existing dataset. No input is required. You can call it without providing any input.",
    )
]

# Define the Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Respond to the human as helpfully and accurately as possible. You have access to the following tools:\n\n{tools}\n\nUse a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).\n\nValid \"action\" values: \"Final Answer\" or {tool_names}\n\nProvide only ONE action per $JSON_BLOB, as shown:\n\n```\n{{\n\"action\": $TOOL_NAME,\n\"action_input\": $INPUT\n}}\n```\n\nFollow this format:\n\nQuestion: input question to answer\nThought: consider previous and subsequent steps\nAction:\n```\n$JSON_BLOB\n```\nObservation: action result\n... (repeat Thought/Action/Observation N times)\nThought: I know what to respond\nAction:\n```\n{{\n\"action\": \"Final Answer\",\n\"action_input\": \"Final response to human\"\n}}\n\nBegin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation"),
    ("placeholder", "{chat_history}"),
    ("human", "{input}\n\n{agent_scratchpad}\n(reminder to respond in a JSON blob no matter what)"),
])


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")


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
