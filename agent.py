# ==============================================================================
# agent.py: AGENT TOOLS & MULTI-STEP REASONING ARCHITECTURE
# ==============================================================================

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from ingest import get_vector_store

# Load environment variables from .env file if present
load_dotenv()

# Initialize the primary reasoning LLM (Llama 3.3 70B via Groq)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=4096
)

# ------------------------------------------------------------------------------
# TOOL 1: KNOWLEDGE BASE SEARCH TOOL (WITH METADATA CITATIONS)
# ------------------------------------------------------------------------------
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the uploaded PDF documents for relevant facts, figures, or policies.
    Always pass clear, targeted search keywords. If results are insufficient,
    refine or rephrase the query and call this tool again.
    """
    vector_store = get_vector_store()
    if vector_store is None:
        return "Error: Knowledge base is empty. Please upload and process PDF documents first."

    # Retrieve top 4 most relevant chunks based on cosine similarity
    results = vector_store.similarity_search(query, k=4)

    if not results:
        return "No matching information found for this query. Try refining your search terms."

    formatted_chunks = []
    for idx, doc in enumerate(results, 1):
        src = doc.metadata.get("source_file", "Unknown")
        pg = doc.metadata.get("page_number", "N/A")
        formatted_chunks.append(
            f"[Chunk {idx} | Source: {src}, Page: {pg}]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(formatted_chunks)

# ------------------------------------------------------------------------------
# TOOL 2: PRECISION CALCULATOR TOOL
# ------------------------------------------------------------------------------
@tool
def calculator_tool(expression: str) -> str:
    """
    Evaluate mathematical expressions accurately (e.g., '(45000 - 32000) / 32000 * 100').
    Use this tool whenever arithmetic, percentages, or numeric differences are required.
    """
    try:
        allowed_chars = set("0123456789+-*/().% ")
        if not all(char in allowed_chars for char in expression):
            return "Error: Invalid characters in expression. Use only numbers and math operators (+, -, *, /, %)."

        result = eval(expression, {"__builtins__": None}, {})
        return f"Calculation Result: {round(result, 4) if isinstance(result, float) else result}"
    except Exception as e:
        return f"Math Evaluation Error: {str(e)}"

# ------------------------------------------------------------------------------
# TOOL 3: STATISTICAL DATA ANALYSIS TOOL
# ------------------------------------------------------------------------------
@tool
def data_analysis_tool(numbers_csv: str, operation: str) -> str:
    """
    Perform statistical analysis on a comma-separated list of numbers (e.g., '150, 200, 275, 310').
    Supported operations: 'sum', 'average', 'min', 'max', 'growth_rate'.
    Note: 'growth_rate' calculates percentage change between the first and last number.
    """
    try:
        values = [float(x.strip()) for x in numbers_csv.split(",") if x.strip()]
        if not values:
            return "Error: No valid numeric values provided."

        op = operation.lower().strip()

        if op == "sum":
            return f"Total Sum: {sum(values)}"
        elif op == "average":
            return f"Average (Mean): {sum(values) / len(values):.2f}"
        elif op == "min":
            return f"Minimum Value: {min(values)}"
        elif op == "max":
            return f"Maximum Value: {max(values)}"
        elif op == "growth_rate":
            if len(values) < 2 or values[0] == 0:
                return "Error: Growth rate requires at least two numbers and a non-zero starting value."
            rate = ((values[-1] - values[0]) / values[0]) * 100
            return f"Percentage Growth Rate (from {values[0]} to {values[-1]}): {rate:.2f}%"
        else:
            return f"Unsupported operation '{operation}'. Choose from: sum, average, min, max, growth_rate."
    except Exception as e:
        return f"Data Analysis Error: {str(e)}"

# ------------------------------------------------------------------------------
# TOOL 4: DOCUMENT SUMMARIZER & SYNTHESIZER TOOL
# ------------------------------------------------------------------------------
@tool
def summarizer_tool(text_content: str) -> str:
    """
    Condense lengthy extracted passages into concise, structured bullet points.
    Use this when comparing multiple documents or synthesizing large sections of text.
    """
    summary_prompt = (
        "Summarize the following extracted document text into clear, factual bullet points. "
        "Preserve all key metrics, dates, and source citations:\n\n"
        f"{text_content}"
    )
    response = llm.invoke(summary_prompt)
    return response.content

# ------------------------------------------------------------------------------
# INITIALIZE THE LANGGRAPH REACT AGENT
# ------------------------------------------------------------------------------
agent_tools = [
    search_knowledge_base,
    calculator_tool,
    data_analysis_tool,
    summarizer_tool
]

AGENT_SYSTEM_PROMPT = """You are DocuMind Agent, an advanced autonomous Agentic RAG Assistant.
Your mission is to answer complex, multi-step questions accurately using the provided document knowledge base and your specialized tools.

STRICT OPERATIONAL WORKFLOW:
1. PLAN: Break down complex user questions into smaller sub-tasks.
2. RETRIEVE & REFINE: Call `search_knowledge_base` to find facts. If the initial retrieval misses key details or returns weak matches, automatically rephrase/refine your search query and search again.
3. COMPARE & COMPUTE: For comparisons across documents or years, retrieve data for each entity separately. Always use `calculator_tool` or `data_analysis_tool` for math/statistics, and `summarizer_tool` to condense long findings.
4. CITE SOURCES: Every factual claim from the documents MUST include its exact citation in the format: [Source: filename.pdf, Page: X].
5. GROUNDED HONESTY: If the answer is not found in the documents after query refinement, clearly state that the uploaded documents do not contain that information. Do not hallucinate."""

documind_agent = create_react_agent(
    model=llm,
    tools=agent_tools,
    prompt=AGENT_SYSTEM_PROMPT
)

def run_documind_agent(user_query: str):
    """
    Executes the multi-step agent loop and captures both the final answer
    and a step-by-step trace of the agent's reasoning and tool selections.
    """
    if get_vector_store() is None:
        return (
            "⚠️ **Knowledge Base Not Found:** Please go to the **Knowledge Base Setup** tab, upload your PDF files, and click **Build & Index Knowledge Base** first.",
            "No reasoning steps executed."
        )

    inputs = {"messages": [("user", user_query)]}
    reasoning_trace = []
    final_response = ""
    step_counter = 1

    # Stream intermediate steps from the LangGraph execution loop
    for state in documind_agent.stream(inputs, stream_mode="values"):
        latest_msg = state["messages"][-1]

        # Capture when the Agent decides to call a tool
        if hasattr(latest_msg, "tool_calls") and latest_msg.tool_calls:
            for call in latest_msg.tool_calls:
                tool_name = call["name"]
                tool_args = call["args"]
                reasoning_trace.append(
                    f"### Step {step_counter}: Tool Selected ➔ `{tool_name}`\n"
                    f"**Parameters / Query:** `{tool_args}`\n"
                )
                step_counter += 1

        # Capture the Agent's final synthesized answer
        elif latest_msg.type == "ai" and latest_msg.content:
            final_response = latest_msg.content

    trace_output = "\n---\n".join(reasoning_trace) if reasoning_trace else "Direct reasoning without external tool calls."
    return final_response, trace_output
