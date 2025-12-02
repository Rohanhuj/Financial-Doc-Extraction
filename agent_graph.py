from setup_config import *
from ingest_vectorize import *
from rag_tools import *

# ## The Mastermind - Orchestrating the Reasoning Engine

# ### Define the Master Graph and Enhanced State
# 
# We will now define the structure for our more advanced agent. The `AgentState` 
# needs to be enhanced to handle the new cognitive steps. It will now track 
# verification results and potential clarification questions. 

class AgentState(TypedDict):
    """Defines the state of our agent graph."""
    original_request: str
    plan: List[str]
    intermediate_steps: List[Dict[str, Any]]
    verification_history: List[Dict[str, Any]] # For self-correction
    final_response: str

print("AgentState TypedDict defined.")

#supervisor_llm = ChatOpenAI(model="gpt-4o", temperature=0)
supervisor_llm = ChatOllama(model="mistral-small3.2", temperature=0)

def create_planner_prompt(tools):
    tool_descriptions = "\n".join([f"- {tool.name}: {tool.description.strip()}" for tool in tools])
    return f"""You are an information extraction agent, the Supervisor. Your task is to create a step-by-step plan to extract information per the user's request by intelligently selecting from the available tools.
**Available Tools:**
{tool_descriptions}
**Instructions:**
1. Analyze the user's request.
2. Create a clear, step-by-step plan. Each step must be a call to one of the available tools.
3. The final step in your plan should ALWAYS be 'FINISH'.
**Output Format:**
Return the plan as a list of strings using valid python3 syntax.
For example: ['librarian_rag_tool("Extract firm name and location")', 'FINISH']
---
User Request: {{request}}
Plan:"""

planner_prompt_template = create_planner_prompt(tools)

def planner_node(state: AgentState) -> Dict[str, Any]:
    print("\n-- Planner Node --")
    request = state['original_request']
    prompt = planner_prompt_template.format(request=request)
    plan_str = supervisor_llm.invoke(prompt).content
    print(f"Planner node llm invocation returned : {plan_str}")
    try:
        plan = eval(plan_str)
        print(f"  - Generated Plan: {plan}")
        return {"plan": plan}
    except Exception as e:
        print(f"Error parsing plan: {e}. Falling back to FINISH.")
        return {"plan": ['FINISH']}


def tool_executor_node(state: AgentState) -> Dict[str, Any]:
    print("\n-- Tool Executor Node --")
    next_step = state['plan'][0]
    try:
        tool_name = next_step.split('(')[0]
        tool_input_str = next_step[len(tool_name)+1:-1]
        tool_input = eval(tool_input_str)
    except Exception as e:
        print(f"  - Error parsing tool call {next_step}: as {e}. Skipping step.")
        return {"plan": state['plan'][1:], "intermediate_steps": state.get('intermediate_steps', [])}

    print(f"  - Executing tool: {tool_name} with input: '{tool_input}'")
    tool_to_call = tool_map[tool_name]
    #FIXME : Some tools can additional arguments not just the query string... This should use an LLM to predict
    # the tool arguments to be passed as in like MCP.
    result = tool_to_call.invoke({"query": tool_input})

    new_intermediate_step = {
        'tool_name': tool_name,
        'tool_input': tool_input,
        'tool_output': result
    }

    current_steps = state.get('intermediate_steps', [])
    return {
        "intermediate_steps": current_steps + [new_intermediate_step],
        "plan": state['plan'][1:]
    }


class VerificationResult(BaseModel):
    """Structured output for the Auditor node."""
    is_relevant: bool = Field(description="Is the output relevant to the original user request?")
    is_consistent: bool = Field(description="Is the output internally consistent?")
    confidence_score: int = Field(description="Score from 1-5 based on confidence in the tool's output.")
    reasoning: str = Field(description="Brief reasoning for the scores.")

#auditor_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(VerificationResult)
#auditor_llm = ChatOllama(model="mistral-small3.2", temperature=0).with_structured_output(VerificationResult)
auditor_llm = ChatOllama(model="gpt-oss:20b", temperature=0).with_structured_output(VerificationResult)
#auditor_llm = ChatOllama(model="llama3.1:8b", temperature=0).with_structured_output(VerificationResult)

def verification_node(state: AgentState) -> Dict[str, Any]:
    """Audits the most recent tool output for quality and relevance."""
    print("\n-- Auditor (Self-Correction) Node --")
    request = state['original_request']
    last_step = state['intermediate_steps'][-1]

    prompt = f"""You are a meticulous fact-checker and auditor. Given the user's original request and the output from a tool, audit the output.

    **User Request:** {request}
    **Tool:** {last_step['tool_name']}
    **Tool Output:** {json.dumps(last_step['tool_output'])}

    **Audit Checklist:**
    1.  **Relevance:** Is this output directly relevant to answering the user's request? (Boolean, yes or no).
    2.  **Consistency:** Is the data internally consistent i.e. without contradictions ? (Boolean, yes or no).
    3.  **Confidence:** A measure of confidence in the quality of the tool output (Score 1-5, where 5 is highest)

    Based on these ratings, provide a brief reasoning for the same.
    """

    print(f"Auditor prompt: {prompt}")
    audit_result = auditor_llm.invoke(prompt)
    print(f"  - Audit Confidence Score: {audit_result.confidence_score}/5")

    current_history = state.get('verification_history', [])
    # We only need to return the verification history, the router will handle the rest
    return {"verification_history": current_history + [audit_result.dict()]}


def router_node(state: AgentState) -> str:
    """This node decides the next step in the graph based on the current state."""
    print("\n-- Advanced Router Node --")

    # Check for clarification first, this is a terminal state
    # Check if we need to start the main workflow
    if not state.get("plan"):
        print("  - Decision: New request. Routing to planner.")
        return "planner"

    # Check the last verification result if it exists
    if state.get("verification_history"):
        last_verification = state["verification_history"][-1]
        if last_verification["confidence_score"] < 3:
            print("  - Decision: Verification failed. Returning to planner.")
            # Clear the plan to force replanning
            state['plan'] = [] 
            return "planner"

    # Check if the plan is complete
    if not state.get("plan") or state["plan"][0] == "FINISH":
        print("  - Decision: Plan is complete. Routing to synthesizer.")
        return "synthesize"
    else:
        print("  - Decision: Plan has more steps. Routing to tool executor.")
        return "execute_tool"

# Read config json
#config = Config().read_config_file()

# Read schema json
#json_schema = Json_schema().read_json_schema_file()

#from pydantic_report import ReportInformation


extraction_llm = ChatOllama(model="gpt-oss:20b", temperature=0, num_ctx=32768)
#extraction_llm = ChatOllama(model="gpt-oss:20b", temperature=0).with_structured_output(ReportInformation)


def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    print("\n-- Strategist (Synthesizer) Node --")
    request = state['original_request']
    context = "\n\n".join([f"## Tool: {step['tool_name']}\nInput: {step['tool_input']}\nOutput: {json.dumps(step['tool_output'], indent=2)}" for step in state['intermediate_steps']])

    systemMessage = "You are an advanced information extraction agent that reads document information and returns specific extracted fields as valid JSON."

    prompt = f"""{systemMessage}
Your task is to synthesize a comprehensive answer to the information extraction request based on the context provided.

**Context:**
---
{context}
---

**Information extraction Request:**
{request}

**Instructions:**
1.  Carefully review the context from the tool outputs and the information extraction request.
2.  Construct a clear, well-written, and accurate response to the information extraction request.
3.  Produce the answer as valid JSON

Final Answer:
"""
    with open("synthesizer_prompt.txt",'w') as f:
        f.write(prompt)

    print(f"Extraction Agent Prompt : {prompt}")
    result = extraction_llm.invoke(prompt)
    print(f"Generated final result as {result}")
    final_answer = result.content
    # FIXME : For some reason the with_structured_output(ReportInformation) does not work
    # so we work around it for now using json string parsing
    try:
        result = extract_and_parse_json(final_answer)
        if result:
            final_answer = result
    except Exception as e:
        print(f"Caught exception while parsing json, its probably invalid formatting as {e}")
    print(f"  - Generated final answer as {final_answer}")
    return {"final_response": final_answer}

# ### Compile and Run the Advanced Graph
# 
# It's time to assemble all our new and existing nodes into the complete 
# reasoning graph. The flow is now much more complex and powerful, incorporating 
# our Gatekeeper, Auditor, and multiple feedback loops.

def build_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("execute_tool", tool_executor_node)
    workflow.add_node("verify", verification_node)
    workflow.add_node("synthesize", synthesizer_node)

    # Define the entry point
    workflow.set_entry_point("planner")

    # After planning, always execute a tool
    workflow.add_edge("planner", "execute_tool")

    # After execution, always verify
    #workflow.add_edge("execute_tool", "verify")
    workflow.add_edge("execute_tool", "synthesize")

    # The ADVANCED ROUTER connects the verification step to the next logical node
    #workflow.add_conditional_edges(
        #"verify",
        #router_node,
        #{
            #"planner": "planner",
            #"execute_tool": "execute_tool",
            #"synthesize": "synthesize",
        #}
    #)

    # The synthesizer is a terminal node
    workflow.add_edge("synthesize", END)

    # Compile the graph
    graph_workflow = workflow.compile()

    print("Graph compiled successfully!")

    return graph_workflow


def save_graph_to_png(graph: CompiledStateGraph, file_path: str):
    """
    Saves a compiled LangGraph to a PNG file directly, bypassing Mermaid.
    Requires pygraphviz and graphviz to be installed.
    """
    try:
        from langchain_core.runnables.graph import MermaidDrawMethod
        png_data = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER)
        with open(file_path, "wb") as f:
            f.write(png_data)
        print(f"Graph PNG saved to {file_path}")
    except ImportError:
        print("Please install 'pygraphviz' and 'graphviz' to use this function.")
    except Exception as e:
        print(f"An error occurred: {e}")


def run_graph(graph, query: str):
    # A wrapper to run the graph and print the final output cleanly
    print(f"--- Running Graph with Query ---")
    print(f"Query: {query}")
    # Ensure initial state has empty lists for accumulation
    inputs = {"original_request": query, "verification_history": [], "intermediate_steps": []}
    final_state = {}
    # Use a for loop to stream and see the flow, but capture the last state for the final answer
    for output in graph.stream(inputs, stream_mode="values"):
        final_state.update(output)

    print("\n--- FINAL SYNTHESIZED RESPONSE ---")
    print(final_state['final_response'])
    return final_state

