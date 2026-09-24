Project Conventions:

Language: Python 3.11+

Backend: FastAPI with Uvicorn.

Data/AI Stack: LangChain for retrieval, Pinecone for vector DB, TRL/PEFT for QLoRA fine-tuning.

Style: Strict type hinting on all Python functions.

Rule: Never mock database connections; always write code that expects the Dockerized MSSQL and real Pinecone API

# Agentic Behavioral Directives

## 1. Plan Adherence & State Tracking
* Before writing any code, silently read `planning.md`. 
* Never skip a phase. If a phase is incomplete, do not move to the next.
* Whenever you complete a milestone, update a `STATUS.md` file with a timestamp and a one-sentence summary of what works.

## 2. The Verification Rule (No Blind Coding)
* Never assume your code works. 
* For every API endpoint, database schema, or ML script you write, you MUST write a corresponding temporary test script (e.g., `test_api.py`), run it locally, and read the output.
* If a test fails, you are authorized to fix the code and re-run the test up to 3 times before stopping to ask the user for help.

## 3. Communication
* Do not print large blocks of code to the terminal unless asked.
* When you finish a task, print a short, 3-bullet-point summary to the terminal: What was built, how it was tested, and what the test output was.

## 4. Project Stack & Conventions
* Python 3.11+, FastAPI, Pinecone, LangChain.
* ML Stack: TRL SFTTrainer, PEFT (QLoRA), PyTorch.
* Treat backend validation strictly—validate inputs and outputs at each step, similar to managing state transitions in a LangGraph workflow.


# Agent Evaluation Setup
- **Test Runner:** We use DeepEval for agent testing. Run tests using `deepeval test run <filename>`.
- **Model Constraints:** We use Gemini (or custom local models) for the DeepEval judge, not OpenAI. 
- **Strict Rule 1:** NEVER modify a test file or the test dataset just to make a failing test pass. 
- **Strict Rule 2:** If a test fails, you must read the LLM judge's reasoning in the output to understand *why* the agent failed.
- **Strict Rule 3:** Fix failures by improving the agent's system prompt, tool logic, or LangGraph state—never by hardcoding the answer.