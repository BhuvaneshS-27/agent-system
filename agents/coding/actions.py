"""The one JSON object the model returns at every step, and the checks on its arguments.

The JSON schema makes the SHAPE always valid ({thought, tool, args}); `args` is a free dict
there, so validate_action() checks the arguments of the chosen tool with pydantic and turns
every problem into a readable message that is told to the model.
"""
import copy
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

TOOLS = ("web_search", "fetch_page", "read_file", "write_file", "list_files",
         "run_python", "run_shell", "startup_research", "finish")

# Tools that really work in the current build step. startup_research is added in Step 8.
# run_shell needs an approval gate and WEB_TOOLS need a WebTools object: agent_loop drops
# them by itself when it is started without one (and adds WEB_TOOLS when it is given one).
DEFAULT_ENABLED = ("list_files", "read_file", "write_file", "run_python", "run_shell", "finish")
WEB_TOOLS = ("web_search", "fetch_page")

# name -> (example arguments, what it does). Used for the system prompt and for error messages.
TOOL_DOCS = {
    "list_files": ('{"path": "."}', "list the files in the workspace"),
    "read_file": ('{"path": "notes.txt"}', "read a file"),
    "write_file": ('{"path": "solution.py", "content": "..."}',
                   "create or overwrite a file (a .py file is syntax-checked at once)"),
    "run_python": ('{"path": "tests.py"}  or  {"code": "print(1)"}',
                   "run Python (30 second limit, no input()); you get the output or the traceback"),
    "run_shell": ('{"command": "pip install rich"}', "run a shell command; the user must approve it first"),
    "web_search": ('{"query": "..."}', "search the web: you get 5 results with title, URL and a short snippet"),
    "fetch_page": ('{"url": "https://..."}',
                   'read the text of a web page, 3000 characters at a time; add "start": 3000 for the next part'),
    "startup_research": ('{"category": "FinTech"}', "rank the top startups of a category"),
    "finish": ('{"answer": "..."}', "give the final answer when the task is done and checked"),
}


class Action(BaseModel):
    thought: str
    tool: Literal["web_search", "fetch_page", "read_file", "write_file", "list_files",
                  "run_python", "run_shell", "startup_research", "finish"]
    args: dict[str, Any]


ACTION_SCHEMA = Action.model_json_schema()


def action_schema(enabled=DEFAULT_ENABLED) -> dict:
    """The action schema with the tool list cut down to the tools that work right now,
    so the model cannot even choose a tool that is not available."""
    schema = copy.deepcopy(ACTION_SCHEMA)
    schema["properties"]["tool"]["enum"] = [t for t in TOOLS if t in enabled]
    return schema


# ------------------------------------------------------------------ arguments per tool
class ListFilesArgs(BaseModel):
    path: str = "."


class ReadFileArgs(BaseModel):
    path: str = Field(min_length=1)


class WriteFileArgs(BaseModel):
    path: str = Field(min_length=1)
    content: str


class RunPythonArgs(BaseModel):
    code: str = ""
    path: str = ""

    @model_validator(mode="after")
    def one_of(self):
        if bool(self.code.strip()) == bool(self.path.strip()):
            raise ValueError("give exactly one of 'code' or 'path'")
        return self


class RunShellArgs(BaseModel):
    command: str = Field(min_length=1)


class WebSearchArgs(BaseModel):
    query: str = Field(min_length=1)


class FetchPageArgs(BaseModel):
    url: str = Field(pattern=r"^https?://")
    start: int = Field(default=0, ge=0)


class StartupResearchArgs(BaseModel):
    category: str = Field(min_length=1)


class FinishArgs(BaseModel):
    answer: str = Field(min_length=1)


ARG_MODELS = {
    "list_files": ListFilesArgs, "read_file": ReadFileArgs, "write_file": WriteFileArgs,
    "run_python": RunPythonArgs, "run_shell": RunShellArgs, "web_search": WebSearchArgs,
    "fetch_page": FetchPageArgs, "startup_research": StartupResearchArgs, "finish": FinishArgs,
}


def validate_action(act: Action, enabled=DEFAULT_ENABLED) -> dict:
    """Returns the cleaned arguments (unknown keys dropped, defaults filled in), or raises
    ValueError with a message the model can act on."""
    if act.tool not in enabled:
        raise ValueError(f"tool '{act.tool}' is not available. Use one of: "
                         f"{', '.join(t for t in TOOLS if t in enabled)}")
    example = TOOL_DOCS[act.tool][0]
    try:
        return ARG_MODELS[act.tool].model_validate(act.args).model_dump()
    except ValidationError as e:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'args'}: {err['msg']}" for err in e.errors())
        raise ValueError(f"wrong arguments for {act.tool} ({problems}). Expected {example}") from None