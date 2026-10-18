"""The one JSON object the model must return at every step."""
from typing import Any, Literal

from pydantic import BaseModel

TOOLS = ("web_search", "fetch_page", "read_file", "write_file", "list_files",
         "run_python", "run_shell", "startup_research", "finish")


class Action(BaseModel):
    thought: str
    tool: Literal["web_search", "fetch_page", "read_file", "write_file", "list_files",
                  "run_python", "run_shell", "startup_research", "finish"]
    args: dict[str, Any]


ACTION_SCHEMA = Action.model_json_schema()