"""Shared tool schemas used by the scientific data exploration samplers."""

from __future__ import annotations

from typing import Any, Dict


PYTHON_TOOL_SCHEMA: Dict[str, Any] = {
    "name": "python_interpreter",
    "description": (
        "Execute Python code to explore and analyze scientific dataset files and "
        "directories. Pre-installed libraries: pandas, numpy, scipy, sklearn, "
        "matplotlib, seaborn, os, glob, json, csv, pathlib. Use this tool to list "
        "directories, read files, inspect schemas, compute statistics, and derive "
        "insights. Always print() results so you can observe the output."
    ),
    "parameters": [
        {
            "name": "code",
            "type": "string",
            "description": "Python code to execute.",
            "required": True,
        }
    ],
}


__all__ = ["PYTHON_TOOL_SCHEMA"]
