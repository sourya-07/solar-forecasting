"""
Grid Optimizer — Agentic AI Pipeline (Milestone 2)
====================================================
Implements a 5-node custom state-machine pipeline:

  Node 1  ForecastAnalyzer      — compute forecast statistics
  Node 2  RiskIdentifier        — detect high-risk time windows
  Node 3  KnowledgeRetriever    — RAG: pull relevant grid guidelines
  Node 4  RecommendationGenerator — LLM call (Groq) with JSON output + retry
  Node 5  ReportAssembler       — assemble final structured report

All nodes have try/except guards; the pipeline never crashes the UI.
LLM failures trigger a rule-based fallback automatically.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


# Agent State

@dataclass
class AgentState:
    """Shared mutable state passed between all pipeline nodes."""
    forecast_data: dict = field(default_factory=dict)
    forecast_summary: dict = field(default_factory=dict)
    risk_report: list[dict] = field(default_factory=list)
    retrieved_guidelines: dict = field(default_factory=dict)
    llm_response: dict = field(default_factory=dict)
    final_report: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    used_llm: bool = False