from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from app.generator import generator

@pytest.fixture(scope="session", autouse=True)
def load_generator():
    if generator.model is None:
        generator.load_model()
