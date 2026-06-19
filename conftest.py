"""Rend les modules de `src/` importables par les tests (`from ast_extractor import …`)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
