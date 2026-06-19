"""
End-to-end and unit tests for ast_extractor.extract().
"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))   # le test tourne où qu'il soit cloné
SRC = os.path.join(os.path.dirname(HERE), "src")    # ast_extractor.py vit dans src/

from ast_extractor import extract


class TestExtractor(unittest.TestCase):

    def test_extract_returns_contract_keys(self):
        """extract() returns a dict with exactly the top-level keys 'imports' and 'fonctions'."""
        source = "import os\ndef f(): pass\n"
        result = extract(source)
        self.assertIsInstance(result, dict)
        self.assertIn("imports", result)
        self.assertIn("fonctions", result)
        self.assertEqual(set(result.keys()), {"imports", "fonctions"})

    def test_imports_are_top_level_module_names(self):
        """imports list contains only top-level module names, deduped."""
        source = (
            "import jwt\n"
            "import os.path\n"
            "from redis import Redis\n"
            "def f(): pass\n"
        )
        result = extract(source)
        imports = result["imports"]
        self.assertIn("jwt", imports)
        self.assertIn("os", imports)
        self.assertIn("redis", imports)
        # Must NOT contain dotted names or class names
        self.assertNotIn("os.path", imports)
        self.assertNotIn("Redis", imports)

    def test_functions_have_nom_and_params(self):
        """A function def appears in fonctions with correct nom and params."""
        source = "def login(email, password):\n    pass\n"
        result = extract(source)
        fonctions = result["fonctions"]
        self.assertTrue(len(fonctions) >= 1)
        login_entry = next((f for f in fonctions if f["nom"] == "login"), None)
        self.assertIsNotNone(login_entry, "Expected 'login' in fonctions")
        self.assertEqual(login_entry["params"], ["email", "password"])
        self.assertIn("appels", login_entry)

    def test_appels_only_internal(self):
        """appels contains only internal function calls, not external dotted calls."""
        source = (
            "def save_token(user_id, token):\n"
            "    pass\n"
            "\n"
            "def login(email, password):\n"
            "    import bcrypt\n"
            "    hashed = b'x'\n"
            "    bcrypt.checkpw(password, hashed)\n"
            "    save_token(1, 'tok')\n"
        )
        result = extract(source)
        fonctions = result["fonctions"]
        login_entry = next((f for f in fonctions if f["nom"] == "login"), None)
        self.assertIsNotNone(login_entry, "Expected 'login' in fonctions")
        # save_token is internal — must appear
        self.assertIn("save_token", login_entry["appels"])
        # bcrypt.checkpw is external (dotted) — must NOT appear
        for call in login_entry["appels"]:
            self.assertNotIn(".", call, f"Dotted call found in appels: {call}")

    def test_cli_end_to_end(self):
        """Running the CLI as a subprocess exits 0 and stdout is valid JSON."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "ast_extractor.py", "ast_extractor.py"],
            capture_output=True,
            text=True,
            cwd=SRC,
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        data = json.loads(result.stdout)
        self.assertIn("imports", data)
        self.assertIn("fonctions", data)

    # --- Plan 02 edge-case tests ---

    def test_relative_import_no_crash(self):
        """extract() on relative imports does NOT raise; absolute module names still captured."""
        source = "from . import utils\nfrom .helpers import x\n"
        # Must not raise
        result = extract(source)
        self.assertIsInstance(result, dict)
        # "helpers" from "from .helpers import x" should appear (module name = "helpers")
        self.assertIn("helpers", result["imports"])

    def test_async_function_captured(self):
        """extract() on 'async def fetch(url): pass' includes fetch in fonctions."""
        source = "async def fetch(url): pass\n"
        result = extract(source)
        fonctions = result["fonctions"]
        fetch_entry = next((f for f in fonctions if f["nom"] == "fetch"), None)
        self.assertIsNotNone(fetch_entry, "Expected 'fetch' in fonctions")
        self.assertEqual(fetch_entry["params"], ["url"])

    def test_duplicate_imports_deduped(self):
        """extract() on 'import os\\nimport os' yields imports == ['os'] (single entry)."""
        source = "import os\nimport os\n"
        result = extract(source)
        self.assertEqual(result["imports"].count("os"), 1,
                         "Duplicate import 'os' must appear only once")

    def test_duplicate_internal_calls_deduped(self):
        """A function calling helper() three times yields appels == ['helper'] (single entry)."""
        source = (
            "def helper(): pass\n"
            "\n"
            "def caller():\n"
            "    helper()\n"
            "    helper()\n"
            "    helper()\n"
        )
        result = extract(source)
        caller_entry = next((f for f in result["fonctions"] if f["nom"] == "caller"), None)
        self.assertIsNotNone(caller_entry, "Expected 'caller' in fonctions")
        self.assertEqual(caller_entry["appels"].count("helper"), 1,
                         "helper() called 3 times must appear in appels exactly once")

    def test_nested_function_captured(self):
        """extract() on outer() containing inner() includes BOTH in fonctions."""
        source = (
            "def outer():\n"
            "    def inner():\n"
            "        pass\n"
            "    inner()\n"
        )
        result = extract(source)
        noms = [f["nom"] for f in result["fonctions"]]
        self.assertIn("outer", noms, "outer must be in fonctions")
        self.assertIn("inner", noms, "inner (nested) must be in fonctions")

    def test_cli_invalid_source_errors(self):
        """CLI on invalid Python exits non-zero with non-empty stderr and no success JSON."""
        import subprocess
        import tempfile
        import os

        # Write clearly invalid Python to a temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def (:\n")
            tmp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, "ast_extractor.py", tmp_path],
                capture_output=True,
                text=True,
                cwd=SRC,
            )
            # Must exit non-zero
            self.assertNotEqual(result.returncode, 0,
                                "CLI must exit non-zero for invalid Python")
            # Must write a non-empty message to stderr
            self.assertNotEqual(result.stderr.strip(), "",
                                "CLI must write an error message to stderr")
            # stdout must NOT be parseable as the success JSON
            stdout_is_json = True
            try:
                json.loads(result.stdout)
            except (json.JSONDecodeError, ValueError):
                stdout_is_json = False
            self.assertFalse(stdout_is_json,
                             "CLI stdout must not be valid JSON on error")
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
