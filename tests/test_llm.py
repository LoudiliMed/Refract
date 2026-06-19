"""Tests CP3 — llm_client, SANS appel réseau (mock du SDK anthropic).

Vérifie : modèle Haiku 4.5, prompt strict (règles + « Je ne sais pas »), contexte
compressé injecté, sortie structurée parsée en {answer, source}.
"""
import json
import types
import unittest
from unittest import mock

import llm_client as lc


def _fake_response(payload: dict):
    """Forge une réponse SDK minimale : .content = [bloc texte JSON]."""
    block = types.SimpleNamespace(type="text", text=json.dumps(payload))
    return types.SimpleNamespace(content=[block])


SRC = (
    "MAX_PORT = 65535\n"                                  # constante module -> parse_port = TOOL
    "def parse_port(value):\n"
    "    if int(value) > MAX_PORT: raise ValueError('bad')\n"
    "    return int(value)\n"
)


class TestAskClaude(unittest.TestCase):

    def _call(self, payload):
        client = mock.MagicMock()
        client.messages.create.return_value = _fake_response(payload)
        result = lc.ask_claude("Que lève parse_port ?", SRC, client=client)
        return client, result

    def test_uses_haiku_4_5_model(self):
        client, _ = self._call({"answer": "ValueError", "source": "parse_port"})
        kwargs = client.messages.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "claude-haiku-4-5-20251001")

    def test_system_prompt_is_strict(self):
        client, _ = self._call({"answer": "ValueError", "source": "parse_port"})
        system = client.messages.create.call_args.kwargs["system"]
        self.assertIn("UNIQUEMENT", system)
        self.assertIn("Je ne sais pas", system)          # fallback documenté
        self.assertIn("Cite", system)                     # citation de la source

    def test_compressed_structure_injected_not_raw(self):
        client, _ = self._call({"answer": "ValueError", "source": "parse_port"})
        msg = client.messages.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("def parse_port(value):", msg)      # signature présente
        # le contexte est la structure compressée (avec tags), pas le code brut :
        self.assertIn("# needs", msg)
        self.assertIn("raises", msg)

    def test_structured_output_requested(self):
        client, _ = self._call({"answer": "ValueError", "source": "parse_port"})
        oc = client.messages.create.call_args.kwargs["output_config"]
        self.assertEqual(oc["format"]["type"], "json_schema")
        self.assertEqual(set(oc["format"]["schema"]["required"]), {"answer", "source"})

    def test_parses_answer_and_source(self):
        _, result = self._call({"answer": "ValueError", "source": "parse_port"})
        self.assertEqual(result["answer"], "ValueError")
        self.assertEqual(result["source"], "parse_port")

    def test_fallback_je_ne_sais_pas(self):
        _, result = self._call({"answer": lc.NE_SAIS_PAS, "source": ""})
        self.assertEqual(result["answer"], "Je ne sais pas")
        self.assertEqual(result["source"], "")


if __name__ == "__main__":
    unittest.main()
