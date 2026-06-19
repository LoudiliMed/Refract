"""
test_mcp — garde-fous du compresseur MCP (ÉTAPE 1) + round-trip de base.

Prouve que :
  * chaque construct NON géré (allOf, oneOf discriminé, contraintes) est DÉTECTÉ
    (jamais droppé en silence) -> FLAG dans mcp_signal_check ;
  * un cycle de $ref est détecté (ref_cycles) ET ne fait pas boucler la
    reconstruction (anti-crash) ;
  * un schéma propre ne lève aucun FLAG, et le round-trip compress -> reconstruct
    préserve le contrat appelable.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from mcp_demo import _param_to_schema, compressed_to_anthropic_tool
from mcp_optimizer import (build_index, collect_defs, compress_defs, compress_tool,
                           ref_cycles, unhandled_constructs)
from mcp_signal_check import main as signal_main


def _tool(name, schema):
    return {"name": name, "description": "desc.", "inputSchema": schema}


def _catalog_file(catalog: dict) -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(catalog, f)
    return path


class TestUnhandledDetection(unittest.TestCase):
    """unhandled_constructs recense bien chaque construct non géré."""

    def test_allof_handled_not_flagged(self):
        """ÉTAPE 2.1 : allOf est désormais géré -> plus jamais dans les non gérés."""
        sch = {"type": "object", "properties": {
            "x": {"allOf": [{"type": "object", "properties": {"a": {"type": "string"}}}]}}}
        self.assertNotIn("allOf", unhandled_constructs(sch))

    def test_discriminator_handled_not_flagged(self):
        """ÉTAPE 2.3 : oneOf discriminé géré -> discriminator plus dans les non gérés."""
        sch = {"type": "object", "properties": {
            "pet": {"oneOf": [{"$ref": "#/$defs/Cat"}, {"$ref": "#/$defs/Dog"}],
                    "discriminator": {"propertyName": "kind"}}}}
        self.assertNotIn("discriminator", unhandled_constructs(sch))

    def test_pattern_handled_not_flagged(self):
        """ÉTAPE 2.2 : pattern géré (préservé) -> plus dans les non gérés."""
        sch = {"type": "object", "properties": {
            "id": {"type": "string", "pattern": "^D[a-z]+$"}}}
        self.assertNotIn("pattern", unhandled_constructs(sch))

    def test_min_max_handled_not_flagged(self):
        """ÉTAPE 2.4 : minimum/maximum gérés -> plus dans les non gérés."""
        sch = {"type": "object", "properties": {
            "n": {"type": "integer", "minimum": 1, "maximum": 10}}}
        found = unhandled_constructs(sch)
        self.assertNotIn("minimum", found)
        self.assertNotIn("maximum", found)

    def test_nested_detection(self):
        """Un construct ENCORE non géré, enfoui dans items / anyOf, reste détecté."""
        sch = {"type": "object", "properties": {
            "arr": {"type": "array", "items": {"type": "string", "minLength": 3}},
            "u": {"anyOf": [{"type": "integer", "multipleOf": 2}, {"type": "null"}]}}}
        found = unhandled_constructs(sch)
        self.assertIn("minLength", found)           # encore non géré
        self.assertIn("multipleOf", found)          # encore non géré

    def test_clean_schema_no_flag(self):
        """Scalaires / enum / $ref / anyOf[T,null] / arrays : rien à flagger."""
        sch = {"type": "object", "properties": {
            "s": {"type": "string"},
            "e": {"type": "string", "enum": ["a", "b"]},
            "r": {"$ref": "#/$defs/Foo"},
            "opt": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "arr": {"type": "array", "items": {"$ref": "#/$defs/Foo"}}}}
        self.assertEqual(unhandled_constructs(sch), set())


class TestRefCycles(unittest.TestCase):
    """ref_cycles détecte les $defs récursifs ; la reconstruction ne boucle pas."""

    def test_self_cycle_detected(self):
        defs = {"Node": {"type": "object", "properties": {
            "child": {"$ref": "#/$defs/Node"}}}}
        self.assertEqual(ref_cycles(defs), {"Node"})

    def test_mutual_cycle_detected(self):
        defs = {"A": {"properties": {"b": {"$ref": "#/$defs/B"}}},
                "B": {"properties": {"a": {"$ref": "#/$defs/A"}}}}
        self.assertEqual(ref_cycles(defs), {"A", "B"})

    def test_acyclic_no_cycle(self):
        defs = {"A": {"properties": {"b": {"$ref": "#/$defs/B"}}},
                "B": {"properties": {"x": {"type": "string"}}}}
        self.assertEqual(ref_cycles(defs), set())

    def test_reconstruction_no_infinite_loop(self):
        """_param_to_schema sur un $ref récursif s'arrête (anti-crash)."""
        raw = {"Node": {"type": "object", "properties": {
            "child": {"$ref": "#/$defs/Node"}, "v": {"type": "string"}}}}
        comp_defs = compress_defs(raw)
        # ne doit PAS lever RecursionError
        out = _param_to_schema({"ref": "Node"}, comp_defs)
        self.assertEqual(out.get("type"), "object")
        self.assertIn("child", out["properties"])


class TestSignalCheckFlags(unittest.TestCase):
    """mcp_signal_check lève un FLAG (exit 1) sur chaque construct non géré."""

    def _run(self, schema) -> int:
        path = _catalog_file({"srv": [_tool("t", schema)]})
        try:
            return signal_main(["mcp_signal_check.py", path])
        finally:
            os.remove(path)

    def test_allof_passes(self):
        """ÉTAPE 2.1 : un allOf de composition propre ne lève plus de FLAG."""
        self.assertEqual(self._run(
            {"type": "object", "properties": {
                "x": {"allOf": [{"type": "object", "properties": {"a": {"type": "string"}}},
                                {"type": "object", "properties": {"b": {"type": "integer"}}}]}}}), 0)

    def test_discriminator_passes(self):
        """ÉTAPE 2.3 : oneOf discriminé propre (branches $ref résolues) -> pas de FLAG."""
        self.assertEqual(self._run(
            {"type": "object",
             "properties": {"pet": {
                 "oneOf": [{"$ref": "#/$defs/Cat"}, {"$ref": "#/$defs/Dog"}],
                 "discriminator": {"propertyName": "kind"}}},
             "$defs": {"Cat": {"type": "object", "properties": {"meow": {"type": "boolean"}}},
                       "Dog": {"type": "object", "properties": {"bark": {"type": "boolean"}}}}}), 0)

    def test_pattern_passes(self):
        """ÉTAPE 2.2 : pattern préservé -> plus de FLAG."""
        self.assertEqual(self._run(
            {"type": "object", "properties": {"id": {"type": "string", "pattern": "^x$"}}}), 0)

    def test_minmax_passes(self):
        """ÉTAPE 2.4 : minimum/maximum préservés -> plus de FLAG."""
        self.assertEqual(self._run(
            {"type": "object", "properties": {
                "n": {"type": "integer", "minimum": 0, "maximum": 100}}}), 0)

    def test_flag_remaining_constraint(self):
        """Une contrainte ENCORE non gérée (minLength) lève toujours le FLAG."""
        self.assertEqual(self._run(
            {"type": "object", "properties": {"s": {"type": "string", "minLength": 3}}}), 1)

    def test_flag_ref_cycle(self):
        """Catalogue avec $defs récursif -> FLAG global."""
        tool = _tool("t", {"type": "object", "properties": {"n": {"$ref": "#/$defs/Node"}},
                           "$defs": {"Node": {"type": "object", "properties": {
                               "child": {"$ref": "#/$defs/Node"}}}}})
        path = _catalog_file({"srv": [tool]})
        try:
            self.assertEqual(signal_main(["mcp_signal_check.py", path]), 1)
        finally:
            os.remove(path)

    def test_clean_catalog_passes(self):
        """Un schéma propre passe (exit 0) — pas de faux positif."""
        self.assertEqual(self._run(
            {"type": "object",
             "properties": {"name": {"type": "string", "description": "Required. The name."},
                            "kind": {"type": "string", "enum": ["a", "b"]}},
             "required": ["name"]}), 0)


class TestRoundTrip(unittest.TestCase):
    """compress_tool -> compressed_to_anthropic_tool : contrat appelable préservé."""

    def test_ref_resolved_from_index(self):
        tool = _tool("make", {
            "type": "object",
            "properties": {"who": {"$ref": "#/$defs/Person"},
                           "tags": {"type": "array", "items": {"type": "string"}}},
            "required": ["who"],
            "$defs": {"Person": {"type": "object", "properties": {
                "email": {"type": "string"}}}}})
        defs = collect_defs([tool])
        index_defs = build_index([tool], defs).get("$defs", {})
        rebuilt = compressed_to_anthropic_tool(compress_tool(tool), index_defs)
        sch = rebuilt["input_schema"]
        self.assertEqual(sch["required"], ["who"])
        self.assertEqual(sch["properties"]["who"]["properties"]["email"]["type"], "string")
        self.assertEqual(sch["properties"]["tags"]["items"]["type"], "string")

    def test_allof_merges_props_and_keeps_ref(self):
        """allOf [$ref Base, {props extra}] : props inline fusionnée + $ref résolu."""
        tool = _tool("evt", {
            "type": "object",
            "properties": {"payload": {"allOf": [
                {"$ref": "#/$defs/Base"},
                {"type": "object", "properties": {"extra": {"type": "string"}}}]}},
            "required": ["payload"],
            "$defs": {"Base": {"type": "object", "properties": {"id": {"type": "string"}}}}})
        defs = collect_defs([tool])
        index_defs = build_index([tool], defs).get("$defs", {})
        rebuilt = compressed_to_anthropic_tool(compress_tool(tool), index_defs)
        payload = rebuilt["input_schema"]["properties"]["payload"]
        # propriété inline fusionnée dans l'objet
        self.assertEqual(payload["properties"]["extra"]["type"], "string")
        # branche $ref préservée ET résolue depuis le $defs de l'index
        self.assertEqual(payload["allOf"][0]["properties"]["id"]["type"], "string")

    def test_pattern_preserved(self):
        """ÉTAPE 2.2 : la regex pattern survit au round-trip (signal pour le modèle)."""
        tool = _tool("set_id", {"type": "object",
            "properties": {"id": {"type": "string", "pattern": "^D[0-9]+$"}},
            "required": ["id"]})
        rebuilt = compressed_to_anthropic_tool(compress_tool(tool), {})
        self.assertEqual(rebuilt["input_schema"]["properties"]["id"]["pattern"], "^D[0-9]+$")

    def test_discriminated_oneof_roundtrips(self):
        """oneOf discriminé : branches $ref résolues + discriminant/mapping préservés."""
        tool = _tool("add_pet", {
            "type": "object",
            "properties": {"pet": {
                "oneOf": [{"$ref": "#/$defs/Cat"}, {"$ref": "#/$defs/Dog"}],
                "discriminator": {"propertyName": "kind",
                                  "mapping": {"cat": "#/$defs/Cat", "dog": "#/$defs/Dog"}}}},
            "required": ["pet"],
            "$defs": {"Cat": {"type": "object", "properties": {"meow": {"type": "boolean"}}},
                      "Dog": {"type": "object", "properties": {"bark": {"type": "boolean"}}}}})
        defs = collect_defs([tool])
        index_defs = build_index([tool], defs).get("$defs", {})
        pet = compressed_to_anthropic_tool(compress_tool(tool), index_defs)["input_schema"]["properties"]["pet"]
        self.assertEqual(pet["discriminator"]["propertyName"], "kind")
        self.assertEqual(pet["discriminator"]["mapping"]["cat"], "#/$defs/Cat")
        # branches reconstruites en oneOf et $ref résolus depuis l'index
        self.assertEqual(pet["oneOf"][0]["properties"]["meow"]["type"], "boolean")
        self.assertEqual(pet["oneOf"][1]["properties"]["bark"]["type"], "boolean")

    def test_min_max_preserved(self):
        """ÉTAPE 2.4 : les bornes minimum/maximum survivent au round-trip."""
        tool = _tool("paginate", {"type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 100}},
            "required": ["limit"]})
        limit = compressed_to_anthropic_tool(compress_tool(tool), {})["input_schema"]["properties"]["limit"]
        self.assertEqual(limit["minimum"], 1)
        self.assertEqual(limit["maximum"], 100)


if __name__ == "__main__":
    unittest.main()
