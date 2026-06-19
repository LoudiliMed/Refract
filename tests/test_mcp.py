"""
test_mcp — Guards for the MCP compressor (STEP 1) + round-trip.

Proves that:
  * every unhandled construct (allOf, discriminated oneOf, constraints) is DETECTED
    (never silently dropped) -> FLAG in mcp_signal_check;
  * a $ref cycle is detected (ref_cycles) and reconstruction doesn't loop
    (anti-crash);
  * a clean schema raises no FLAG, and the round-trip compress -> reconstruct
    preserves the callable contract.

No imports from mcp_demo (deleted). Round-trip uses local helpers
(_expand / _anthropic_tool) that mirror the proxy's reconstruction logic
and additionally resolve $ref pointers from the compressed index $defs.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from mcp_optimizer import (build_index, collect_defs, compress_defs, compress_tool,
                           ref_cycles, unhandled_constructs)
from mcp_signal_check import main as signal_main
from refract_proxy import _param_to_schema


# ─── local reconstruction helpers (test-only) ────────────────────────────────

def _expand(cp: dict, defs: dict | None = None) -> dict:
    """Expand a compressed param to a full JSON Schema, resolving $refs from defs.

    defs is the compressed $defs dict from build_index (e.g. {"Person": {...}}).
    If a $ref name is not found in defs the pointer is kept verbatim.
    """
    defs = defs or {}
    if not isinstance(cp, dict):
        return {}
    if "ref" in cp:
        name = cp["ref"]
        return _expand(defs[name], defs) if name in defs else {"$ref": f"#/$defs/{name}"}
    out: dict = {}
    if "t" in cp:
        out["type"] = cp["t"]
    if "enum" in cp:
        out["enum"] = cp["enum"]
    if "of" in cp:
        out["items"] = _expand(cp["of"], defs)
    if "props" in cp:
        out["properties"] = {k: _expand(v, defs) for k, v in cp["props"].items()}
        out.setdefault("type", "object")
    if "d" in cp:
        out["description"] = cp["d"]
    if "pat" in cp:
        out["pattern"] = cp["pat"]
    if "min" in cp:
        out["minimum"] = cp["min"]
    if "max" in cp:
        out["maximum"] = cp["max"]
    if "any" in cp:
        branches = [_expand(b, defs) for b in cp["any"]]
        disc = cp.get("disc")
        if disc:
            d: dict = {"propertyName": disc["prop"]}
            if "map" in disc:
                d["mapping"] = {k: f"#/$defs/{v}" for k, v in disc["map"].items()}
            out["discriminator"] = d
            out["oneOf"] = branches
        else:
            out["anyOf"] = branches
    if "all" in cp:
        out["allOf"] = [_expand(b, defs) for b in cp["all"]]
    return out


def _anthropic_tool(compressed: dict, index_defs: dict | None = None) -> dict:
    """Build a tool dict with a fully-resolved input_schema from a compressed tool."""
    defs = index_defs or {}
    params = compressed.get("params", {})
    props = {k: _expand(v, defs) for k, v in params.items()}
    required = [k for k, v in params.items() if v.get("req")]
    schema: dict = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return {
        "name": compressed["name"],
        "description": compressed.get("desc", ""),
        "input_schema": schema,
    }


# ─── test fixtures ────────────────────────────────────────────────────────────

def _tool(name, schema):
    return {"name": name, "description": "desc.", "inputSchema": schema}


def _catalog_file(catalog: dict) -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(catalog, f)
    return path


# ─── fixture loader for real JSON schemas ────────────────────────────────────

def load_fixture(name: str) -> dict:
    path = os.path.join(os.path.dirname(__file__), "..", "schemas", name)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
class TestUnhandledDetection(unittest.TestCase):
    """unhandled_constructs correctly lists every unhandled construct."""

    def test_allof_handled_not_flagged(self):
        """STEP 2.1: allOf is now handled -> never in the unhandled set."""
        sch = {"type": "object", "properties": {
            "x": {"allOf": [{"type": "object", "properties": {"a": {"type": "string"}}}]}}}
        self.assertNotIn("allOf", unhandled_constructs(sch))

    def test_discriminator_handled_not_flagged(self):
        """STEP 2.3: discriminated oneOf handled -> discriminator not in unhandled."""
        sch = {"type": "object", "properties": {
            "pet": {"oneOf": [{"$ref": "#/$defs/Cat"}, {"$ref": "#/$defs/Dog"}],
                    "discriminator": {"propertyName": "kind"}}}}
        self.assertNotIn("discriminator", unhandled_constructs(sch))

    def test_pattern_handled_not_flagged(self):
        """STEP 2.2: pattern handled (preserved) -> not in unhandled."""
        sch = {"type": "object", "properties": {
            "id": {"type": "string", "pattern": "^D[a-z]+$"}}}
        self.assertNotIn("pattern", unhandled_constructs(sch))

    def test_min_max_handled_not_flagged(self):
        """STEP 2.4: minimum/maximum handled -> not in unhandled."""
        sch = {"type": "object", "properties": {
            "n": {"type": "integer", "minimum": 1, "maximum": 10}}}
        found = unhandled_constructs(sch)
        self.assertNotIn("minimum", found)
        self.assertNotIn("maximum", found)

    def test_nested_detection(self):
        """An unhandled construct buried inside items/anyOf is still detected."""
        sch = {"type": "object", "properties": {
            "arr": {"type": "array", "items": {"type": "string", "minLength": 3}},
            "u": {"anyOf": [{"type": "integer", "multipleOf": 2}, {"type": "null"}]}}}
        found = unhandled_constructs(sch)
        self.assertIn("minLength", found)
        self.assertIn("multipleOf", found)

    def test_clean_schema_no_flag(self):
        """Scalars / enum / $ref / anyOf[T,null] / arrays: nothing to flag."""
        sch = {"type": "object", "properties": {
            "s": {"type": "string"},
            "e": {"type": "string", "enum": ["a", "b"]},
            "r": {"$ref": "#/$defs/Foo"},
            "opt": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "arr": {"type": "array", "items": {"$ref": "#/$defs/Foo"}}}}
        self.assertEqual(unhandled_constructs(sch), set())


# ─────────────────────────────────────────────────────────────────────────────
class TestRefCycles(unittest.TestCase):
    """ref_cycles detects recursive $defs; reconstruction doesn't loop."""

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
        """_param_to_schema on a recursive $ref emits a pointer (no inlining = no crash)."""
        raw_defs = {"Node": {"type": "object", "properties": {
            "child": {"$ref": "#/$defs/Node"}, "v": {"type": "string"}}}}
        comp_defs = compress_defs(raw_defs)
        # The proxy emits {"$ref": ...} without inlining -> zero recursion
        out = _param_to_schema({"ref": "Node"})
        self.assertEqual(out, {"$ref": "#/$defs/Node"})
        # compress_defs on cyclic defs must not crash either
        self.assertIn("Node", comp_defs)
        self.assertIn("child", comp_defs["Node"].get("props", {}))


# ─────────────────────────────────────────────────────────────────────────────
class TestSignalCheckFlags(unittest.TestCase):
    """mcp_signal_check raises FLAG (exit 1) for every unhandled construct."""

    def _run(self, schema) -> int:
        path = _catalog_file({"srv": [_tool("t", schema)]})
        try:
            return signal_main(["mcp_signal_check.py", path])
        finally:
            os.remove(path)

    def test_allof_passes(self):
        """STEP 2.1: a clean allOf composition raises no FLAG."""
        self.assertEqual(self._run(
            {"type": "object", "properties": {
                "x": {"allOf": [{"type": "object", "properties": {"a": {"type": "string"}}},
                                {"type": "object", "properties": {"b": {"type": "integer"}}}]}}}), 0)

    def test_discriminator_passes(self):
        """STEP 2.3: clean discriminated oneOf with resolved $ref branches -> no FLAG."""
        self.assertEqual(self._run(
            {"type": "object",
             "properties": {"pet": {
                 "oneOf": [{"$ref": "#/$defs/Cat"}, {"$ref": "#/$defs/Dog"}],
                 "discriminator": {"propertyName": "kind"}}},
             "$defs": {"Cat": {"type": "object", "properties": {"meow": {"type": "boolean"}}},
                       "Dog": {"type": "object", "properties": {"bark": {"type": "boolean"}}}}}), 0)

    def test_pattern_passes(self):
        """STEP 2.2: preserved pattern -> no FLAG."""
        self.assertEqual(self._run(
            {"type": "object", "properties": {"id": {"type": "string", "pattern": "^x$"}}}), 0)

    def test_minmax_passes(self):
        """STEP 2.4: preserved minimum/maximum -> no FLAG."""
        self.assertEqual(self._run(
            {"type": "object", "properties": {
                "n": {"type": "integer", "minimum": 0, "maximum": 100}}}), 0)

    def test_flag_remaining_constraint(self):
        """An unhandled constraint (minLength) still raises FLAG."""
        self.assertEqual(self._run(
            {"type": "object", "properties": {"s": {"type": "string", "minLength": 3}}}), 1)

    def test_flag_ref_cycle(self):
        """A catalog with recursive $defs -> global FLAG."""
        tool = _tool("t", {"type": "object", "properties": {"n": {"$ref": "#/$defs/Node"}},
                           "$defs": {"Node": {"type": "object", "properties": {
                               "child": {"$ref": "#/$defs/Node"}}}}})
        path = _catalog_file({"srv": [tool]})
        try:
            self.assertEqual(signal_main(["mcp_signal_check.py", path]), 1)
        finally:
            os.remove(path)

    def test_clean_catalog_passes(self):
        """A clean schema passes (exit 0) — no false positive."""
        self.assertEqual(self._run(
            {"type": "object",
             "properties": {"name": {"type": "string", "description": "Required. The name."},
                            "kind": {"type": "string", "enum": ["a", "b"]}},
             "required": ["name"]}), 0)


# ─────────────────────────────────────────────────────────────────────────────
class TestRoundTrip(unittest.TestCase):
    """compress_tool -> _anthropic_tool: callable contract preserved."""

    def test_required_preserved(self):
        """required params survive the full compress -> reconstruct cycle."""
        tool = _tool("make", {
            "type": "object",
            "properties": {
                "who": {"$ref": "#/$defs/Person"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["who"],
            "$defs": {"Person": {"type": "object", "properties": {
                "email": {"type": "string"}}}},
        })
        defs = collect_defs([tool])
        index_defs = build_index([tool], defs).get("$defs", {})
        rebuilt = _anthropic_tool(compress_tool(tool), index_defs)
        sch = rebuilt["input_schema"]
        self.assertEqual(sch["required"], ["who"])
        self.assertEqual(sch["properties"]["tags"]["items"]["type"], "string")

    def test_ref_resolved_from_index(self):
        """$ref pointers are resolved to the actual schema from the compressed index."""
        tool = _tool("make", {
            "type": "object",
            "properties": {"who": {"$ref": "#/$defs/Person"}},
            "required": ["who"],
            "$defs": {"Person": {"type": "object", "properties": {
                "email": {"type": "string"}}}},
        })
        defs = collect_defs([tool])
        index_defs = build_index([tool], defs).get("$defs", {})
        rebuilt = _anthropic_tool(compress_tool(tool), index_defs)
        who = rebuilt["input_schema"]["properties"]["who"]
        self.assertEqual(who["properties"]["email"]["type"], "string")

    def test_allof_merges_props_and_keeps_ref(self):
        """allOf [$ref Base, {inline props}]: inline props merged + $ref branch resolved."""
        tool = _tool("evt", {
            "type": "object",
            "properties": {"payload": {"allOf": [
                {"$ref": "#/$defs/Base"},
                {"type": "object", "properties": {"extra": {"type": "string"}}}]}},
            "required": ["payload"],
            "$defs": {"Base": {"type": "object", "properties": {"id": {"type": "string"}}}},
        })
        defs = collect_defs([tool])
        index_defs = build_index([tool], defs).get("$defs", {})
        rebuilt = _anthropic_tool(compress_tool(tool), index_defs)
        payload = rebuilt["input_schema"]["properties"]["payload"]
        # inline prop merged
        self.assertEqual(payload["properties"]["extra"]["type"], "string")
        # $ref branch resolved from index
        self.assertEqual(payload["allOf"][0]["properties"]["id"]["type"], "string")

    def test_pattern_preserved(self):
        """STEP 2.2: regex pattern survives the round-trip."""
        tool = _tool("set_id", {
            "type": "object",
            "properties": {"id": {"type": "string", "pattern": "^D[0-9]+$"}},
            "required": ["id"],
        })
        rebuilt = _anthropic_tool(compress_tool(tool), {})
        self.assertEqual(
            rebuilt["input_schema"]["properties"]["id"]["pattern"], "^D[0-9]+$"
        )

    def test_discriminated_oneof_roundtrips(self):
        """discriminated oneOf: branches resolved + discriminant/mapping preserved."""
        tool = _tool("add_pet", {
            "type": "object",
            "properties": {"pet": {
                "oneOf": [{"$ref": "#/$defs/Cat"}, {"$ref": "#/$defs/Dog"}],
                "discriminator": {
                    "propertyName": "kind",
                    "mapping": {"cat": "#/$defs/Cat", "dog": "#/$defs/Dog"},
                },
            }},
            "required": ["pet"],
            "$defs": {
                "Cat": {"type": "object", "properties": {"meow": {"type": "boolean"}}},
                "Dog": {"type": "object", "properties": {"bark": {"type": "boolean"}}},
            },
        })
        defs = collect_defs([tool])
        index_defs = build_index([tool], defs).get("$defs", {})
        pet = _anthropic_tool(compress_tool(tool), index_defs)["input_schema"]["properties"]["pet"]
        self.assertEqual(pet["discriminator"]["propertyName"], "kind")
        self.assertEqual(pet["discriminator"]["mapping"]["cat"], "#/$defs/Cat")
        # branches resolved to actual schemas
        cat_schema = pet["oneOf"][0]
        dog_schema = pet["oneOf"][1]
        self.assertEqual(cat_schema["properties"]["meow"]["type"], "boolean")
        self.assertEqual(dog_schema["properties"]["bark"]["type"], "boolean")

    def test_min_max_preserved(self):
        """STEP 2.4: minimum/maximum survive the round-trip."""
        tool = _tool("paginate", {
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 100}},
            "required": ["limit"],
        })
        limit = _anthropic_tool(compress_tool(tool), {})["input_schema"]["properties"]["limit"]
        self.assertEqual(limit["minimum"], 1)
        self.assertEqual(limit["maximum"], 100)


# ─────────────────────────────────────────────────────────────────────────────
class TestFixtureCompression(unittest.TestCase):
    """Smoke tests against the real JSON schema fixtures in schemas/."""

    def test_calendar_tools_all_compress(self):
        raw = load_fixture("mcp_calendar_schemas.json")["calendar"]
        for tool in raw:
            ct = compress_tool(tool)
            self.assertIsNotNone(ct)
            self.assertEqual(ct["name"], tool["name"])

    def test_build_index_contains_all_tool_names(self):
        raw = load_fixture("mcp_calendar_schemas.json")["calendar"]
        defs = collect_defs(raw)
        index = build_index(raw, defs)
        tool_names = {t["name"] for t in raw}
        self.assertTrue(tool_names.issubset(set(index.get("tools", {}).keys())))

    def test_enterprise_schemas_compress_without_crash(self):
        raw_catalog = load_fixture("mcp_enterprise_schemas.json")
        tools = [t for lst in raw_catalog.values() for t in lst]
        for tool in tools:
            ct = compress_tool(tool)
            self.assertIn("name", ct)
            self.assertIn("params", ct)

    def test_index_is_smaller_than_raw(self):
        """The TIER 1 index must be smaller than all raw schemas loaded at once."""
        raw = load_fixture("mcp_calendar_schemas.json")["calendar"]
        defs = collect_defs(raw)
        index = build_index(raw, defs)
        from ast_extractor import count_tokens
        raw_tok = count_tokens(json.dumps(raw, ensure_ascii=False))
        idx_tok = count_tokens(json.dumps(index, ensure_ascii=False))
        self.assertLess(idx_tok, raw_tok)


if __name__ == "__main__":
    unittest.main()
