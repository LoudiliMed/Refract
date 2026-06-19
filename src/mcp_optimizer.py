"""
mcp_optimizer — applies the Token Optimizer pattern to MCP tool definitions.

Problem: an MCP client loads ALL tool JSON schemas into context on every
request -> huge recurring token cost.

Method (reuses the project pattern):
  TIER 1  build_index(tools, defs) -> compact index {name: short desc} + shared
                                  $defs kept ONCE (always loaded)
  TIER 2  compress_tool(tool)  -> compressed schema on demand (essential params,
                                  condensed desc, $ref = compact pointer to the
                                  $defs in the index, verbose stripped)

A (raw schemas)  ->  B (index + compressed schemas on-demand)  at reduced cost.
"""

from __future__ import annotations

import json
import re

from ast_extractor import count_tokens


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
# boilerplate prefixes from real schemas (Google: "Optional. ...", "Required. ...",
# "Output only. ...", "Deprecated: ...") — without this, first_sentence only keeps
# the empty word "Optional." and discards the useful sentence.
_BOILER = re.compile(r"^\s*(optional|required|output only|deprecated|read-only)\b[.:]?\s*",
                     re.IGNORECASE)


def first_sentence(text: str, maxlen: int = 140) -> str:
    if not text:
        return ""
    text = text.strip()
    prev = None
    while prev != text:                       # strip stacked boilerplate prefixes
        prev = text
        text = _BOILER.sub("", text)
    s = re.split(r"(?<=[.!?])\s", text)[0]
    return s[:maxlen].rstrip()


def _meta(p: dict, out: dict) -> dict:
    """Carries over cross-cutting signal (desc/default/format) to the compressed form."""
    if p.get("description"):
        out["d"] = first_sentence(p["description"], 80)
    if "default" in p:
        out["default"] = p["default"]               # default value = signal
    if p.get("format"):
        out["fmt"] = p["format"]                     # uri / date-time / byte…
    return out


def compress_param(p: dict) -> dict:
    # shared $ref: do NOT inline (otherwise the definition is duplicated in
    # each tool). Keep a compact pointer to the $defs kept once in the index (TIER 1).
    if "$ref" in p:
        return {"ref": p["$ref"].split("/")[-1]}

    # anyOf / oneOf: JSON-Schema unions. Very common with Pydantic servers
    # (an optional field = anyOf[T, null]). Without this we lost the TYPE entirely.
    union = p.get("anyOf") or p.get("oneOf")
    if union:
        branches = [b for b in union if isinstance(b, dict)]
        non_null = [b for b in branches if b.get("type") != "null"]
        out: dict = {}
        if len(non_null) == 1:                       # optional T -> compress T
            out = compress_param(non_null[0])
        elif non_null:                               # real union -> keep all branches
            out["any"] = [compress_param(b) for b in non_null]
        if len(non_null) < len(branches):
            out["null"] = 1                          # nullable (at least one null branch)
        # discriminated oneOf: the discriminant tells the model WHICH field
        # selects the branch -> critical signal. Keep propertyName + mapping
        # (value -> $ref name, compact pointer resolved from index).
        disc = p.get("discriminator")
        if "any" in out and isinstance(disc, dict) and disc.get("propertyName"):
            d = {"prop": disc["propertyName"]}
            mp = disc.get("mapping")
            if isinstance(mp, dict):
                d["map"] = {k: str(v).split("/")[-1] for k, v in mp.items()}
            out["disc"] = d
        return _meta(p, out)

    out = {}
    if "type" in p:
        out["t"] = p["type"]
    if "enum" in p:
        out["enum"] = p["enum"]                      # strong signal -> kept
    if "pattern" in p:
        out["pat"] = p["pattern"]                    # format constraint (regex) -> signal
    if "minimum" in p:
        out["min"] = p["minimum"]                    # numeric bound -> low cost
    if "maximum" in p:
        out["max"] = p["maximum"]
    if p.get("type") == "array" and isinstance(p.get("items"), dict):
        # recurse on items: captures nested $ref / enum / type (e.g. array of
        # Attendee, or array of enum) — otherwise this signal is lost.
        of = compress_param(p["items"])
        if of:
            out["of"] = of

    # properties: top-level + MERGE of allOf sub-schemas.
    # allOf = "valid against ALL sub-schemas" -> object composition.
    # We merge properties from inline branches (the compressible component)
    # and keep $ref branches as pointers (never inlined) in `all`.
    props: dict = {}
    if isinstance(p.get("properties"), dict):        # nested object (after $ref)
        props.update({k: compress_param(v) for k, v in p["properties"].items()})
    all_refs: list = []
    for sub in p.get("allOf") or []:
        if not isinstance(sub, dict):
            continue
        cs = compress_param(sub)
        if "props" in cs:
            props.update(cs["props"])                # merge properties
        if "ref" in cs:
            all_refs.append(cs)                      # $ref branch -> kept as pointer
        for k in ("t", "enum", "of"):                # carry over type/enum/items from sub-schema
            if k in cs and k not in out:
                out[k] = cs[k]
    if props:
        out["props"] = props
        out.setdefault("t", "object")
    if all_refs:
        out["all"] = all_refs                        # $ref sub-schemas to also satisfy
    return _meta(p, out)


def compress_tool(tool: dict, defs: dict | None = None) -> dict:
    """Compressed schema: condensed desc + essential params (type/req/enum/short desc).

    $ref pointers to the shared $defs remain compact ({ref: Name}):
    the definition is NOT inlined here, it lives once in the index.
    """
    sch = tool.get("inputSchema", {})
    props = sch.get("properties", {})
    required = set(sch.get("required", []))
    params = {}
    for name, p in props.items():
        cp = compress_param(p)
        if name in required:
            cp["req"] = 1
        params[name] = cp
    return {"name": tool["name"],
            "desc": first_sentence(tool.get("description", "")),
            "params": params}


def compress_defs(defs: dict) -> dict:
    """Compresses the shared $defs once (kept in the index, TIER 1)."""
    return {name: compress_param(d) for name, d in (defs or {}).items()}


def collect_defs(tools: list[dict]) -> dict:
    """Aggregates `$defs` EMBEDDED in each `inputSchema` (real-world case: actual
    MCP servers — Gmail, Calendar… — embed their definitions per tool, draft-07).

    Multiple tools duplicate the SAME definitions (e.g. Calendar: `Attendee`,
    `Reminder`… repeated in `create_event` AND `update_event`). Deduplicated
    by name -> kept ONCE in the index (TIER 1).
    First definition seen wins (they are identical across tools in practice).
    """
    defs: dict = {}
    for t in tools:
        embedded = (t.get("inputSchema", {}) or {}).get("$defs") or {}
        for name, d in embedded.items():
            defs.setdefault(name, d)
    return defs


# --------------------------------------------------------------------------- #
# safety guards: no unhandled construct should be silently dropped.
# compress_param does its best; these two probes list what it cannot yet
# translate losslessly -> mcp_signal_check turns them into FLAGs.
# --------------------------------------------------------------------------- #
# JSON-Schema keys that compress_param does NOT yet preserve
UNHANDLED_KEYS = frozenset({
    # allOf: handled (STEP 2.1) -> property merge + $ref kept as pointers.
    # discriminator: handled (STEP 2.3) -> propertyName + mapping preserved ("disc" key).
    # minimum/maximum: handled (STEP 2.4) -> bounds preserved ("min"/"max" keys).
    "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    # pattern: handled (STEP 2.2) -> preserved as signal ("pat" key).
    "minLength", "maxLength", "minItems", "maxItems",  # constraints not yet handled
})


def unhandled_constructs(node: dict) -> set[str]:
    """Recursively lists unhandled constructs present in a raw schema.

    Traverses properties / items / anyOf / oneOf / allOf. Does NOT follow `$ref`
    (kept as pointers; cycles handled by `ref_cycles`). Safety guard: anything
    found here is listed then FLAGged, never silently dropped.
    """
    found: set[str] = set()

    def walk(s) -> None:
        if not isinstance(s, dict):
            return
        found.update(UNHANDLED_KEYS & s.keys())
        if isinstance(s.get("items"), dict):
            walk(s["items"])
        if isinstance(s.get("properties"), dict):
            for v in s["properties"].values():
                walk(v)
        for key in ("anyOf", "oneOf", "allOf"):
            for b in s.get(key) or []:
                walk(b)

    walk(node)
    return found


def ref_cycles(defs: dict) -> set[str]:
    """Detects `$defs` caught in a `$ref` cycle (prevents infinite loop / crash).

    Builds the graph name -> referenced names, then colored DFS: an edge to a
    GRAY node (currently on the stack) = cycle. Returns the names involved.
    """
    def refs_of(s, acc: set[str]) -> set[str]:
        if isinstance(s, dict):
            if "$ref" in s:
                acc.add(s["$ref"].split("/")[-1])
            for v in s.values():
                refs_of(v, acc)
        elif isinstance(s, list):
            for v in s:
                refs_of(v, acc)
        return acc

    graph = {name: refs_of(d, set()) for name, d in (defs or {}).items()}
    color: dict[str, int] = {n: 0 for n in graph}        # 0 white, 1 gray, 2 black
    stack: list[str] = []
    in_cycle: set[str] = set()

    def dfs(n: str) -> None:
        color[n] = 1
        stack.append(n)
        for m in graph.get(n, ()):
            if m not in graph:                           # external/unknown ref -> skip
                continue
            if color[m] == 1:                            # back edge -> cycle
                in_cycle.update(stack[stack.index(m):])
            elif color[m] == 0:
                dfs(m)
        stack.pop()
        color[n] = 2

    for n in graph:
        if color[n] == 0:
            dfs(n)
    return in_cycle


def build_index(tools: list[dict], defs: dict | None = None) -> dict:
    """TIER 1: always loaded. Names -> short sentence + shared $defs kept once."""
    idx: dict = {"tools": {t["name"]: first_sentence(t.get("description", ""), 70)
                           for t in tools}}
    if defs:
        idx["$defs"] = compress_defs(defs)
    return idx


# --------------------------------------------------------------------------- #
# benchmark
# --------------------------------------------------------------------------- #
def _j(o) -> str:
    return json.dumps(o, ensure_ascii=False)


if __name__ == "__main__":
    import os
    _schemas = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schemas")
    data = json.load(open(os.path.join(_schemas, "mcp_fixtures.json"), encoding="utf-8"))
    tools = data["tools"]
    defs = data.get("$defs", {})

    raw_all = count_tokens(_j(tools))
    index = build_index(tools, defs)
    idx_tok = count_tokens(_j(index))

    print(f"{len(tools)} MCP tools\n")
    print(f"{'':34}{'tokens':>8}")
    print("-" * 44)
    print(f"{'A — all raw schemas':<34}{raw_all:>8}   (loaded on EVERY request today)")
    print(f"{'TIER 1 — full index':<34}{idx_tok:>8}   (always loaded)")

    print(f"\n{'tool':<26}{'raw':>7}{'compressed':>10}{'gain':>7}")
    print("-" * 50)
    tot_raw = tot_comp = 0
    for t in tools:
        rb = count_tokens(_j(t))
        cb = count_tokens(_j(compress_tool(t, defs)))
        tot_raw += rb
        tot_comp += cb
        print(f"{t['name']:<26}{rb:>7}{cb:>10}{(1-cb/rb)*100:>6.0f}%")
    print("-" * 50)
    print(f"{'TOTAL schemas':<26}{tot_raw:>7}{tot_comp:>10}{(1-tot_comp/tot_raw)*100:>6.0f}%")

    print("\n=== Scenario: one request uses k tools ===")
    print(f"{'k':>3}{'  today (A)':>18}{'  two-tier (B)':>19}{'  reduction':>12}")
    by_name = {t["name"]: t for t in tools}
    used = ["github_create_issue", "fs_read_file", "slack_post_message"]
    for k in (1, 2, 3):
        sel = used[:k]
        b = idx_tok + sum(count_tokens(_j(compress_tool(by_name[n], defs))) for n in sel)
        print(f"{k:>3}{raw_all:>18}{b:>19}{(1-b/raw_all)*100:>11.0f}%")
