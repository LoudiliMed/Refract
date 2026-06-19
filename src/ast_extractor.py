"""
ast_extractor — Token Optimizer (consolidated implementation).

Two levels:
  * extract(source)      -> flat contract {imports, functions} (P1 contract)
  * compress(source)     -> compressed S5 bundle (layered view + deps)
  * count_tokens(text)   -> token count via tiktoken

CLI: python ast_extractor.py <file.py>  -> flat contract JSON on stdout.
"""

from __future__ import annotations

import ast
import json
import sys


# =========================================================================== #
# Flat contract  (tested by test_extractor.py)
# =========================================================================== #
def _module_function_names(tree: ast.AST) -> set[str]:
    return {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _extract_imports(tree: ast.AST) -> list[str]:
    """Top-level module names, deduplicated, without dotted parts or class names."""
    out: list[str] = []
    seen: set[str] = set()

    def add(name: str):
        if name and name not in seen:
            seen.add(name)
            out.append(name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # relative import ('from . import x') -> module is None: skip, no crash
            if node.module:
                add(node.module.split(".")[0])
    return out


def _extract_function(node, func_names: set[str]) -> dict:
    params = [a.arg for a in node.args.args]
    calls: list[str] = []
    seen: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            name = n.func.id
            if name in func_names and name not in seen:   # internal calls only
                seen.add(name)
                calls.append(name)
    return {"nom": node.name, "params": params, "appels": calls}


def extract(source: str) -> dict:
    """Flat contract: {'imports': [...], 'fonctions': [{nom, params, appels}]}."""
    tree = ast.parse(source)
    func_names = _module_function_names(tree)
    fonctions = [
        _extract_function(n, func_names)
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    return {"imports": _extract_imports(tree), "fonctions": fonctions}


def _appels_complets(node) -> list[str]:
    """All calls (full dotted name), deduplicated, in order.
    Fixes the flat contract where `appels` (internal only) was empty on real code.
    Format matches summary.md: ['bcrypt.checkpw', 'jwt.encode'...]."""
    out, seen = [], set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            try:
                name = ast.unparse(n.func)
            except Exception:
                continue
            if name and name not in seen:
                seen.add(name)
                out.append(name)
    return out


def extract_json(source: str) -> dict:
    """Fixed JSON output: appels = all calls (summary.md contract)."""
    tree = ast.parse(source)
    fonctions = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            f = {"nom": n.name, "params": [a.arg for a in n.args.args],
                 "appels": _appels_complets(n)}
            if (doc := ast.get_docstring(n)):
                f["doc"] = doc.strip().splitlines()[0]
            fonctions.append(f)
    return {"imports": _extract_imports(tree), "fonctions": fonctions}


# =========================================================================== #
# S5 engine: layered view + dependencies
# =========================================================================== #
class IndexModule:
    """Index of module-level definitions (the 'vocabulary')."""

    def __init__(self, tree: ast.AST):
        self.data: set[str] = set()
        self.classes: set[str] = set()
        self.fonctions: set[str] = set()
        self.imports: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        self.data.add(t.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                self.data.add(node.target.id)
            elif isinstance(node, ast.ClassDef):
                self.classes.add(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.fonctions.add(node.name)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    self.imports.add((a.asname or a.name).split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    self.imports.add(a.asname or a.name)


class ClassIndex:
    """Class vocabulary: attributes (self.x + class-level) + methods.
    Plays for `self.x` the same role IndexModule plays for module-level names."""

    def __init__(self, node: ast.ClassDef):
        self.methods: set[str] = set()
        self.attrs: set[str] = set()        # all self.X = ... + class-level attrs
        self.attrs_init: set[str] = set()   # those defined in __init__/__new__
        for m in node.body:
            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.methods.add(m.name)
            elif isinstance(m, ast.Assign):
                for t in m.targets:
                    if isinstance(t, ast.Name):
                        self.attrs.add(t.id)
            elif isinstance(m, ast.AnnAssign) and isinstance(m.target, ast.Name):
                self.attrs.add(m.target.id)
        for m in node.body:                  # instance attributes (self.X = ...)
            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                target = self.attrs_init if m.name in ("__init__", "__new__") else None
                for n in ast.walk(m):
                    if (isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
                            and isinstance(n.value, ast.Name) and n.value.id == "self"):
                        self.attrs.add(n.attr)
                        if target is not None:
                            target.add(n.attr)


# names that are NOT real data dependencies (noise)
_TYPEVAR_NOISE = {"T", "R", "V", "K", "S", "U", "_", "P"}


def dependances(node, idx: IndexModule, cls: "ClassIndex | None" = None) -> dict:
    """Real dependencies: reads Name LOADs (not just Calls).
    If `cls` is provided, also resolves `self.attr` / `self.method()` against the
    class vocabulary (otherwise those accesses are ignored, pure S5 behavior)."""
    locals_: set[str] = {a.arg for a in node.args.args}
    locals_ |= {a.arg for a in node.args.kwonlyargs}
    if node.args.vararg:
        locals_.add(node.args.vararg.arg)
    if node.args.kwarg:
        locals_.add(node.args.kwarg.arg)
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            locals_.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not node:
            locals_.add(n.name)

    data, interne, externe, typ = set(), set(), set(), set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            name = n.id
            if name in locals_ or name in _TYPEVAR_NOISE:
                continue
            if name in idx.data:
                data.add(name)
            elif name in idx.fonctions:
                interne.add(name)
            elif name in idx.classes:
                typ.add(name)
            elif name in idx.imports:
                externe.add(name)
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            root = n.value.id
            if root in idx.imports and root not in locals_:
                externe.add(root)
            elif (cls is not None and root == "self"
                  and isinstance(n.ctx, ast.Load)):
                if n.attr in cls.methods:
                    interne.add("self." + n.attr)        # sibling method -> ORCH
                elif n.attr in cls.attrs:
                    data.add("self." + n.attr)           # class vocabulary -> TOOL

    return {"data": sorted(data), "type": sorted(typ),
            "interne": sorted(interne), "externe": sorted(externe)}


def role(deps: dict) -> str:
    n_int = len(deps["interne"])
    if n_int >= 2:
        return "ORCH"
    if deps["data"] or deps["type"] or deps["externe"]:
        return "TOOL"
    return "PURE"


# --------------------------------------------------------------------------- #
# S5 compressed bundle rendering (text)
# --------------------------------------------------------------------------- #
def _sig(node) -> str:
    params = []
    a = node.args
    defaults = [None] * (len(a.args) - len(a.defaults)) + list(a.defaults)
    for arg, d in zip(a.args, defaults):
        s = arg.arg + (f": {ast.unparse(arg.annotation)}" if arg.annotation else "")
        if d is not None:
            s += f" = {ast.unparse(d)}"
        params.append(s)
    if a.vararg:
        params.append("*" + a.vararg.arg)
    for arg, d in zip(a.kwonlyargs, a.kw_defaults):
        s = arg.arg + (f" = {ast.unparse(d)}" if d is not None else "")
        params.append(s)
    if a.kwarg:
        params.append("**" + a.kwarg.arg)
    pre = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{pre}def {node.name}({', '.join(params)}){ret}:"


def _doc(node, indent="    "):
    d = ast.get_docstring(node)
    return f'{indent}"""{d.strip().splitlines()[0]}"""' if d else None


def _own_nodes(node):
    """Descends into `node`'s body WITHOUT entering nested functions/lambdas,
    to correctly attribute raise/yield/warn to the right function."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        yield child
        yield from _own_nodes(child)


def _contract_tags(node) -> dict:
    """Invisible contracts not in the signature, extracted WITHOUT keeping the body:
    raises (exception types), yields (generator), warns (warnings emitted).
    ~0 tokens, recovers the most dangerous signal from stripped functions."""
    raises, warns, is_gen = [], [], False
    for n in _own_nodes(node):
        if isinstance(n, ast.Raise) and n.exc:
            exc = n.exc.func if isinstance(n.exc, ast.Call) else n.exc
            try:
                raises.append(ast.unparse(exc))
            except Exception:
                pass
        elif isinstance(n, (ast.Yield, ast.YieldFrom)):
            is_gen = True
        elif isinstance(n, ast.Call):
            try:
                name = ast.unparse(n.func)
            except Exception:
                name = ""
            if "warn" in name.lower():
                warns.append(name)
    tags = {}
    if raises:
        tags["raises"] = list(dict.fromkeys(raises))
    if is_gen:
        tags["yields"] = True
    if warns:
        tags["warns"] = list(dict.fromkeys(warns))
    return tags


def _render_func(node, idx, indent="", tags=False, cls=None):
    deps = dependances(node, idx, cls)
    r = role(deps)
    if r == "PURE":                                   # body kept verbatim
        return ["\n".join(indent + l for l in ast.unparse(node).splitlines())]
    lines = [indent + _sig(node)]
    doc = _doc(node, indent + "    ")
    if doc:
        lines.append(doc)
    needed = {k: v for k, v in deps.items() if v}
    if tags:
        needed.update(_contract_tags(node))           # raises/yields/warns
    if needed:
        lines.append(indent + "    # needs: " + json.dumps(needed))
    return lines


def _render_class(node, idx, tags=False, resolve_self=False):
    bases = ", ".join(ast.unparse(b) for b in node.bases)
    out = [f"class {node.name}({bases}):"]
    d = _doc(node)
    if d:
        out.append(d)
    cls = ClassIndex(node) if resolve_self else None

    if resolve_self:
        # class vocabulary kept once (class-level attrs)
        for m in node.body:
            if isinstance(m, (ast.Assign, ast.AnnAssign)):
                out.append("    " + ast.unparse(m))
        # instance attributes defined outside __init__: declared in manifest
        # (otherwise a self.x dep would be named without being anchored in the bundle)
        outside_init = sorted(cls.attrs - cls.attrs_init)
        if outside_init:
            out.append("    # attrs: " + ", ".join(outside_init))

    for m in node.body:
        if not isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if resolve_self and m.name in ("__init__", "__new__"):
            # body defines the instance vocabulary -> kept verbatim
            out.append("\n".join("    " + l for l in ast.unparse(m).splitlines()))
        else:
            out += _render_func(m, idx, indent="    ", tags=tags, cls=cls)
    return out


def compress(source: str, tags: bool = False, resolve_self: bool = False) -> str:
    """S5 bundle: vocabulary kept + functions compressed by role.

    tags=False (default)    : pure S5 (body stripped = signature + deps).
    tags=True               : adds raises/yields/warns contracts to # needs,
                              without keeping the body (~0 token cost, signal recovered).
    resolve_self=False (def): self.attr/self.method() ignored (pure S5).
    resolve_self=True       : resolves self.x against the class vocabulary and
                              strips OO methods. Signal 100% preserved (attrs
                              + __init__ kept), BUT net compression is NEGATIVE
                              (-6 pts measured on 6 repos, see bench_self.py): OO
                              method bodies are too short for 'signature + needs +
                              vocabulary' to be more compact.
                              Keep OFF for compression; useful for self.x closure
                              resolution in targeted context (expand)."""
    tree = ast.parse(source)
    idx = IndexModule(tree)
    out: list[str] = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)):
            out.append(ast.unparse(node))            # imports + DATA vocabulary
        elif isinstance(node, ast.ClassDef):
            out += _render_class(node, idx, tags=tags, resolve_self=resolve_self)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out += _render_func(node, idx, tags=tags)

    return "\n".join(out)


# =========================================================================== #
# Tokens
# =========================================================================== #
def count_tokens(text: str) -> int:
    import tiktoken
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


# =========================================================================== #
# CLI
# =========================================================================== #
def main(argv):
    if len(argv) != 2:
        print("Usage: python ast_extractor.py <file.py>", file=sys.stderr)
        return 1
    try:
        with open(argv[1], "r", encoding="utf-8") as f:
            source = f.read()
        result = extract(source)
    except SyntaxError as e:
        print(f"Syntax error: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"File error: {e}", file=sys.stderr)
        return 3
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
