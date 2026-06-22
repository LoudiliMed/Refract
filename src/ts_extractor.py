"""
ts_extractor — JavaScript / TypeScript counterpart to ast_extractor.

Mirrors ast_extractor's public surface, backed by tree-sitter instead of the
stdlib ``ast`` module:

  * extract(source, language)   -> {imports, classes, fonctions}
  * compress(source, language)  -> compressed string (bodies stripped)
  * count_tokens(text)          -> token count via tiktoken
  * expand(source, targets, language) -> verbatim defs + dependency context

Supported languages: "javascript" (also .jsx), "typescript", "tsx".

tree-sitter is error-tolerant: a malformed file still yields a tree (with ERROR
nodes) rather than raising. Callers should still guard against ImportError /
ValueError so Python-only deployments keep working when the JS grammars or the
tree-sitter wheel are absent.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Node types that declare a function with its own body.
_FUNC_DECL = frozenset({
    "function_declaration",
    "generator_function_declaration",
})
# Function *expressions* (assigned to a name via a declarator/field).
_FUNC_EXPR = frozenset({
    "arrow_function",
    "function_expression",
    "generator_function",
})
_METHODS = frozenset({
    "method_definition",
    "method_signature",
    "abstract_method_signature",
})
_CLASSES = frozenset({"class_declaration", "abstract_class_declaration"})
# Top-level nodes kept verbatim by compress() — they are pure "vocabulary"
# (types / interfaces / enums), already body-free.
_VERBATIM_DECL = frozenset({
    "interface_declaration",
    "type_alias_declaration",
    "enum_declaration",
})

_LANG_ALIASES = {
    "js": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "javascript": "javascript",
    "typescript": "typescript",
    "tsx": "tsx",
}

# Cache parsers per language (building a Language is cheap, but reuse is tidy).
_PARSERS: dict[str, object] = {}


# =========================================================================== #
# Parser plumbing
# =========================================================================== #
def _get_parser(language: str):
    """Return a cached tree-sitter Parser for *language*.

    Raises ValueError for an unknown language and ImportError (via RuntimeError)
    when the tree-sitter wheels are not installed.
    """
    lang = _LANG_ALIASES.get(language.lower())
    if lang is None:
        raise ValueError(f"Unsupported language: {language!r}")

    if lang in _PARSERS:
        return _PARSERS[lang]

    try:
        from tree_sitter import Language, Parser
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            f"tree-sitter not installed: {exc}. "
            "Run `pip install tree-sitter tree-sitter-javascript tree-sitter-typescript`"
        ) from exc

    if lang == "javascript":
        import tree_sitter_javascript as ts_js
        ts_lang = Language(ts_js.language())
    elif lang == "typescript":
        import tree_sitter_typescript as ts_ts
        ts_lang = Language(ts_ts.language_typescript())
    else:  # tsx
        import tree_sitter_typescript as ts_ts
        ts_lang = Language(ts_ts.language_tsx())

    parser = Parser(ts_lang)
    _PARSERS[lang] = parser
    return parser


def _parse(source: str, language: str):
    """Parse *source*; returns (root_node, source_bytes)."""
    parser = _get_parser(language)
    data = source.encode("utf-8")
    tree = parser.parse(data)
    return tree.root_node, data


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _unwrap_export(node):
    """If *node* is an export_statement, return the wrapped declaration (or None)."""
    if node.type != "export_statement":
        return node
    decl = node.child_by_field_name("declaration")
    if decl is not None:
        return decl
    # `export default <decl>` / `export { ... }` — find a declaration-ish child.
    for c in node.named_children:
        if c.type in _FUNC_DECL or c.type in _CLASSES or c.type in (
            "lexical_declaration", "variable_declaration", *_VERBATIM_DECL
        ):
            return c
    return None


# =========================================================================== #
# Names / params / calls
# =========================================================================== #
def _name_of(node, src: bytes) -> str | None:
    n = node.child_by_field_name("name")
    return _text(n, src) if n is not None else None


def _func_value_declarator(decl_node):
    """For a lexical/variable declaration, yield (name_node, func_node) pairs
    where the declared value is a function expression."""
    for vd in decl_node.named_children:
        if vd.type != "variable_declarator":
            continue
        val = vd.child_by_field_name("value")
        if val is not None and val.type in _FUNC_EXPR:
            yield vd.child_by_field_name("name"), val


def _param_names(params_node, src: bytes) -> list[str]:
    """Extract parameter names (handles JS patterns + TS typed parameters)."""
    if params_node is None:
        return []
    out: list[str] = []
    for p in params_node.named_children:
        out.append(_one_param_name(p, src))
    return out


def _one_param_name(p, src: bytes) -> str:
    t = p.type
    if t == "identifier":
        return _text(p, src)
    if t in ("required_parameter", "optional_parameter"):  # TS
        pat = p.child_by_field_name("pattern")
        return _text(pat, src) if pat is not None else _text(p, src)
    if t == "assignment_pattern":  # JS default value
        left = p.child_by_field_name("left")
        return _text(left, src) if left is not None else _text(p, src)
    if t == "rest_pattern":
        return _text(p, src)
    # object_pattern / array_pattern (destructuring) — keep the literal text.
    return _text(p, src)


def _module_function_names(root, src: bytes) -> set[str]:
    """Names callable by a bare identifier: top-level function declarations and
    `const fn = () => …` style declarators (anywhere in the module)."""
    names: set[str] = set()

    def visit(node):
        if node.type in _FUNC_DECL:
            nm = _name_of(node, src)
            if nm:
                names.add(nm)
        elif node.type in ("lexical_declaration", "variable_declaration"):
            for name_node, _ in _func_value_declarator(node):
                if name_node is not None and name_node.type == "identifier":
                    names.add(_text(name_node, src))
        for c in node.named_children:
            visit(c)

    visit(root)
    return names


def _import_specifiers(root, src: bytes) -> list[str]:
    """Module specifiers from `import … from "x"` and `require("x")`, in order."""
    out: list[str] = []
    seen: set[str] = set()

    def add(spec: str):
        spec = spec.strip("'\"`")
        if spec and spec not in seen:
            seen.add(spec)
            out.append(spec)

    def visit(node):
        if node.type == "import_statement":
            srcf = node.child_by_field_name("source")
            if srcf is not None:
                add(_text(srcf, src))
        elif node.type == "call_expression":
            fn = node.child_by_field_name("function")
            if fn is not None and _text(fn, src) == "require":
                args = node.child_by_field_name("arguments")
                if args is not None:
                    for a in args.named_children:
                        if a.type == "string":
                            add(_text(a, src))
        for c in node.named_children:
            visit(c)

    visit(root)
    return out


def _import_bindings(root, src: bytes) -> set[str]:
    """Local identifiers introduced by imports (default / named / namespace)."""
    names: set[str] = set()

    def visit(node):
        if node.type == "import_clause":
            for ident in _descendants_of_type(node, "identifier"):
                names.add(_text(ident, src))
        for c in node.named_children:
            visit(c)

    visit(root)
    return names


def _descendants_of_type(node, type_name: str):
    for c in node.named_children:
        if c.type == type_name:
            yield c
        yield from _descendants_of_type(c, type_name)


def _internal_calls(fn_node, func_names: set[str], src: bytes) -> list[str]:
    """Bare-identifier calls inside *fn_node* that target module functions."""
    out: list[str] = []
    seen: set[str] = set()

    def visit(node):
        if node.type == "call_expression":
            fn = node.child_by_field_name("function")
            if fn is not None and fn.type == "identifier":
                name = _text(fn, src)
                if name in func_names and name not in seen:
                    seen.add(name)
                    out.append(name)
        for c in node.named_children:
            visit(c)

    body = fn_node.child_by_field_name("body")
    if body is not None:
        visit(body)
    return out


def _external_refs(fn_node, bindings: set[str], src: bytes) -> list[str]:
    """Imported identifiers referenced inside *fn_node*."""
    out: list[str] = []
    seen: set[str] = set()
    body = fn_node.child_by_field_name("body") or fn_node
    for ident in _descendants_of_type(body, "identifier"):
        name = _text(ident, src)
        if name in bindings and name not in seen:
            seen.add(name)
            out.append(name)
    return out


# =========================================================================== #
# extract
# =========================================================================== #
def extract(source: str, language: str = "javascript") -> dict:
    """Flat contract for JS/TS: {imports, classes, fonctions}.

    - imports:   module specifiers (`"fs"`, `"./foo"`), import + require.
    - classes:   class names (incl. abstract classes).
    - fonctions: every named function / method, as {nom, params, appels},
                 where ``appels`` lists internal calls (module functions only),
                 matching ast_extractor.extract semantics.
    """
    root, src = _parse(source, language)
    func_names = _module_function_names(root, src)

    classes: list[str] = []
    fonctions: list[dict] = []

    def visit(node):
        if node.type in _CLASSES:
            nm = _name_of(node, src)
            if nm:
                classes.append(nm)
        elif node.type in _FUNC_DECL:
            _add_function(node, _name_of(node, src))
        elif node.type in _METHODS:
            _add_function(node, _name_of(node, src))
        elif node.type in ("lexical_declaration", "variable_declaration"):
            for name_node, func in _func_value_declarator(node):
                nm = _text(name_node, src) if name_node is not None else None
                _add_function(func, nm)
        for c in node.named_children:
            visit(c)

    def _add_function(fn_node, nm):
        if not nm:
            return
        fonctions.append({
            "nom": nm,
            "params": _param_names(fn_node.child_by_field_name("parameters"), src),
            "appels": _internal_calls(fn_node, func_names, src),
        })

    visit(root)
    return {
        "imports": _import_specifiers(root, src),
        "classes": classes,
        "fonctions": fonctions,
    }


# =========================================================================== #
# compress
# =========================================================================== #
def _sig_with_stripped_body(node, src: bytes, indent: str = "") -> str:
    """Return everything up to the body, then ``{ ... }`` (body discarded)."""
    body = node.child_by_field_name("body")
    if body is None:  # signature-only (e.g. TS overload / abstract method)
        return indent + _text(node, src).rstrip()
    sig = src[node.start_byte:body.start_byte].decode("utf-8", "replace").rstrip()
    return indent + sig + " { ... }"


def _render_class(node, src: bytes, indent: str = "") -> list[str]:
    body = node.child_by_field_name("body")
    if body is None:
        return [indent + _text(node, src)]
    header = src[node.start_byte:body.start_byte].decode("utf-8", "replace").rstrip()
    out = [indent + header + " {"]
    inner = indent + "  "
    for m in body.named_children:
        if m.type in _METHODS:
            out.append(_sig_with_stripped_body(m, src, inner))
        elif m.type in ("field_definition", "public_field_definition"):
            out.append(inner + _text(m, src).rstrip())  # class-level vocabulary
        # other members (decorators handled by their owner) are skipped
    out.append(indent + "}")
    return out


def _render_decl(node, src: bytes, prefix: str = "") -> list[str]:
    """Render one top-level declaration; *prefix* carries `export `/`export default `."""
    t = node.type

    if t in _FUNC_DECL:
        return [prefix + _sig_with_stripped_body(node, src).lstrip()]

    if t in _CLASSES:
        lines = _render_class(node, src)
        lines[0] = prefix + lines[0]
        return lines

    if t in ("lexical_declaration", "variable_declaration"):
        declarators = [c for c in node.named_children if c.type == "variable_declarator"]
        func_pairs = list(_func_value_declarator(node))
        if len(declarators) == 1 and func_pairs:
            _, func = func_pairs[0]
            body = func.child_by_field_name("body")
            if body is not None:
                head = src[node.start_byte:body.start_byte].decode("utf-8", "replace").rstrip()
                return [prefix + head + " { ... }"]
        return [prefix + _text(node, src).rstrip()]  # data vocabulary, kept verbatim

    # imports, types, interfaces, enums, and anything else: keep verbatim.
    return [prefix + _text(node, src).rstrip()]


def compress(source: str, language: str = "javascript") -> str:
    """S5-style compression for JS/TS: signatures + vocabulary kept, bodies stripped.

    Kept: imports, function/method signatures (with TS return types), class &
    method names, class fields, top-level data declarations, type/interface/enum.
    Stripped: function/method bodies (replaced by ``{ ... }``).
    """
    root, src = _parse(source, language)
    out: list[str] = []

    for node in root.named_children:
        if node.type == "export_statement":
            inner = _unwrap_export(node)
            has_default = any(c.type == "default" for c in node.children)
            prefix = "export default " if has_default else "export "
            if inner is None:
                out.append(_text(node, src).rstrip())  # `export { a, b }` etc.
            else:
                out.extend(_render_decl(inner, src, prefix=prefix))
        else:
            out.extend(_render_decl(node, src))

    return "\n".join(out)


# =========================================================================== #
# expand
# =========================================================================== #
def expand(source: str, targets: list[str], language: str = "javascript") -> dict:
    """Return named functions/classes verbatim + their dependency context.

    Mirrors the shape ast_extractor-based expansion produces in refract_server:
    per target {kind, source, [dependencies]} plus a `missing` list and the
    module's import context.
    """
    root, src = _parse(source, language)
    func_names = _module_function_names(root, src)
    bindings = _import_bindings(root, src)
    wanted = set(targets)
    found: dict[str, dict] = {}

    def visit(node):
        if not wanted - set(found):
            return
        if node.type in _CLASSES:
            nm = _name_of(node, src)
            if nm in wanted and nm not in found:
                found[nm] = {"kind": "class", "source": _text(node, src)}
        elif node.type in _FUNC_DECL or node.type in _METHODS:
            nm = _name_of(node, src)
            if nm in wanted and nm not in found:
                found[nm] = _func_entry(node, nm)
        elif node.type in ("lexical_declaration", "variable_declaration"):
            for name_node, func in _func_value_declarator(node):
                nm = _text(name_node, src) if name_node is not None else None
                if nm in wanted and nm not in found:
                    found[nm] = _func_entry(func, nm, whole=node)
        for c in node.named_children:
            visit(c)

    def _func_entry(fn_node, nm, whole=None):
        return {
            "kind": "function",
            "source": _text(whole if whole is not None else fn_node, src),
            "dependencies": {
                "data": [],
                "type": [],
                "interne": sorted(_internal_calls(fn_node, func_names, src)),
                "externe": sorted(_external_refs(fn_node, bindings, src)),
            },
        }

    visit(root)
    return {
        "targets": found,
        "missing": sorted(wanted - set(found)),
        "imports": _import_specifiers(root, src),
    }


# =========================================================================== #
# Tokens
# =========================================================================== #
def count_tokens(text: str) -> int:
    import tiktoken
    return len(tiktoken.get_encoding("cl100k_base").encode(text))
