"""
ast_extractor — Token Optimizer (solution consolidée).

Deux niveaux :
  * extract(source)      -> contrat plat {imports, fonctions} (contrat P1)
  * compress(source)     -> bundle compressé S5 (vision en couches + deps)
  * count_tokens(text)   -> tokens tiktoken (si dispo)

CLI : python ast_extractor.py <fichier.py>  -> JSON du contrat plat sur stdout.
"""

from __future__ import annotations

import ast
import json
import sys


# =========================================================================== #
# Contrat plat  (testé par test_extractor.py)
# =========================================================================== #
def _module_function_names(tree: ast.AST) -> set[str]:
    return {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _extract_imports(tree: ast.AST) -> list[str]:
    """Noms de modules top-level, dédupliqués, sans partie pointée ni classe."""
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
            # relatif ('from . import x') -> module None : on ignore, pas de crash
            if node.module:
                add(node.module.split(".")[0])
    return out


def _extract_function(node, func_names: set[str]) -> dict:
    params = [a.arg for a in node.args.args]
    appels: list[str] = []
    seen: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            nom = n.func.id
            if nom in func_names and nom not in seen:   # interne uniquement
                seen.add(nom)
                appels.append(nom)
    return {"nom": node.name, "params": params, "appels": appels}


def extract(source: str) -> dict:
    """Contrat plat : {'imports': [...], 'fonctions': [{nom, params, appels}]}."""
    tree = ast.parse(source)
    func_names = _module_function_names(tree)
    fonctions = [
        _extract_function(n, func_names)
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    return {"imports": _extract_imports(tree), "fonctions": fonctions}


def _appels_complets(node) -> list[str]:
    """Tous les appels (nom complet pointé), dédupliqués, dans l'ordre.
    Corrige le contrat plat dont `appels` (internes only) était vide sur du
    vrai code. Format conforme à summary.md : ['bcrypt.checkpw', 'jwt.encode'...]."""
    out, seen = [], set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            try:
                nom = ast.unparse(n.func)
            except Exception:
                continue
            if nom and nom not in seen:
                seen.add(nom)
                out.append(nom)
    return out


def extract_json(source: str) -> dict:
    """Sortie JSON CORRIGÉE : appels = tous les appels (contrat summary.md)."""
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
# Moteur S5 : vision en couches + dépendances
# =========================================================================== #
class IndexModule:
    """Index des définitions module-level (le 'vocabulaire')."""

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
    """Vocabulaire d'une classe : attributs (self.x + niveau classe) + méthodes.
    Joue pour `self.x` le rôle qu'IndexModule joue pour les noms module-level."""

    def __init__(self, node: ast.ClassDef):
        self.methods: set[str] = set()
        self.attrs: set[str] = set()        # tous les self.X = ... + attrs de classe
        self.attrs_init: set[str] = set()   # ceux définis dans __init__/__new__
        for m in node.body:
            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.methods.add(m.name)
            elif isinstance(m, ast.Assign):
                for t in m.targets:
                    if isinstance(t, ast.Name):
                        self.attrs.add(t.id)
            elif isinstance(m, ast.AnnAssign) and isinstance(m.target, ast.Name):
                self.attrs.add(m.target.id)
        for m in node.body:                  # attributs d'instance (self.X = ...)
            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cible = self.attrs_init if m.name in ("__init__", "__new__") else None
                for n in ast.walk(m):
                    if (isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
                            and isinstance(n.value, ast.Name) and n.value.id == "self"):
                        self.attrs.add(n.attr)
                        if cible is not None:
                            cible.add(n.attr)


# noms qui ne sont PAS de vraies dépendances de données (bruit)
_TYPEVAR_NOISE = {"T", "R", "V", "K", "S", "U", "_", "P"}


def dependances(node, idx: IndexModule, cls: "ClassIndex | None" = None) -> dict:
    """Dépendances réelles : on lit les Name LUS (pas seulement les Call).
    Si `cls` est fourni, on résout aussi `self.attr` / `self.methode()` contre le
    vocabulaire de la classe (sinon ces accès sont ignorés, comportement S5 pur)."""
    locales: set[str] = {a.arg for a in node.args.args}
    locales |= {a.arg for a in node.args.kwonlyargs}
    if node.args.vararg:
        locales.add(node.args.vararg.arg)
    if node.args.kwarg:
        locales.add(node.args.kwarg.arg)
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            locales.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not node:
            locales.add(n.name)

    data, interne, externe, typ = set(), set(), set(), set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            nom = n.id
            if nom in locales or nom in _TYPEVAR_NOISE:
                continue
            if nom in idx.data:
                data.add(nom)
            elif nom in idx.fonctions:
                interne.add(nom)
            elif nom in idx.classes:
                typ.add(nom)
            elif nom in idx.imports:
                externe.add(nom)
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            racine = n.value.id
            if racine in idx.imports and racine not in locales:
                externe.add(racine)
            elif (cls is not None and racine == "self"
                  and isinstance(n.ctx, ast.Load)):
                if n.attr in cls.methods:
                    interne.add("self." + n.attr)        # méthode soeur -> ORCH
                elif n.attr in cls.attrs:
                    data.add("self." + n.attr)           # vocabulaire de classe -> TOOL

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
# Rendu du bundle compressé S5 (texte)
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
    """Descend dans le corps de `node` SANS entrer dans les fonctions/lambdas
    imbriquées — pour attribuer un raise/yield/warn à la bonne fonction."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        yield child
        yield from _own_nodes(child)


def _contract_tags(node) -> dict:
    """Contrats invisibles dans la signature, extraits SANS garder le corps :
    raises (types levés), yields (générateur), warns (avertissements émis).
    ~0 token, rebouche le signal le plus dangereux des fonctions strippées."""
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
                nom = ast.unparse(n.func)
            except Exception:
                nom = ""
            if "warn" in nom.lower():
                warns.append(nom)
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
    if r == "PURE":                                   # corps gardé
        return ["\n".join(indent + l for l in ast.unparse(node).splitlines())]
    lignes = [indent + _sig(node)]
    doc = _doc(node, indent + "    ")
    if doc:
        lignes.append(doc)
    besoin = {k: v for k, v in deps.items() if v}
    if tags:
        besoin.update(_contract_tags(node))           # raises/yields/warns
    if besoin:
        lignes.append(indent + "    # needs: " + json.dumps(besoin))
    return lignes


def _render_class(node, idx, tags=False, resolve_self=False):
    bases = ", ".join(ast.unparse(b) for b in node.bases)
    out = [f"class {node.name}({bases}):"]
    d = _doc(node)
    if d:
        out.append(d)
    cls = ClassIndex(node) if resolve_self else None

    if resolve_self:
        # vocabulaire de classe gardé une fois (attrs niveau classe)
        for m in node.body:
            if isinstance(m, (ast.Assign, ast.AnnAssign)):
                out.append("    " + ast.unparse(m))
        # attributs d'instance définis hors __init__ : déclarés en manifeste
        # (sinon une dep self.x serait nommée sans être ancrée dans le bundle)
        hors_init = sorted(cls.attrs - cls.attrs_init)
        if hors_init:
            out.append("    # attrs: " + ", ".join(hors_init))

    for m in node.body:
        if not isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if resolve_self and m.name in ("__init__", "__new__"):
            # le corps définit le vocabulaire d'instance -> gardé verbatim
            out.append("\n".join("    " + l for l in ast.unparse(m).splitlines()))
        else:
            out += _render_func(m, idx, indent="    ", tags=tags, cls=cls)
    return out


def compress(source: str, tags: bool = False, resolve_self: bool = False) -> str:
    """Bundle S5 : vocabulaire gardé + fonctions compressées par rôle.

    tags=False (défaut)      : S5 pur (corps strippé = signature + deps).
    tags=True                : ajoute les contrats raises/yields/warns au # needs,
                               sans garder le corps (coût ~0 token, signal récupéré).
    resolve_self=False (déf.): self.attr/self.methode() ignorés (S5 pur).
    resolve_self=True        : résout self.x contre le vocabulaire de classe et
                               strippe les méthodes OO. Signal préservé à 100% (attrs
                               + __init__ gardés), MAIS coût net NÉGATIF en compression
                               (-6 pts mesurés sur 6 repos, cf. bench_self.py) : les
                               corps de méthodes OO sont trop courts pour que
                               'signature + needs + vocabulaire' soit plus compact.
                               Garder OFF pour compresser ; utile pour la résolution
                               de clôture self.x en contexte ciblé (expand)."""
    tree = ast.parse(source)
    idx = IndexModule(tree)
    out: list[str] = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)):
            out.append(ast.unparse(node))            # imports + vocabulaire DATA
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
        print("Usage: python ast_extractor.py <fichier.py>", file=sys.stderr)
        return 1
    try:
        with open(argv[1], "r", encoding="utf-8") as f:
            source = f.read()
        result = extract(source)
    except SyntaxError as e:
        print(f"Erreur de syntaxe : {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"Erreur fichier : {e}", file=sys.stderr)
        return 3
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
