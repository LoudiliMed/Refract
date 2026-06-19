"""
mcp_optimizer — applique le patron Token Optimizer aux définitions de tools MCP.

Problème : un client MCP charge TOUS les schémas JSON des tools dans le contexte
à chaque requête -> coût de tokens récurrent énorme.

Méthode (réutilise le patron du projet) :
  TIER 1  build_index(tools, defs) -> index compact {nom: desc courte} + $defs
                                  partagé gardé UNE fois (toujours chargé)
  TIER 2  compress_tool(tool)  -> schéma comprimé à la demande (params essentiels,
                                  desc condensée, $ref = pointeur compact vers le
                                  $defs de l'index, verbeux viré)

A (schémas bruts)  ->  B (index + schémas comprimés on-demand)  à coût réduit.
"""

from __future__ import annotations

import json
import re

from ast_extractor import count_tokens   # on réutilise la base du projet


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
# préfixes de boilerplate des schémas réels (Google : "Optional. ...", "Required. ...",
# "Output only. ...", "Deprecated: ...") — sans ça, first_sentence ne garde que le mot
# vide "Optional." et jette toute la phrase utile.
_BOILER = re.compile(r"^\s*(optional|required|output only|deprecated|read-only)\b[.:]?\s*",
                     re.IGNORECASE)


def first_sentence(text: str, maxlen: int = 140) -> str:
    if not text:
        return ""
    text = text.strip()
    prev = None
    while prev != text:                       # retire les préfixes boilerplate empilés
        prev = text
        text = _BOILER.sub("", text)
    s = re.split(r"(?<=[.!?])\s", text)[0]
    return s[:maxlen].rstrip()


def _meta(p: dict, out: dict) -> dict:
    """Reporte le signal transverse (desc/default/format) sur la forme comprimée."""
    if p.get("description"):
        out["d"] = first_sentence(p["description"], 80)
    if "default" in p:
        out["default"] = p["default"]               # valeur par défaut = signal
    if p.get("format"):
        out["fmt"] = p["format"]                     # uri / date-time / byte…
    return out


def compress_param(p: dict) -> dict:
    # $ref partagé : on ne clôture PAS (sinon la définition est dupliquée dans
    # chaque tool). On garde un pointeur compact vers le $defs gardé une fois
    # dans l'index (TIER 1).
    if "$ref" in p:
        return {"ref": p["$ref"].split("/")[-1]}

    # anyOf / oneOf : unions JSON-Schema. Très courant chez les serveurs Pydantic
    # (un champ optionnel = anyOf[T, null]). Sans ça on perdait le TYPE entièrement.
    union = p.get("anyOf") or p.get("oneOf")
    if union:
        branches = [b for b in union if isinstance(b, dict)]
        non_null = [b for b in branches if b.get("type") != "null"]
        out: dict = {}
        if len(non_null) == 1:                       # optionnel T -> on compresse T
            out = compress_param(non_null[0])
        elif non_null:                               # vraie union -> on les garde toutes
            out["any"] = [compress_param(b) for b in non_null]
        if len(non_null) < len(branches):
            out["null"] = 1                          # nullable (au moins une branche null)
        # oneOf discriminé (ÉTAPE 2.3) : le discriminant dit au modèle QUEL champ
        # choisit la branche -> signal critique. On garde propertyName + mapping
        # (valeur -> nom de $ref, pointeur compact résolu depuis l'index).
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
        out["enum"] = p["enum"]                      # signal fort -> gardé
    if "pattern" in p:
        out["pat"] = p["pattern"]                    # contrainte de forme (regex) -> signal (tier enum)
    if "minimum" in p:
        out["min"] = p["minimum"]                    # borne numérique (ÉTAPE 2.4) -> peu coûteux
    if "maximum" in p:
        out["max"] = p["maximum"]
    if p.get("type") == "array" and isinstance(p.get("items"), dict):
        # récursion sur les items : capture $ref / enum / type imbriqués (ex. un
        # tableau d'Attendee, ou un tableau d'enum) — sinon ce signal est perdu.
        of = compress_param(p["items"])
        if of:
            out["of"] = of

    # propriétés : top-level + FUSION des sous-schémas allOf (ÉTAPE 2.1).
    # allOf = « valide contre TOUS les sous-schémas » -> composition d'objet.
    # On fusionne les propriétés des branches inline (la composante compressible)
    # et on garde les branches $ref comme pointeurs (jamais inlinées) dans `all`.
    props: dict = {}
    if isinstance(p.get("properties"), dict):        # objet imbriqué (après $ref)
        props.update({k: compress_param(v) for k, v in p["properties"].items()})
    all_refs: list = []
    for sub in p.get("allOf") or []:
        if not isinstance(sub, dict):
            continue
        cs = compress_param(sub)
        if "props" in cs:
            props.update(cs["props"])                # merge des propriétés
        if "ref" in cs:
            all_refs.append(cs)                      # branche $ref -> gardée en pointeur
        for k in ("t", "enum", "of"):                # report type/enum/items d'un sous-schéma
            if k in cs and k not in out:
                out[k] = cs[k]
    if props:
        out["props"] = props
        out.setdefault("t", "object")
    if all_refs:
        out["all"] = all_refs                        # sous-schémas $ref à satisfaire aussi
    return _meta(p, out)


def compress_tool(tool: dict, defs: dict | None = None) -> dict:
    """Schéma comprimé : desc condensée + params essentiels (type/req/enum/desc courte).

    Les $ref vers le $defs partagé restent des pointeurs compacts ({ref: Nom}) :
    la définition n'est PAS inlinée ici, elle vit une seule fois dans l'index.
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
    """Compresse le $defs partagé une seule fois (gardé dans l'index, TIER 1)."""
    return {name: compress_param(d) for name, d in (defs or {}).items()}


def collect_defs(tools: list[dict]) -> dict:
    """Agrège les `$defs` EMBARQUÉS dans chaque `inputSchema` (cas réel : les vrais
    serveurs MCP — Gmail, Calendar… — embarquent leurs définitions par tool, draft-07).

    Plusieurs tools dupliquent les MÊMES définitions (ex. Calendar : `Attendee`,
    `Reminder`… répétés dans `create_event` ET `update_event`). On les dédoublonne
    par nom -> on les gardera UNE seule fois dans l'index (TIER 1) = la tâche (a).
    Première définition vue gagne (elles sont identiques entre tools en pratique).
    """
    defs: dict = {}
    for t in tools:
        embedded = (t.get("inputSchema", {}) or {}).get("$defs") or {}
        for name, d in embedded.items():
            defs.setdefault(name, d)
    return defs


# --------------------------------------------------------------------------- #
# garde-fous : aucun construct non géré ne doit être droppé EN SILENCE.
# compress_param fait au mieux ; ces deux sondes recensent ce qu'il ne sait pas
# (encore) traduire sans perte -> mcp_signal_check les transforme en FLAG.
# --------------------------------------------------------------------------- #
# clés JSON-Schema que compress_param NE préserve pas encore (cf. ÉTAPE 2).
UNHANDLED_KEYS = frozenset({
    # allOf : géré (ÉTAPE 2.1) -> fusion des propriétés + $ref gardés en pointeurs.
    # discriminator : géré (ÉTAPE 2.3) -> propertyName + mapping préservés (clé "disc").
    # minimum/maximum : gérés (ÉTAPE 2.4) -> bornes préservées (clés "min"/"max").
    "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    # pattern : géré (ÉTAPE 2.2) -> préservé comme signal (clé "pat").
    "minLength", "maxLength", "minItems", "maxItems",  # contraintes encore non gérées
})


def unhandled_constructs(node: dict) -> set[str]:
    """Recense récursivement les constructs non gérés présents dans un schéma brut.

    Traverse properties / items / anyOf / oneOf / allOf. NE suit PAS les `$ref`
    (gardés en pointeurs ; les cycles sont traités par `ref_cycles`). Sert de
    garde-fou : tout ce qui est ici est listé puis FLAGé, jamais perdu en silence.
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
    """Détecte les `$defs` pris dans un cycle de `$ref` (anti-boucle-infinie / crash).

    Construit le graphe nom -> noms référencés, puis DFS coloré : une arête vers un
    nœud GRIS (dans la pile courante) = cycle. Renvoie les noms impliqués.
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
    color: dict[str, int] = {n: 0 for n in graph}        # 0 blanc, 1 gris, 2 noir
    stack: list[str] = []
    in_cycle: set[str] = set()

    def dfs(n: str) -> None:
        color[n] = 1
        stack.append(n)
        for m in graph.get(n, ()):
            if m not in graph:                           # ref externe/inconnue -> ignorée
                continue
            if color[m] == 1:                            # arête arrière -> cycle
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
    """TIER 1 : toujours chargé. Noms -> phrase courte + $defs partagé gardé une fois."""
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
    index = build_index(tools, defs)          # $defs partagé gardé une fois ici
    idx_tok = count_tokens(_j(index))

    print(f"{len(tools)} tools MCP\n")
    print(f"{'':<34}{'tokens':>8}")
    print("-" * 44)
    print(f"{'A — tous les schémas bruts':<34}{raw_all:>8}   (chargé à CHAQUE requête aujourd'hui)")
    print(f"{'TIER 1 — index complet':<34}{idx_tok:>8}   (toujours chargé)")

    # compression par tool
    print(f"\n{'tool':<26}{'brut':>7}{'comprimé':>10}{'gain':>7}")
    print("-" * 50)
    tot_raw = tot_comp = 0
    for t in tools:
        rb = count_tokens(_j(t))
        cb = count_tokens(_j(compress_tool(t, defs)))
        tot_raw += rb
        tot_comp += cb
        print(f"{t['name']:<26}{rb:>7}{cb:>10}{(1-cb/rb)*100:>6.0f}%")
    print("-" * 50)
    print(f"{'TOTAL schémas':<26}{tot_raw:>7}{tot_comp:>10}{(1-tot_comp/tot_raw)*100:>6.0f}%")

    # scénario réaliste : une requête utilise k tools
    print("\n=== Scénario : une requête utilise k tools ===")
    print(f"{'k':>3}{'  aujourd hui (A)':>18}{'  deux étages (B)':>19}{'  réduction':>12}")
    by_name = {t["name"]: t for t in tools}
    used = ["github_create_issue", "fs_read_file", "slack_post_message"]
    for k in (1, 2, 3):
        sel = used[:k]
        b = idx_tok + sum(count_tokens(_j(compress_tool(by_name[n], defs))) for n in sel)
        print(f"{k:>3}{raw_all:>18}{b:>19}{(1-b/raw_all)*100:>11.0f}%")
