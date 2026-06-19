"""
mcp_signal_check — l'équivalent MCP de signal_check.py : un chiffre de compression
ne vaut rien si on jette le signal dont le modèle a besoin pour APPELER l'outil.

On compare, pour chaque tool, le « contrat appelable » du schéma brut vs comprimé :
  * tous les paramètres présents (un param perdu = appel cassé) ;
  * le drapeau `required` conservé ;
  * les `enum` conservés (contrainte de valeurs) ;
  * les liens `$ref` conservés ET résolvables depuis le $defs de l'index
    (top-level OU dans les items d'un tableau) — sinon le modèle ne connaît
    pas la forme de l'objet à fournir.

Sortie : couverture par tool + total. FLAG si quoi que ce soit est perdu.
"""

from __future__ import annotations

import json
import os
import sys

from mcp_optimizer import (build_index, collect_defs, compress_tool, ref_cycles,
                           unhandled_constructs)

_SCHEMAS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schemas")
CATALOG = os.path.join(_SCHEMAS, "mcp_enterprise_schemas.json")


def _branches(p: dict) -> list:
    return [b for b in (p.get("anyOf") or p.get("oneOf") or []) if isinstance(b, dict)]


# --- atomes de contrat extraits du schéma BRUT ------------------------------- #
def raw_refs(p: dict) -> set[str]:
    refs: set[str] = set()
    if not isinstance(p, dict):
        return refs
    if "$ref" in p:
        refs.add(p["$ref"].split("/")[-1])
    if isinstance(p.get("items"), dict):
        refs |= raw_refs(p["items"])
    for b in _branches(p):
        refs |= raw_refs(b)
    for b in p.get("allOf") or []:               # sous-schémas allOf
        refs |= raw_refs(b)
    if isinstance(p.get("properties"), dict):
        for v in p["properties"].values():
            refs |= raw_refs(v)
    return refs


def raw_enums(p: dict) -> set[str]:
    enums: set[str] = set()
    if not isinstance(p, dict):
        return enums
    if "enum" in p:
        enums |= {str(x) for x in p["enum"]}
    if isinstance(p.get("items"), dict):
        enums |= raw_enums(p["items"])
    for b in _branches(p):
        enums |= raw_enums(b)
    for b in p.get("allOf") or []:               # sous-schémas allOf
        enums |= raw_enums(b)
    if isinstance(p.get("properties"), dict):
        for v in p["properties"].values():
            enums |= raw_enums(v)
    return enums


def raw_patterns(p: dict) -> set[str]:
    pats: set[str] = set()
    if not isinstance(p, dict):
        return pats
    if "pattern" in p:
        pats.add(p["pattern"])
    if isinstance(p.get("items"), dict):
        pats |= raw_patterns(p["items"])
    for b in _branches(p):
        pats |= raw_patterns(b)
    for b in p.get("allOf") or []:
        pats |= raw_patterns(b)
    if isinstance(p.get("properties"), dict):
        for v in p["properties"].values():
            pats |= raw_patterns(v)
    return pats


def raw_bounds(p: dict) -> set:
    """Bornes numériques (minimum/maximum) du schéma brut, en tuples (genre, valeur)."""
    b: set = set()
    if not isinstance(p, dict):
        return b
    if "minimum" in p:
        b.add(("min", p["minimum"]))
    if "maximum" in p:
        b.add(("max", p["maximum"]))
    if isinstance(p.get("items"), dict):
        b |= raw_bounds(p["items"])
    for br in _branches(p):
        b |= raw_bounds(br)
    for br in p.get("allOf") or []:
        b |= raw_bounds(br)
    if isinstance(p.get("properties"), dict):
        for v in p["properties"].values():
            b |= raw_bounds(v)
    return b


def raw_disc(p: dict) -> set[str]:
    """propertyName des oneOf discriminés présents dans le schéma brut."""
    discs: set[str] = set()
    if not isinstance(p, dict):
        return discs
    d = p.get("discriminator")
    if isinstance(d, dict) and d.get("propertyName"):
        discs.add(d["propertyName"])
    if isinstance(p.get("items"), dict):
        discs |= raw_disc(p["items"])
    for b in _branches(p):
        discs |= raw_disc(b)
    for b in p.get("allOf") or []:
        discs |= raw_disc(b)
    if isinstance(p.get("properties"), dict):
        for v in p["properties"].values():
            discs |= raw_disc(v)
    return discs


def raw_typed(p: dict) -> bool:
    """Le param brut porte-t-il un type exploitable (type/$ref/union/enum/allOf) ?"""
    return bool(isinstance(p, dict) and
                ("type" in p or "$ref" in p or "enum" in p or _branches(p)
                 or p.get("allOf")))


def comp_typed(cp: dict) -> bool:
    """La forme comprimée porte-t-elle encore une info de type ?"""
    return bool(isinstance(cp, dict) and
                any(k in cp for k in ("t", "ref", "any", "all", "of", "props", "enum")))


# --- mêmes atomes extraits de la forme COMPRIMÉE ----------------------------- #
def comp_refs(cp: dict) -> set[str]:
    refs: set[str] = set()
    if not isinstance(cp, dict):
        return refs
    if "ref" in cp:
        refs.add(cp["ref"])
    if isinstance(cp.get("of"), dict):          # items de tableau (param comprimé)
        refs |= comp_refs(cp["of"])
    for b in cp.get("any", []):                 # branches d'union
        refs |= comp_refs(b)
    for b in cp.get("all", []):                 # sous-schémas allOf
        refs |= comp_refs(b)
    if isinstance(cp.get("props"), dict):
        for v in cp["props"].values():
            refs |= comp_refs(v)
    return refs


def comp_enums(cp: dict) -> set[str]:
    enums: set[str] = set()
    if not isinstance(cp, dict):
        return enums
    if "enum" in cp:
        enums |= {str(x) for x in cp["enum"]}
    if isinstance(cp.get("of"), dict):          # items de tableau (param comprimé)
        enums |= comp_enums(cp["of"])
    for b in cp.get("any", []):                 # branches d'union
        enums |= comp_enums(b)
    for b in cp.get("all", []):                 # sous-schémas allOf
        enums |= comp_enums(b)
    if isinstance(cp.get("props"), dict):
        for v in cp["props"].values():
            enums |= comp_enums(v)
    return enums


def comp_patterns(cp: dict) -> set[str]:
    pats: set[str] = set()
    if not isinstance(cp, dict):
        return pats
    if "pat" in cp:
        pats.add(cp["pat"])
    if isinstance(cp.get("of"), dict):
        pats |= comp_patterns(cp["of"])
    for b in cp.get("any", []):
        pats |= comp_patterns(b)
    for b in cp.get("all", []):
        pats |= comp_patterns(b)
    if isinstance(cp.get("props"), dict):
        for v in cp["props"].values():
            pats |= comp_patterns(v)
    return pats


def comp_bounds(cp: dict) -> set:
    b: set = set()
    if not isinstance(cp, dict):
        return b
    if "min" in cp:
        b.add(("min", cp["min"]))
    if "max" in cp:
        b.add(("max", cp["max"]))
    if isinstance(cp.get("of"), dict):
        b |= comp_bounds(cp["of"])
    for br in cp.get("any", []):
        b |= comp_bounds(br)
    for br in cp.get("all", []):
        b |= comp_bounds(br)
    if isinstance(cp.get("props"), dict):
        for v in cp["props"].values():
            b |= comp_bounds(v)
    return b


def comp_disc(cp: dict) -> set[str]:
    discs: set[str] = set()
    if not isinstance(cp, dict):
        return discs
    d = cp.get("disc")
    if isinstance(d, dict) and d.get("prop"):
        discs.add(d["prop"])
    if isinstance(cp.get("of"), dict):
        discs |= comp_disc(cp["of"])
    for b in cp.get("any", []):
        discs |= comp_disc(b)
    for b in cp.get("all", []):
        discs |= comp_disc(b)
    if isinstance(cp.get("props"), dict):
        for v in cp["props"].values():
            discs |= comp_disc(v)
    return discs


def main(argv) -> int:
    catalog = argv[1] if len(argv) > 1 else CATALOG
    data = json.load(open(catalog, encoding="utf-8"))
    tools = [t for lst in data.values() for t in lst]
    defs = collect_defs(tools)
    index_defs = set(build_index(tools, defs).get("$defs", {}))

    # garde-fou global : cycles de $ref (risque de boucle infinie à la reconstruction)
    cyclic = ref_cycles(defs)

    print(f"Vérif. du contrat appelable — {catalog} — {len(tools)} tools réels, "
          f"$defs index = {sorted(index_defs)}\n")
    if cyclic:
        print(f"⚠ FLAG cycles $ref (récursifs, non gérés) : {sorted(cyclic)}\n")
    print(f"{'tool':<22}{'params':>8}{'type':>7}{'required':>10}"
          f"{'enums':>8}{'$ref':>7}{'pat':>6}  verdict")
    print("-" * 76)

    ok_all = True
    for t in tools:
        sch = t.get("inputSchema", {}) or {}
        props = sch.get("properties", {}) or {}
        req = set(sch.get("required", []))

        ct = compress_tool(t)
        cparams = ct["params"]

        # 1. paramètres présents
        missing = set(props) - set(cparams)
        # 2. type conservé (un param typé brut doit rester typé comprimé)
        typed = {n for n, p in props.items() if raw_typed(p)}
        type_kept = {n for n in typed if comp_typed(cparams.get(n, {}))}
        type_lost = typed - type_kept
        # 3. required conservé
        req_kept = {n for n in req if cparams.get(n, {}).get("req")}
        req_lost = req - req_kept
        # 4. enums conservés
        r_enum = {e for p in props.values() for e in raw_enums(p)}
        c_enum = {e for cp in cparams.values() for e in comp_enums(cp)}
        enum_lost = r_enum - c_enum
        # 5. $ref conservés ET résolvables dans l'index
        r_ref = {x for p in props.values() for x in raw_refs(p)}
        c_ref = {x for cp in cparams.values() for x in comp_refs(cp)}
        ref_lost = r_ref - c_ref
        ref_unresolved = c_ref - index_defs
        # 6. patterns (regex) conservés comme signal (ÉTAPE 2.2)
        r_pat = {x for p in props.values() for x in raw_patterns(p)}
        c_pat = {x for cp in cparams.values() for x in comp_patterns(cp)}
        pat_lost = r_pat - c_pat
        # 7. discriminants oneOf conservés (ÉTAPE 2.3)
        r_disc = {x for p in props.values() for x in raw_disc(p)}
        c_disc = {x for cp in cparams.values() for x in comp_disc(cp)}
        disc_lost = r_disc - c_disc
        # 8. bornes numériques minimum/maximum conservées (ÉTAPE 2.4)
        r_bnd = {x for p in props.values() for x in raw_bounds(p)}
        c_bnd = {x for cp in cparams.values() for x in comp_bounds(cp)}
        bound_lost = r_bnd - c_bnd
        # 9. garde-fou : constructs non gérés (jamais droppés en silence -> FLAG)
        unhandled = unhandled_constructs(sch)

        good = not (missing or type_lost or req_lost or enum_lost or ref_lost
                    or ref_unresolved or pat_lost or disc_lost or bound_lost
                    or unhandled)
        ok_all &= good
        verdict = "OK" if good else "FLAG"
        flags = []
        if missing:
            flags.append(f"params perdus:{sorted(missing)}")
        if type_lost:
            flags.append(f"type perdu:{sorted(type_lost)}")
        if req_lost:
            flags.append(f"required perdu:{sorted(req_lost)}")
        if enum_lost:
            flags.append(f"enums perdus:{len(enum_lost)}")
        if ref_lost:
            flags.append(f"$ref perdus:{sorted(ref_lost)}")
        if ref_unresolved:
            flags.append(f"$ref non résolus:{sorted(ref_unresolved)}")
        if pat_lost:
            flags.append(f"patterns perdus:{len(pat_lost)}")
        if disc_lost:
            flags.append(f"discriminants perdus:{sorted(disc_lost)}")
        if bound_lost:
            flags.append(f"bornes perdues:{len(bound_lost)}")
        if unhandled:
            flags.append(f"non géré:{sorted(unhandled)}")

        print(f"{t['name']:<22}{len(props):>8}"
              f"{f'{len(type_kept)}/{len(typed)}':>7}"
              f"{f'{len(req_kept)}/{len(req)}':>10}"
              f"{f'{len(c_enum & r_enum)}/{len(r_enum)}':>8}"
              f"{f'{len(c_ref & r_ref)}/{len(r_ref)}':>7}"
              f"{f'{len(c_pat & r_pat)}/{len(r_pat)}':>6}  {verdict}"
              + ("  " + "; ".join(flags) if flags else ""))

    ok_all &= not cyclic                              # un cycle $ref = FLAG global

    print("-" * 76)
    print("✓ contrat appelable 100% préservé sur tous les tools"
          if ok_all else
          "✗ perte de signal / construct non géré détecté (voir FLAG)")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
