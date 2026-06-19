"""
main.py — API FastAPI du Token Optimizer (backend de la démo).

5 endpoints :

  GET  /                health check
  POST /compress        upload .py        -> métriques tokens (avant/après, %)
  POST /compress-text   JSON {code: str}  -> même chose, pour l'outil web Lovable
  POST /blast-radius    .py + fonction    -> qui casse si on change X
  POST /mcp-compress    catalogue MCP     -> index TIER 1 vs tout chargé (-90%)
  POST /mcp-query       requête libre     -> flux réel SANS vs AVEC optimisation

CORS ouvert (front Lovable appelle depuis une autre origine).
Lancement local : uvicorn main:app --reload
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager

# les modules du cœur vivent dans src/ — on les rend importables (point d'entrée Render).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ast_extractor import compress, count_tokens, extract
from mcp_optimizer import build_index, collect_defs, compress_tool
from cache_injector import CacheInjector

# ── Stats globales de session (accumulées par les endpoints MCP) ─────────────
_SESSION_STATS: dict = {
    "total_requests": 0,
    "tokens_economises": 0,
    "_tokens_raw": 0,      # non exposé, sert au calcul de reduction_moyenne_pct
    "par_serveur": {},      # clé = filename ou URL
}


def _record_stats(server_key: str, tokens_raw: int, tokens_compressed: int) -> None:
    saved = max(0, tokens_raw - tokens_compressed)
    _SESSION_STATS["total_requests"] += 1
    _SESSION_STATS["tokens_economises"] += saved
    _SESSION_STATS["_tokens_raw"] += tokens_raw
    entry = _SESSION_STATS["par_serveur"].setdefault(
        server_key, {"requests": 0, "tokens_economises": 0}
    )
    entry["requests"] += 1
    entry["tokens_economises"] += saved

# ─────────────────────────────────────────────────────────────────────────────
# Proxy MCP monté en HTTP/SSE — /proxy/sse + /proxy/messages
# ─────────────────────────────────────────────────────────────────────────────
from refract_proxy import RefractProxy

_PROXY_TARGET = os.environ.get(
    "REFRACT_PROXY_TARGET",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "schemas", "mcp_calendar_schemas.json"),
)
_refract_proxy = RefractProxy(target_url=_PROXY_TARGET, verbose=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await _refract_proxy.connect()
        print(f"[Refract] Proxy HTTP monté → /proxy/sse (cible : {_PROXY_TARGET})")
    except Exception as exc:
        print(f"[Refract] Proxy HTTP non connecté au démarrage : {exc}")
    yield


app = FastAPI(title="Token Optimizer API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/proxy", _refract_proxy.build_asgi_app())


def _pct(before: int, after: int) -> float:
    return round((1 - after / before) * 100, 1) if before else 0.0


def _j(o) -> str:
    return json.dumps(o, ensure_ascii=False)


def _load_catalog(raw: dict | list) -> list[dict]:
    if isinstance(raw, list):
        return raw
    if "tools" in raw and isinstance(raw["tools"], list):
        return raw["tools"]
    tools: list[dict] = []
    for v in raw.values():
        if isinstance(v, list):
            tools.extend(v)
    return tools


# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def health() -> dict:
    return {
        "status": "ok",
        "service": "token-optimizer",
        "endpoints": [
            "/compress", "/compress-text", "/blast-radius",
            "/mcp-compress", "/mcp-query", "/stats", "/cache-estimate",
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 0. Stats de session
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/stats")
def get_stats() -> dict:
    """Statistiques agrégées de la session — tokens économisés, coût évité, réduction moyenne."""
    raw = _SESSION_STATS["_tokens_raw"]
    saved = _SESSION_STATS["tokens_economises"]
    pct = round(saved / raw * 100, 1) if raw else 0.0
    return {
        "session_stats": {
            "total_requests": _SESSION_STATS["total_requests"],
            "tokens_economises": saved,
            "cout_evite_usd": round(saved / 1_000_000 * 3.0, 6),
            "reduction_moyenne_pct": pct,
            "par_serveur": _SESSION_STATS["par_serveur"],
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# 0b. Estimation des économies Refract + prompt cache Anthropic
# ─────────────────────────────────────────────────────────────────────────────
class CacheEstimateInput(BaseModel):
    tokens: int
    requests_per_day: int
    days: int = 30


@app.post("/cache-estimate")
def cache_estimate(data: CacheEstimateInput) -> dict:
    """Estime les économies de la combinaison Refract + prompt cache Anthropic.

    Compare deux scénarios sur ``days`` jours :
    - **Sans cache / sans Refract** : tarif input standard 3,00 $/M, tous les tokens,
      toutes les requêtes.
    - **Avec cache / avec Refract** : cache write 3,75 $/M au premier appel,
      puis cache read 0,30 $/M (×10 moins cher) pour les hits suivants.

    ``tokens`` doit être le nombre de tokens *après* compression par Refract.
    Utilisez ``/mcp-compress`` pour obtenir ce compte.
    """
    if data.tokens <= 0:
        raise HTTPException(status_code=400, detail="tokens doit être > 0")
    if data.requests_per_day <= 0:
        raise HTTPException(status_code=400, detail="requests_per_day doit être > 0")
    return CacheInjector.estimate_cache_savings(
        tokens=data.tokens,
        requests_per_day=data.requests_per_day,
        days=data.days,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1a. Compression via fichier uploadé
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/compress")
async def compress_code(file: UploadFile = File(...)) -> dict:
    source = (await file.read()).decode("utf-8", errors="replace")
    try:
        compressed = compress(source, tags=True)
    except SyntaxError as e:
        raise HTTPException(status_code=400, detail=f"Python invalide : {e}")

    before = count_tokens(source)
    after = count_tokens(compressed)
    return {
        "filename": file.filename,
        "tokens_avant": before,
        "tokens_apres": after,
        "reduction_pct": _pct(before, after),
        "cout_avant_usd": round((before / 1_000_000) * 3.0, 6),
        "cout_apres_usd": round((after / 1_000_000) * 3.0, 6),
        "structure": compressed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1b. Compression via JSON body (Lovable)
# ─────────────────────────────────────────────────────────────────────────────
class CodeInput(BaseModel):
    code: str


@app.post("/compress-text")
def compress_text(data: CodeInput) -> dict:
    source = data.code
    if not source.strip():
        raise HTTPException(status_code=400, detail="Code vide")

    try:
        compressed = compress(source, tags=True)
        parsable = True
    except SyntaxError:
        compressed = ""
        parsable = False

    # Structure JSON pour l'arbre sémantique Lovable
    import ast as ast_module
    structure_json = {}
    if parsable:
        try:
            tree = ast_module.parse(source)
            imports = []
            for n in ast_module.walk(tree):
                if isinstance(n, ast_module.Import):
                    imports += [a.name for a in n.names]
                elif isinstance(n, ast_module.ImportFrom):
                    mod = n.module or ""
                    imports += [f"{mod}.{a.name}" for a in n.names]
            dans_classe = set()
            for c in ast_module.walk(tree):
                if isinstance(c, ast_module.ClassDef):
                    for it in ast_module.walk(c):
                        if isinstance(it, ast_module.FunctionDef):
                            dans_classe.add(id(it))
            def get_appels(node):
                out = []
                for n in ast_module.walk(node):
                    if isinstance(n, ast_module.Call):
                        if isinstance(n.func, ast_module.Attribute):
                            out.append(f"{n.func.value.id if isinstance(n.func.value, ast_module.Name) else '?'}.{n.func.attr}")
                        elif isinstance(n.func, ast_module.Name):
                            out.append(n.func.id)
                return list(dict.fromkeys(out))
            def get_raises(node):
                return [n.exc.func.id for n in ast_module.walk(node)
                        if isinstance(n, ast_module.Raise) and n.exc
                        and isinstance(n.exc, ast_module.Call)
                        and isinstance(n.exc.func, ast_module.Name)]
            classes = []
            for n in ast_module.walk(tree):
                if isinstance(n, ast_module.ClassDef):
                    methodes = [
                        {"nom": it.name, "params": [a.arg for a in it.args.args],
                         "appels": get_appels(it), "raises": get_raises(it)}
                        for it in ast_module.walk(n) if isinstance(it, ast_module.FunctionDef)
                    ]
                    classes.append({"nom": n.name, "methodes": methodes})
            fonctions = [
                {"nom": n.name, "params": [a.arg for a in n.args.args],
                 "appels": get_appels(n), "raises": get_raises(n)}
                for n in ast_module.walk(tree)
                if isinstance(n, ast_module.FunctionDef) and id(n) not in dans_classe
            ]
            structure_json = {"imports": imports, "classes": classes, "fonctions": fonctions}
        except Exception:
            structure_json = {}

    before = count_tokens(source)
    after = count_tokens(compressed) if parsable else before

    return {
        "parsable": parsable,
        "tokens_avant": before,
        "tokens_apres": after,
        "reduction_pct": round((1 - after / before) * 100) if before and parsable else 0,
        "cout_avant_usd": round((before / 1_000_000) * 3.0, 6),
        "cout_apres_usd": round((after / 1_000_000) * 3.0, 6),
        "structure": compressed,
        "structure_json": structure_json,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Blast-radius
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/blast-radius")
async def blast_radius(
    file: UploadFile = File(...),
    target: str = Form(...),
) -> dict:
    source = (await file.read()).decode("utf-8", errors="replace")
    try:
        data = extract(source)
    except SyntaxError as e:
        raise HTTPException(status_code=400, detail=f"Python invalide : {e}")

    fonctions = {f["nom"]: set(f["appels"]) for f in data["fonctions"]}
    if target not in fonctions:
        raise HTTPException(
            status_code=404,
            detail=f"Fonction '{target}' introuvable. Disponibles : {sorted(fonctions)}",
        )

    direct = sorted(n for n, calls in fonctions.items() if target in calls)

    impacted: set[str] = set()
    frontier = {target}
    while frontier:
        nxt = {
            n for n, calls in fonctions.items()
            if calls & frontier and n not in impacted and n != target
        }
        impacted |= nxt
        frontier = nxt

    return {
        "filename": file.filename,
        "target": target,
        "direct_callers": direct,
        "blast_radius": sorted(impacted),
        "impacted_count": len(impacted),
        "total_functions": len(fonctions),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Compression schémas MCP
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/mcp-compress")
async def mcp_compress(file: UploadFile = File(...)) -> dict:
    try:
        raw = json.loads((await file.read()).decode("utf-8", errors="replace"))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"JSON invalide : {e}")

    tools = _load_catalog(raw)
    if not tools:
        raise HTTPException(status_code=400, detail="Aucun tool trouvé dans le fichier.")

    def facing(t: dict) -> dict:
        return {
            "name": t.get("name", ""),
            "description": t.get("description", ""),
            "inputSchema": t.get("inputSchema", {}),
        }

    raw_all = count_tokens(_j([facing(t) for t in tools]))
    defs = collect_defs(tools)
    index = build_index(tools, defs)
    idx_tok = count_tokens(_j(index))

    comp = sorted(
        (count_tokens(_j(compress_tool(t))) for t in tools), reverse=True
    )
    scenario = [
        {
            "k": k,
            "tokens": idx_tok + sum(comp[:k]),
            "reduction_pct": _pct(raw_all, idx_tok + sum(comp[:k])),
        }
        for k in (1, 3, 5)
        if k <= len(comp)
    ]

    _record_stats(file.filename or "upload", raw_all, idx_tok)
    return {
        "filename": file.filename,
        "n_tools": len(tools),
        "tokens_all_loaded": raw_all,
        "tokens_index": idx_tok,
        "reduction_pct": _pct(raw_all, idx_tok),
        "shared_defs": sorted(defs),
        "per_request": scenario,
        "tools_names": [t.get("name", "") for t in tools],
        "tools_compressed": {
            t.get("name", ""): count_tokens(_j(compress_tool(t)))
            for t in tools
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Flux réel SANS vs AVEC optimisation (requête libre)
# ─────────────────────────────────────────────────────────────────────────────
class QueryInput(BaseModel):
    query: str
    catalog: list


def _identifier_tool(query: str, tools: list) -> str:
    """Identifie le tool le plus pertinent par score de mots-clés. Zéro appel LLM."""
    query_words = set(query.lower().split())
    scores = {}
    for t in tools:
        name = t.get("name", "").lower().replace("_", " ")
        desc = t.get("description", "").lower()
        score = sum(1 for w in query_words if w in name or w in desc)
        scores[t.get("name", "")] = score
    best = max(scores, key=scores.get) if scores else ""
    return best


@app.post("/mcp-query")
def mcp_query(data: QueryInput) -> dict:
    tools = data.catalog
    if not tools:
        raise HTTPException(status_code=400, detail="Catalogue vide")

    tokens_query = count_tokens(data.query)

    # SANS optimisation — tous les schémas dans le prompt
    all_schemas = [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t.get("inputSchema", {}),
        }
        for t in tools
    ]
    tokens_avant = count_tokens(_j(all_schemas))

    # AVEC optimisation — index compact + 1 schéma à la demande
    defs = collect_defs(tools)
    index = build_index(tools, defs)
    tokens_index = count_tokens(_j(index))

    # Identification par mots-clés (zéro LLM, zéro coût)
    tool_name = _identifier_tool(data.query, tools)
    tool = next((t for t in tools if t.get("name") == tool_name), None)
    compressed = compress_tool(tool) if tool else {}
    tokens_tool = count_tokens(_j(compressed))
    tokens_apres = tokens_index + tokens_tool

    _record_stats("mcp-query", tokens_avant, tokens_apres)
    return {
        "query": data.query,
        "tool_identifie": tool_name,
        "flux_sans": {
            "etapes": [
                {"label": "Requête agent", "tokens": tokens_query},
                {"label": f"{len(tools)} schémas chargés d'un coup", "tokens": tokens_avant},
                {"label": "TOTAL envoyé au LLM", "tokens": tokens_query + tokens_avant},
            ],
            "tokens_total": tokens_query + tokens_avant,
            "cout_usd": round(((tokens_query + tokens_avant) / 1_000_000) * 3, 6),
            "tools_charges": len(tools),
        },
        "flux_avec": {
            "etapes": [
                {"label": "Requête agent", "tokens": tokens_query},
                {"label": "Index compact chargé", "tokens": tokens_index},
                {"label": f"Tool identifié : {tool_name}", "tokens": 0},
                {"label": "Schéma compressé chargé", "tokens": tokens_tool},
                {"label": "TOTAL envoyé au LLM", "tokens": tokens_query + tokens_apres},
            ],
            "tokens_index": tokens_index,
            "tokens_tool": tokens_tool,
            "tokens_total": tokens_query + tokens_apres,
            "cout_usd": round(((tokens_query + tokens_apres) / 1_000_000) * 3, 6),
        },
        "reduction_pct": round((1 - tokens_apres / tokens_avant) * 100) if tokens_avant else 0,
        "economie_usd": round(((tokens_avant - tokens_apres) / 1_000_000) * 3, 6),
    }

# ─────────────────────────────────────────────────────────────────────────────
# 5. Catalogue MCP hébergé (évite les problèmes de repo privé)
# ─────────────────────────────────────────────────────────────────────────────
import pathlib

@app.get("/catalog")
def get_catalog() -> list:
    path = pathlib.Path(__file__).parent / "schemas" / "mcp_calendar_schemas.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="mcp_calendar_schemas.json introuvable")
    raw = json.loads(path.read_text())
    return _load_catalog(raw)
