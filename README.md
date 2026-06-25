<p align="center">
  <img src="assets/logo.png" alt="Refract" width="360">
</p>

# Refract

> Cuts up to 98% of the tokens your AI agents spend using MCP tools — without losing anything.

---

## What it actually changes

| Server | Tools | Before | After | Reduction |
|---|---|---|---|---|
| filesystem (Anthropic) | 14 | 1,892 tok | 236 tok | **−88%** |
| sequential-thinking | 1 | 926 tok | 20 tok | **−98%** |
| Google Calendar | 5 | 5,173 tok | 155 tok | **−97%** |
| Enterprise (Cal + Gmail + Drive) | 12 | 8,649 tok | 882 tok | **−90%** |
| sample_app.js (JavaScript) | — | 799 tok | 284 tok | **−64.5%** |
| sample_app.ts (TypeScript) | — | 378 tok | 266 tok | **−29.6%** |
| ast_extractor.py (Python) | — | 3,600 tok | 868 tok | **−75.9%** |

Fewer tokens sent = lower API bills, faster responses. And nothing is lost. Every check confirmed tools stay 100% usable after compression.

---

## Install

```bash
pip install refract-mcp
```

Optional extras:

```bash
pip install refract-mcp[semantic]   # semantic tool routing with embeddings
pip install refract-mcp[multilang]  # JavaScript, TypeScript, JSX, TSX support
```

---

## Two modes

### Mode 1 — MCP Proxy

Sits between your agent and any MCP server. Compresses tool schemas on the fly so your agent does not load the full catalogue on every request.

**Local subprocess (stdio):**

```bash
refract-proxy --target "npx @modelcontextprotocol/server-filesystem /tmp" --verbose
```

**Remote HTTP/SSE server:**

```bash
# --url implies SSE transport (explicit, recommended for remote endpoints)
refract-proxy --url https://my-mcp-server.com/sse

# or with --transport flag (auto-detection can be overridden)
refract-proxy --target https://my-mcp-server.com/sse --transport sse
```

**Proxy flags:**

| Flag | Default | Description |
|---|---|---|
| `--target URL` | required | MCP target: stdio command, HTTP URL, or JSON file |
| `--stdio-cmd CMD` | — | Alias for `--target` for stdio commands |
| `--url URL` | — | Remote SSE/HTTP endpoint — implies `--transport sse` |
| `--transport {stdio,sse}` | auto | Force transport to the target (overrides auto-detection) |
| `--sse-timeout SECONDS` | 30 | Connection timeout for SSE targets (retries 3×) |
| `--mode {stdio,http}` | stdio | How the proxy serves your agent |
| `--port PORT` | 8080 | Proxy listen port in `--mode http` |
| `--verbose` | off | Print token counts per request |
| `--log-level` | WARNING | DEBUG / INFO / WARNING / ERROR |

Add it to Claude Desktop:

```json
{
  "mcpServers": {
    "my-server-via-refract": {
      "command": "/path/to/refract-proxy",
      "args": [
        "--target",
        "npx @modelcontextprotocol/server-filesystem /path/to/folder",
        "--verbose"
      ]
    }
  }
}
```

For a remote MCP server:

```json
{
  "mcpServers": {
    "remote-via-refract": {
      "command": "/path/to/refract-proxy",
      "args": ["--url", "https://my-mcp-server.com/sse"]
    }
  }
}
```

### Mode 2 — MCP Server

Exposes your codebase as an MCP server. Your agent can index a repo, get compressed file context, expand specific functions, analyze impact, detect breaking changes, and map security risks.

```bash
refract-server --root /path/to/your/repo
```

Add it to Claude Desktop:

```json
{
  "mcpServers": {
    "refract-code": {
      "command": "/path/to/refract-server",
      "args": ["--root", "/path/to/your/repo"]
    }
  }
}
```

---

## How it works, no jargon

Imagine a library with 50 books.

Without Refract: your agent gets a detailed summary of all 50 books on every question, even if the answer only needs one of them.

With Refract: your agent first gets a list of titles (the index). Once it knows which book it needs, it only receives that book's content.

Technically:

The index (always sent): just tool names and a short description of each.

The detail (sent only when needed): the full description of the tool actually used, everything required to use it correctly, nothing more.

The verification: after every compression, Refract automatically checks that nothing important was removed. If there is any doubt, it sends the full version instead of taking a risk.

No AI model is involved in this process. It is fully automatic, fast, and deterministic.

---

## MCP Proxy tools

| Tool | What it does |
|---|---|
| Compression | Compresses tool schemas on the fly, up to 98% reduction |
| Signal check | Verifies callable contract after every compression |
| Semantic routing | Identifies the right tool using embeddings (opt-in) |
| Prompt caching | Injects Anthropic cache_control for repeated requests |

## MCP Server tools

| Tool | Input | Output |
|---|---|---|
| index_repo | repo path | aggregated index of all Python, JS, TS files |
| get_compressed | file path | compressed structure + token stats |
| expand | file path + function names | verbatim source + dependency context |
| blast_radius | file path + function name | all functions that break if target changes |
| semantic_diff | file path + old source + new source | breaking changes vs body-only changes |
| security_surface | repo path | map of dangerous calls (subprocess, eval, pickle, requests) |

---

## blast_radius

Ask Claude which functions break if you change a target function.

Example result:

```json
{
  "target": "authenticate",
  "direct_callers": ["login_user"],
  "all_impacted": ["login_user", "verify_session", "admin_access"],
  "impacted_count": 3,
  "risk_level": "MEDIUM"
}
```

Risk levels: LOW (0 to 2 impacted), MEDIUM (3 to 5), HIGH (6 or more).

---

## semantic_diff

Detects breaking API changes by comparing function interfaces, not bodies. Use it as a CI gate.

Example result:

```json
{
  "breaking": ["authenticate"],
  "body_only": ["logout"],
  "added": ["new_function"],
  "removed": [],
  "unchanged": ["hash_password"],
  "is_breaking": true
}
```

If is_breaking is true, the PR changes the public API and must be reviewed.

---

## security_surface

Maps every function that calls dangerous primitives across your repo.

HIGH risk: subprocess, os.system, eval, exec, pickle, ctypes

MEDIUM risk: open (write mode), socket, requests, httpx, urllib

Example result:

```json
{
  "high_risk": [
    {
      "file": "src/llm_client.py",
      "function": "run_command",
      "calls": ["subprocess.run"]
    }
  ],
  "summary": {
    "high_risk_count": 1,
    "medium_risk_count": 3,
    "total_functions_scanned": 87,
    "clean_files": 8
  }
}
```

---

## Languages supported

Python (via ast module), JavaScript, TypeScript, JSX, TSX (via tree-sitter, opt-in with pip install refract-mcp[multilang]).

Language is auto-detected from file extension. Graceful fallback if tree-sitter is not installed.

---

## Built-in Anthropic caching

Refract integrates with Anthropic prompt caching. as_anthropic_tools() automatically marks the compressed catalogue as cacheable, cutting costs further on repeated requests.

Example over 30 days, 100 requests per day, 5,000 tokens of schemas:

| Scenario | Cost |
|---|---|
| Without Refract, without cache | $45.00 |
| With Refract + cache | $1.49 |

---

## Troubleshooting

**"Failed to spawn process: No such file or directory" in Claude Desktop**

Claude Desktop cannot find refract-proxy in its PATH. Find the absolute path and use it directly:

```bash
which refract-proxy
```

Then use the full path in claude_desktop_config.json:

```json
{
  "mcpServers": {
    "my-tool-via-refract": {
      "command": "/full/path/to/refract-proxy",
      "args": [
        "--target",
        "npx @modelcontextprotocol/server-filesystem /path/to/folder"
      ]
    }
  }
}
```

---

## Works with

Claude Desktop, Cursor, any client that follows the MCP standard, any existing MCP server.

---

## License

MIT — free to use, including commercially.
