<p align="center">
  <img src="assets/logo.png" alt="Refract" width="360">
</p>

# Refract

> MCP proxy that compresses tool schemas. Up to −97% tokens, 100% signal preserved.

Every request your agent makes to an MCP server sends the **full schema catalogue** — even if only one tool is used. Refract sits between the agent and the server, compresses the schemas on the fly, and relays tool calls unchanged.

Zero LLM calls. Fully deterministic. Drop-in compatible with Claude Desktop, Cursor, and any MCP client.

---

## Install

```bash
pip install refract
```

---

## Usage

```bash
# Wrap any stdio MCP server
refract-proxy --target "npx @modelcontextprotocol/server-filesystem /tmp" --verbose

# Wrap an HTTP/SSE MCP server
refract-proxy --target "https://my-mcp-server.com" --mode http --port 8080

# Point at a local JSON schema file (for testing)
refract-proxy --target schemas/mcp_calendar_schemas.json --verbose
```

`--verbose` prints the token savings on every `tools/list` call:

```
[Refract] Connected to npx @modelcontextprotocol/server-filesystem /tmp
  14 tools  |  1892 → 236 tokens  (88% reduction index)
```

---

## Configure with Claude Desktop

Add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "filesystem-refract": {
      "command": "refract-proxy",
      "args": [
        "--target",
        "npx @modelcontextprotocol/server-filesystem /home/user/docs",
        "--verbose"
      ]
    }
  }
}
```

Replace `--target` with any stdio command, SSE URL, or JSON schema file.

---

## Benchmarks (real, measured)

| Server | Tools | Before | After (index) | Reduction |
|---|---|---|---|---|
| `@mcp/server-filesystem` | 14 | 1 892 tok | 236 tok | **−88%** |
| `@mcp/server-sequential-thinking` | 1 | 926 tok | 20 tok | **−98%** |
| Google Calendar | 5 | 5 010 tok | 660 tok | **−87%** |
| Enterprise (Cal + Gmail + Drive) | 12 | 8 649 tok | 882 tok | **−90%** |

Signal check passes at 100% on all servers: every parameter, `required` flag, `enum`, and `$ref` is preserved after compression.

---

## How it works

- **TIER 1 — Index** (always loaded): tool names + one-sentence descriptions + shared `$defs` deduplicated once. This replaces the full catalogue on every request.
- **TIER 2 — Compressed schema** (on demand): when a tool is called, its full compressed schema is loaded. Types, required params, enums, and `$ref` pointers are kept; verbose boilerplate is stripped.
- **Signal check**: after every compression, `mcp_signal_check` verifies the callable contract is intact. If any parameter is lost, the proxy falls back to the raw schema and logs a warning — the call never breaks.

---

## Python API

```python
from refract_proxy import RefractProxy

proxy = RefractProxy(
    target_url="npx @modelcontextprotocol/server-filesystem /tmp",
    verbose=True,
)
await proxy.connect()

# Use compressed tools directly with the Anthropic API
tools = proxy.as_anthropic_tools(use_cache=True)  # adds cache_control on last tool

# Or serve as a local MCP server (stdio)
await proxy.serve()

# Or expose via HTTP/SSE
await proxy.serve_http()  # → http://localhost:8080/sse
```

---

## Anthropic prompt caching

Refract integrates with [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching): `as_anthropic_tools()` injects `cache_control: {type: ephemeral}` on the last tool, so the compressed catalogue is cached across requests.

Combined savings (30 days, 100 requests/day, 5 000 tokens):

| | Cost |
|---|---|
| Without Refract, without cache | $45.00 |
| With Refract + cache | $1.49 |

---

## License

MIT