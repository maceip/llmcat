# llmcat

Fast, pragmatic summaries for code/text. Think `cat`, but with LLM summaries. Supports OpenAI, Anthropic, Google, and local Ollama. Also runs as an MCP stdio server.

## Quickstart

```bash
# Build
go build -o llmcat ./cmd/llmcat

# Set any provider keys you have (or run with -x to auto-discover)
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...

# Summarize a single file
./llmcat README.md

# Pipe from stdin (use '-' explicitly)
echo "3.14159265358979323846" | ./llmcat -

# See status and provider order
./llmcat status

# Start MCP stdio server (no args)
./llmcat

# Manage external MCP servers
./llmcat mcp add myserver --env KEY=value -- python server.py --port 8080
./llmcat mcp list
./llmcat mcp ping myserver
./llmcat mcp call myserver summarize --json '{"text":"hello"}'
```

## Usage

- Popular examples:
  - `llmcat **/*.{go,ts,py}`
  - `git diff -U0 | llmcat --one-line -`
  - `llmcat --provider auto --preset concise README.md`
- MCP management:
  - `llmcat mcp add <name> -- <command> [args...]`
  - `llmcat mcp add --transport sse <name> <url>`
  - `llmcat mcp add --transport http <name> <url>`
  - `llmcat mcp list|get|remove|ping|call`
- Config: optional JSON at `~/.config/llmcat/config.json` (or set `LLMCAT_CONFIG_DIR` to override). CLI flags override config values.

## Notes

- Keys are cached in `~/.config/llmcat/keys.json` with `0600` permissions.
- To summarize stdin now pass `-` explicitly, because no-args starts the MCP server.
- Ollama is auto-detected and preferred in `--provider auto` when available.
