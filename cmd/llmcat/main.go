package main

import (
    "bufio"
    "bytes"
    "context"
    "encoding/json"
    "errors"
    "flag"
    "fmt"
    "io"
    "io/fs"
    "net/http"
    "os"
    "os/exec"
    "path/filepath"
    "runtime"
    "sort"
    "strings"
    "sync"
    "time"
    "unicode/utf8"
)

var httpClient = &http.Client{Timeout: 0}

// Provider identifies an LLM provider
// Supported: openai, anthropic, google, ollama

type Provider string

const (
	ProviderOpenAI   Provider = "openai"
	ProviderAnthropic Provider = "anthropic"
	ProviderGoogle    Provider = "google"
    ProviderOllama    Provider = "ollama"
    ProviderAuto      Provider = "auto"
)

// Config holds runtime configuration

type Config struct {
	Provider    Provider
	Model       string
    Format      string
	MaxChars    int
	HeadLines   int
	TailLines   int
    SymbolLines int
	Concurrency int
	NoColor     bool
    DiscoverKeys bool
    TimeoutSec  int
    MaxTokens   int
    NoPrefix    bool
    Preset      string
    OneLine     bool
    Diagram     bool
    // discovery options
    XClear      bool
}

func defaultConfig() Config {
	con := runtime.NumCPU()
	if con > 4 {
		con = 4
	}
	return Config{
		Provider:    ProviderOpenAI,
		Model:       "",
        // Default to text output so users never see raw JSON unless requested
        Format:      "text",
		MaxChars:    150_000,
        HeadLines:   80,
        TailLines:   80,
        SymbolLines: 300,
		Concurrency: con,
		NoColor:     false,
        TimeoutSec:  120,
        MaxTokens:   1024,
        NoPrefix:    false,
        Preset:      "",
        OneLine:     false,
        Diagram:     false,
        XClear:      false,
	}
}

func main() {
    cfg := defaultConfig()
    // Install custom usage printer with popular examples at the top
    flag.Usage = func() {
        printUsage()
    }
    // Pre-scan for --config/-c to load defaults before parsing flags
    configPath := ""
    for i := 1; i < len(os.Args); i++ {
        arg := os.Args[i]
        if strings.HasPrefix(arg, "--config=") {
            configPath = strings.TrimPrefix(arg, "--config=")
            break
        }
        if arg == "--config" || arg == "-c" {
            if i+1 < len(os.Args) {
                configPath = os.Args[i+1]
            }
            break
        }
    }
    flag.StringVar((*string)(&cfg.Provider), "provider", string(cfg.Provider), "provider: openai|anthropic|google|ollama|auto")
	flag.StringVar(&cfg.Model, "model", cfg.Model, "model id to use (optional)")
    flag.StringVar(&cfg.Format, "format", cfg.Format, "output format: text|json")
	flag.IntVar(&cfg.MaxChars, "max-chars", cfg.MaxChars, "max characters per file to send to LLM")
	flag.IntVar(&cfg.HeadLines, "head", cfg.HeadLines, "number of head lines for pre-summary context")
	flag.IntVar(&cfg.TailLines, "tail", cfg.TailLines, "number of tail lines for pre-summary context")
    flag.IntVar(&cfg.SymbolLines, "symbol-lines", cfg.SymbolLines, "max number of symbol lines to include in compaction")
	flag.IntVar(&cfg.Concurrency, "concurrency", cfg.Concurrency, "number of files to process concurrently")
	flag.BoolVar(&cfg.NoColor, "no-color", cfg.NoColor, "disable ANSI colors")
    flag.BoolVar(&cfg.DiscoverKeys, "x", false, "search environment and directories for API keys, verify and cache")
    flag.IntVar(&cfg.TimeoutSec, "timeout", cfg.TimeoutSec, "overall timeout per file in seconds")
    flag.IntVar(&cfg.MaxTokens, "max-tokens", cfg.MaxTokens, "max tokens to request from provider")
    flag.BoolVar(&cfg.NoPrefix, "no-prefix", cfg.NoPrefix, "disable per-file prefixes in output")
    flag.StringVar(&cfg.Preset, "preset", cfg.Preset, "provider/model preset: concise|detailed")
    flag.BoolVar(&cfg.OneLine, "one-line", cfg.OneLine, "print exactly one line per file (no previews/stream)")
    flag.BoolVar(&cfg.Diagram, "diagram", cfg.Diagram, "allow ASCII diagram output (off by default)")
    var configFlag string
    flag.StringVar(&configFlag, "config", configPath, "path to config file (json)")
    flag.StringVar(&configFlag, "c", configFlag, "path to config file (json) (shorthand)")
    // discovery tuning
    flag.BoolVar(&cfg.XClear, "x-clear", cfg.XClear, "clear cached keys and exit when used with -x")
    flag.Parse()

    // Always allow color by default; honor --no-color flag only.

    // Presets
    switch cfg.Preset {
    case "concise":
        if cfg.MaxTokens == 0 || cfg.MaxTokens > 256 { cfg.MaxTokens = 256 }
    case "detailed":
        if cfg.MaxTokens < 1024 { cfg.MaxTokens = 1024 }
    }

    // After parsing CLI flags, load config file and apply for flags not explicitly set
    // Determine final config path: CLI wins; else default (~/.config/llmcat/config.json)
    if configFlag == "" {
        if def, err := defaultConfigFilePath(); err == nil {
            if _, err := os.Stat(def); err == nil {
                configFlag = def
            }
        }
    }
    // build a set of flags explicitly set
    setFlags := map[string]bool{}
    flag.CommandLine.Visit(func(f *flag.Flag) { setFlags[f.Name] = true })
    if configFlag != "" {
        if overrides, ok := readConfigOverridesTyped(configFlag); ok {
            applyConfigOverridesTyped(&cfg, overrides, setFlags)
        }
    }
    // Force-enable color unless the user explicitly passed --no-color.
    // This ensures config files cannot disable color output.
    if !setFlags["no-color"] { cfg.NoColor = false }

    paths := flag.Args()

    // Handle subcommands: help | status | mcp
    if len(paths) > 0 {
        switch paths[0] {
        case "help", "--help", "-h":
            printUsage()
            return
        case "status":
            runStatus(cfg, configFlag)
            return
        case "mcp":
            if err := runMCP(paths[1:]); err != nil {
                fmt.Fprintln(os.Stderr, "mcp:", err)
                os.Exit(1)
            }
            return
        }
    }
    // If no positional args, start MCP stdio server. To summarize stdin, pass '-' explicitly.
    if len(paths) == 0 {
        runMCPServerStdio(cfg)
        return
    }

    if cfg.DiscoverKeys {
        if cfg.XClear {
            if err := clearKeyCache(); err != nil {
                fmt.Fprintln(os.Stderr, "[key discovery] ", err)
            } else {
                fmt.Fprintln(os.Stderr, "[key discovery] cache cleared")
            }
        }
        if err := runKeyDiscovery(context.Background(), cfg); err != nil {
            fmt.Fprintln(os.Stderr, "[key discovery] ", err)
        }
    }
    // Only auto-discover in provider=auto mode; otherwise avoid surprise network access
    if cfg.Provider == ProviderAuto {
        go func() {
            if cfg.DiscoverKeys { return }
            ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
            defer cancel()
            _ = runKeyDiscovery(ctx, cfg)
        }()
    }

    // Resolve globs and dedupe; handle stdin once
    seen := make(map[string]struct{})
    expanded := make([]string, 0, len(paths))
    add := func(p string) {
        if _, ok := seen[p]; ok { return }
        seen[p] = struct{}{}
        expanded = append(expanded, p)
    }
    if len(paths) == 0 {
        add("-")
    } else {
        hasStdin := false
        for _, p := range paths {
            if p == "-" { hasStdin = true; continue }
            matches, err := filepath.Glob(p)
            if err != nil || matches == nil {
                add(p)
                continue
            }
            for _, m := range matches { add(m) }
        }
        if hasStdin { add("-") }
        sort.Strings(expanded)
    }

	// Output coordinator: prevent interleaving across files by prefixing lines
	var wg sync.WaitGroup
	sem := make(chan struct{}, max(1, cfg.Concurrency))

	for _, p := range expanded {
		path := p
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
            if path == "-" {
                processStdin(cfg)
            } else {
                processFile(path, cfg)
            }
		}()
	}

	wg.Wait()
}

func processFile(path string, cfg Config) {
    info, err := os.Lstat(path)
	if err != nil {
		perr(path, cfg, fmt.Sprintf("error: %v", err))
		return
	}
	if info.IsDir() {
		perr(path, cfg, "is a directory; skipping")
		return
	}
    if (info.Mode() & os.ModeSocket) != 0 {
        perr(path, cfg, "is a socket; skipping")
        return
    }

	f, err := os.Open(path)
	if err != nil {
		perr(path, cfg, fmt.Sprintf("error opening: %v", err))
		return
	}
	defer f.Close()

    // Quick binary sniff
	isText, _, err := sniffText(f, 4096)
	if err != nil {
		perr(path, cfg, fmt.Sprintf("read error: %v", err))
		return
	}
	if !isText {
		perr(path, cfg, "binary or non-text; skipping")
		return
	}

    // Determine language from extension
	lang := languageFromPath(path)

    // Removed eager preview: no immediate printing of file contents

	// Rewind and read with pre-processing
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		perr(path, cfg, fmt.Sprintf("seek error: %v", err))
		return
	}
    raw, err := io.ReadAll(io.LimitReader(f, int64(cfg.MaxChars*4))) // read some extra for compaction
	if err != nil {
		perr(path, cfg, fmt.Sprintf("read error: %v", err))
		return
	}

	compact := compactTextBytes(raw, cfg)

    // Removed pre-summary printing to avoid dumping file contents

	// Stream LLM summary
    // Pre-only mode disabled: always proceed to LLM summary
    ctx, cancel := context.WithTimeout(context.Background(), time.Duration(cfg.TimeoutSec)*time.Second)
	defer cancel()
    providers, err := selectProviders(cfg)
    if err != nil || len(providers) == 0 {
        if err != nil { perr(path, cfg, err.Error()) } else { perr(path, cfg, "no provider keys found; run with -x to discover") }
        return
    }
    prompt := buildPrompt(cfg, path, lang, compact)
    streamWithFallback(ctx, cfg, providers, prompt, path, lang)

    // Suppress trailing done banner for clean summary-only output
}

func processStdin(cfg Config) {
    name := "stdin"
    br := bufio.NewReader(os.Stdin)
    // Peek initial sample
    buf, _ := br.Peek(4096)
    if bytes.Contains(buf, []byte{0}) {
        perr(name, cfg, "binary or non-text; skipping")
        return
    }
    if !isValidUTF8(buf) {
        perr(name, cfg, "non-utf8 input; skipping")
        return
    }
    // no eager previews or pre-summaries
    lang := ""
    // Removed eager preview for stdin
    // Read up to MaxChars*4 from stdin
    raw, _ := io.ReadAll(io.LimitReader(br, int64(cfg.MaxChars*4)))
    compact := compactTextBytes(raw, cfg)
    // Removed pre-summary printing for stdin
    // Pre-only mode disabled for stdin
    ctx, cancel := context.WithTimeout(context.Background(), time.Duration(cfg.TimeoutSec)*time.Second)
    defer cancel()
    providers, err := selectProviders(cfg)
    if err != nil || len(providers) == 0 {
        if err != nil { perr(name, cfg, err.Error()) } else { perr(name, cfg, "no provider keys found; run with -x to discover") }
        return
    }
    prompt := buildPrompt(cfg, name, lang, compact)
    streamWithFallback(ctx, cfg, providers, prompt, name, lang)
    // Suppress trailing done banner for clean summary-only output
}

// sniffText returns whether the file is text-like and a small sample for pre-summary
func sniffText(r io.Reader, n int) (bool, string, error) {
	buf := make([]byte, n)
	m, err := io.ReadFull(r, buf)
	if err != nil && !errors.Is(err, io.ErrUnexpectedEOF) && !errors.Is(err, io.EOF) {
		return false, "", err
	}
	buf = buf[:m]
	if bytes.Contains(buf, []byte{0}) {
		return false, "", nil
	}
	// crude MIME guess
	mt := http.DetectContentType(buf)
	if strings.Contains(mt, "text/") || mt == "application/json" || mt == "application/xml" || mt == "application/javascript" {
		return true, string(buf), nil
	}
	// fallback to UTF-8 check
	if !isValidUTF8(buf) {
		return false, "", nil
	}
	return true, string(buf), nil
}

func isValidUTF8(b []byte) bool {
    return utf8.Valid(b)
}

func languageFromPath(path string) string {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".go":
		return "Go"
	case ".rs":
		return "Rust"
	case ".ts":
		return "TypeScript"
	case ".tsx":
		return "TSX"
	case ".js":
		return "JavaScript"
	case ".jsx":
		return "JSX"
	case ".py":
		return "Python"
	case ".java":
		return "Java"
	case ".rb":
		return "Ruby"
	case ".php":
		return "PHP"
	case ".md":
		return "Markdown"
	case ".json":
		return "JSON"
	case ".yaml", ".yml":
		return "YAML"
	case ".toml":
		return "TOML"
	case ".c":
		return "C"
	case ".cpp", ".cc", ".cxx", ".hpp", ".h":
		return "C++"
	case ".css":
		return "CSS"
	case ".html", ".htm":
		return "HTML"
	default:
		return strings.TrimPrefix(ext, ".")
	}
}

// compactText trims excessive whitespace, collapses blank lines, collects head/tail, keeps likely symbol lines
func compactText(s string, maxChars int, headN, tailN, symbolsN int) string {
    if len(s) <= maxChars {
        return s
    }
    // Stream-compaction in one pass: head, tail ring, symbol lines
    scanner := bufio.NewScanner(strings.NewReader(s))
    scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
    head := make([]string, 0, headN)
    tail := make([]string, 0, tailN)
    symbolLines := make([]string, 0, symbolsN)
    lineIndex := 0
    for scanner.Scan() {
        line := scanner.Text()
        if len(head) < headN {
            head = append(head, line)
        } else {
            tail = append(tail, line)
            if len(tail) > tailN {
                tail = tail[1:]
            }
        }
        if lineIndex >= headN { // only consider symbol lines after head region
            if isSymbolLine(strings.TrimSpace(line)) {
                symbolLines = append(symbolLines, line)
                if len(symbolLines) > symbolsN {
                    symbolLines = symbolLines[1:]
                }
            }
        }
        lineIndex++
    }

    // collapse blank runs
    collapse := func(in []string) []string {
        out := make([]string, 0, len(in))
        blank := 0
        for _, l := range in {
            if strings.TrimSpace(l) == "" {
                blank++
                if blank > 2 {
                    continue
                }
            } else {
                blank = 0
            }
            out = append(out, l)
        }
        return out
    }

    head = collapse(head)
    tail = collapse(tail)
    symbolLines = collapse(symbolLines)

    var buf strings.Builder
    buf.Grow(min(maxChars, len(s)))
    writeSection := func(title string, ls []string) {
        if len(ls) == 0 {
            return
        }
        buf.WriteString("\n// == ")
        buf.WriteString(title)
        buf.WriteString(" ==\n")
        for _, l := range ls {
            buf.WriteString(strings.TrimRight(l, "\r"))
            buf.WriteByte('\n')
            if buf.Len() > maxChars {
                break
            }
        }
    }

    writeSection("HEAD", head)
    writeSection("SYMBOLS", symbolLines)
    writeSection("TAIL", tail)

    out := buf.String()
    if len(out) > maxChars {
        return out[:maxChars]
    }
    if out == "" {
        // as fallback just truncate original
        if len(s) > maxChars {
            return s[:maxChars]
        }
        return s
    }
    return out
}

// compactTextBytes avoids a large []byte -> string copy by scanning bytes.
// It delegates to compactText at the end with the same parameters.
func compactTextBytes(b []byte, cfg Config) string {
    if len(b) <= cfg.MaxChars {
        return string(b)
    }
    scanner := bufio.NewScanner(bytes.NewReader(b))
    scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
    head := make([][]byte, 0, cfg.HeadLines)
    tail := make([][]byte, 0, cfg.TailLines)
    symbols := make([][]byte, 0, cfg.SymbolLines)
    lineIndex := 0
    for scanner.Scan() {
        // Copy the line because Scanner reuses its buffer
        line := append([]byte(nil), scanner.Bytes()...)
        if len(head) < cfg.HeadLines {
            head = append(head, line)
        } else {
            tail = append(tail, line)
            if len(tail) > cfg.TailLines {
                tail = tail[1:]
            }
        }
        if lineIndex >= cfg.HeadLines {
            if isSymbolLine(string(bytes.TrimSpace(line))) {
                symbols = append(symbols, line)
                if len(symbols) > cfg.SymbolLines {
                    symbols = symbols[1:]
                }
            }
        }
        lineIndex++
    }
    collapse := func(in [][]byte) [][]byte {
        out := make([][]byte, 0, len(in))
        blank := 0
        for _, l := range in {
            if len(bytes.TrimSpace(l)) == 0 {
                blank++
                if blank > 2 { continue }
            } else {
                blank = 0
            }
            out = append(out, l)
        }
        return out
    }
    head = collapse(head)
    tail = collapse(tail)
    symbols = collapse(symbols)

    var buf bytes.Buffer
    buf.Grow(min(cfg.MaxChars, len(b)))
    writeSection := func(title string, ls [][]byte) {
        if len(ls) == 0 { return }
        buf.WriteString("\n// == ")
        buf.WriteString(title)
        buf.WriteString(" ==\n")
        for _, l := range ls {
            // trim CR
            if n := len(l); n > 0 && l[n-1] == '\r' { l = l[:len(l)-1] }
            buf.Write(l)
            buf.WriteByte('\n')
            if buf.Len() > cfg.MaxChars { break }
        }
    }
    writeSection("HEAD", head)
    writeSection("SYMBOLS", symbols)
    writeSection("TAIL", tail)

    out := buf.Bytes()
    if len(out) > cfg.MaxChars {
        out = out[:cfg.MaxChars]
    }
    if len(out) == 0 {
        if len(b) > cfg.MaxChars { return string(b[:cfg.MaxChars]) }
        return string(b)
    }
    return string(out)
}

func isSymbolLine(trim string) bool {
	if trim == "" {
		return false
	}
	// Heuristics: function/def/class/package/import/struct/enum/interface etc.
	lower := strings.ToLower(trim)
    if strings.HasPrefix(lower, "def ") || strings.HasPrefix(lower, "class ") || strings.HasPrefix(lower, "package ") || strings.HasPrefix(lower, "import ") || strings.HasPrefix(lower, "func ") || strings.HasPrefix(lower, "struct ") || strings.HasPrefix(lower, "enum ") || strings.HasPrefix(lower, "interface ") || strings.HasPrefix(lower, "type ") || strings.HasPrefix(lower, "const ") || strings.HasPrefix(lower, "var ") {
		return true
	}
	// brace followed signatures
	if strings.Contains(trim, "(") && strings.HasSuffix(trim, ") {") {
		return true
	}
	if strings.HasPrefix(trim, "// ") || strings.HasPrefix(trim, "# ") || strings.HasPrefix(trim, "/* ") {
		return true
	}
	return false
}

// preSummary removed (was unused)

func buildPrompt(cfg Config, path, lang, content string) string {
    // Always ask for a strict JSON object so we can parse without string filtering.
    if cfg.Diagram {
        return fmt.Sprintf(
            "You are a senior engineer. Read the following %s file `%s`. Respond with EXACTLY ONE JSON object and nothing else (no prose, no code fences). The JSON MUST match this schema strictly:\n"+
                "- summary: string (2–3 concise sentences)\n"+
                "- bullets: array of exactly 3 strings\n"+
                "- ascii_diagram: string or null (use null if a diagram would not add clarity; do NOT explain omission)\n\n"+
                "Output shape:\n"+
                "{\n  \"summary\": \"string\",\n  \"bullets\": [\n    \"string\",\n    \"string\",\n    \"string\"\n  ],\n  \"ascii_diagram\": null\n}\n\n"+
                "Rules: no headings or labels, no backticks, no content outside the single JSON object.\n\n"+
                "Context (compacted):\n\n%s",
            lang, filepath.Base(path), content,
        )
    }
    // No diagram requested: keep schema minimal
    return fmt.Sprintf(
        "You are a senior engineer. Read the following %s file `%s`. Respond with EXACTLY ONE JSON object and nothing else (no prose, no code fences). The JSON MUST match this schema strictly:\n"+
            "- summary: string (2–3 concise sentences)\n"+
            "- bullets: array of exactly 3 strings\n\n"+
            "Output shape:\n"+
            "{\n  \"summary\": \"string\",\n  \"bullets\": [\n    \"string\",\n    \"string\",\n    \"string\"\n  ]\n}\n\n"+
            "Rules: no headings or labels, no backticks, no content outside the single JSON object.\n\n"+
            "Context (compacted):\n\n%s",
        lang, filepath.Base(path), content,
    )
}

// quickPreview removed (unused)

// ===== Key discovery, verification, and caching (-x)

type keyCache struct {
    OpenAI    string    `json:"openai,omitempty"`
    Anthropic string    `json:"anthropic,omitempty"`
    UpdatedAt time.Time `json:"updated_at"`
}

func configBaseDir() (string, error) {
    if v := os.Getenv("LLMCAT_CONFIG_DIR"); v != "" {
        if err := os.MkdirAll(v, 0o755); err != nil { return "", err }
        return v, nil
    }
    dir, err := os.UserConfigDir()
    if err != nil { return "", err }
    d := filepath.Join(dir, "llmcat")
    if err := os.MkdirAll(d, 0o755); err != nil { return "", err }
    return d, nil
}

func configFilePath() (string, error) {
    d, err := configBaseDir()
    if err != nil { return "", err }
    return filepath.Join(d, "keys.json"), nil
}

// defaultConfigFilePath returns the default path for loading CLI config
func defaultConfigFilePath() (string, error) {
    d, err := configBaseDir()
    if err != nil { return "", err }
    return filepath.Join(d, "config.json"), nil
}

// readConfigOverrides reads simple JSON with selected fields and returns a map
// Typed overrides struct with pointer fields so omitted values are nil
type ConfigOverrides struct {
    Provider     *string `json:"provider"`
    Model        *string `json:"model"`
    Format       *string `json:"format"`
    MaxChars     *int    `json:"max_chars"`
    HeadLines    *int    `json:"head"`
    TailLines    *int    `json:"tail"`
    SymbolLines  *int    `json:"symbol_lines"`
    Concurrency  *int    `json:"concurrency"`
    NoColor      *bool   `json:"no_color"`
    DiscoverKeys *bool   `json:"x"`
    TimeoutSec   *int    `json:"timeout"`
    MaxTokens    *int    `json:"max_tokens"`
    NoPrefix     *bool   `json:"no_prefix"`
    Preset       *string `json:"preset"`
    OneLine      *bool   `json:"one_line"`
    Diagram      *bool   `json:"diagram"`
    XClear       *bool   `json:"x_clear"`
}

func readConfigOverridesTyped(path string) (ConfigOverrides, bool) {
    var o ConfigOverrides
    b, err := os.ReadFile(path)
    if err != nil { return o, false }
    if err := json.Unmarshal(b, &o); err != nil { return o, false }
    return o, true
}

// legacy map-based getters removed

// applyConfigOverrides sets cfg fields only if the corresponding CLI flag wasn't set
func applyConfigOverridesTyped(cfg *Config, o ConfigOverrides, set map[string]bool) {
    if o.Provider != nil && !set["provider"] { cfg.Provider = Provider(*o.Provider) }
    if o.Model != nil && !set["model"] { cfg.Model = *o.Model }
    if o.Format != nil && !set["format"] { cfg.Format = *o.Format }
    if o.MaxChars != nil && !set["max-chars"] { cfg.MaxChars = *o.MaxChars }
    if o.HeadLines != nil && !set["head"] { cfg.HeadLines = *o.HeadLines }
    if o.TailLines != nil && !set["tail"] { cfg.TailLines = *o.TailLines }
    if o.SymbolLines != nil && !set["symbol-lines"] { cfg.SymbolLines = *o.SymbolLines }
    if o.Concurrency != nil && !set["concurrency"] { cfg.Concurrency = *o.Concurrency }
    if o.NoColor != nil && !set["no-color"] { cfg.NoColor = *o.NoColor }
    if o.DiscoverKeys != nil && !set["x"] { cfg.DiscoverKeys = *o.DiscoverKeys }
    if o.TimeoutSec != nil && !set["timeout"] { cfg.TimeoutSec = *o.TimeoutSec }
    if o.MaxTokens != nil && !set["max-tokens"] { cfg.MaxTokens = *o.MaxTokens }
    if o.NoPrefix != nil && !set["no-prefix"] { cfg.NoPrefix = *o.NoPrefix }
    if o.Preset != nil && !set["preset"] { cfg.Preset = *o.Preset }
    if o.OneLine != nil && !set["one-line"] { cfg.OneLine = *o.OneLine }
    if o.Diagram != nil && !set["diagram"] { cfg.Diagram = *o.Diagram }
    if o.XClear != nil && !set["x-clear"] { cfg.XClear = *o.XClear }
}

func loadKeyCache() (keyCache, bool) {
    var kc keyCache
    path, err := configFilePath()
    if err != nil {
        return kc, false
    }
    b, err := os.ReadFile(path)
    if err != nil {
        return kc, false
    }
    if err := json.Unmarshal(b, &kc); err != nil {
        return kc, false
    }
    return kc, true
}

func saveKeyCache(kc keyCache) error {
    path, err := configFilePath()
    if err != nil {
        return err
    }
    kc.UpdatedAt = time.Now()
    b, err := json.MarshalIndent(kc, "", "  ")
    if err != nil {
        return err
    }
    return os.WriteFile(path, b, 0o600)
}

func clearKeyCache() error {
    path, err := configFilePath()
    if err != nil {
        return err
    }
    if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
        return err
    }
    return nil
}

func runKeyDiscovery(ctx context.Context, cfg Config) error {
    // 1) Try cached keys and verify
    if kc, ok := loadKeyCache(); ok {
        verified := false
        if kc.OpenAI != "" && verifyOpenAIKey(ctx, kc.OpenAI, "gpt-4o-mini") {
            os.Setenv("OPENAI_API_KEY", kc.OpenAI)
            verified = true
        }
        if kc.Anthropic != "" && verifyAnthropicKey(ctx, kc.Anthropic, "claude-3-5-sonnet-latest") {
            os.Setenv("ANTHROPIC_API_KEY", kc.Anthropic)
            // also set CLAUDE_API_KEY for compatibility
            os.Setenv("CLAUDE_API_KEY", kc.Anthropic)
            verified = true
        }
        if verified {
            return nil // do not scan files again
        }
    }

    // 2) Check environment variables
    var found keyCache
    if v := os.Getenv("OPENAI_API_KEY"); v != "" {
        if verifyOpenAIKey(ctx, v, "gpt-4o-mini") {
            found.OpenAI = v
        }
    }
    // Anthropic or Claude alias
    anth := os.Getenv("ANTHROPIC_API_KEY")
    if anth == "" {
        anth = os.Getenv("CLAUDE_API_KEY")
    }
    if anth != "" {
        if verifyAnthropicKey(ctx, anth, "claude-3-5-sonnet-latest") {
            found.Anthropic = anth
        }
    }
    // Optional: verify Google key quickly if present
    if v := os.Getenv("GOOGLE_API_KEY"); v != "" {
        // Minimal ping for Google is not exposed as simple; treat presence as hint
        // We do not cache Google key here as we don't store it in keyCache
        _ = v
    }

    // If we found any verified key in env, cache and set
    if found.OpenAI != "" || found.Anthropic != "" {
        if found.OpenAI != "" {
            os.Setenv("OPENAI_API_KEY", found.OpenAI)
        }
        if found.Anthropic != "" {
            os.Setenv("ANTHROPIC_API_KEY", found.Anthropic)
            os.Setenv("CLAUDE_API_KEY", found.Anthropic)
        }
        // Merge with prior cache if exists
        prev, _ := loadKeyCache()
        if found.OpenAI == "" {
            found.OpenAI = prev.OpenAI
        }
        if found.Anthropic == "" {
            found.Anthropic = prev.Anthropic
        }
        _ = saveKeyCache(found)
        return nil
    }

    // 3) Scan from the user's HOME, max depth 3, and only files with names starting with ".env"
    foundFS := keyCache{}
    ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
    defer cancel()
    home, herr := os.UserHomeDir()
    if herr == nil {
        root := home
        maxDepth := 3
        _ = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
            if err != nil { return nil }
            // enforce depth limit
            if d.IsDir() {
                rel, relErr := filepath.Rel(root, path)
                if relErr != nil { return nil }
                if rel == "." { return nil }
                depth := 1
                if rel != "" {
                    depth = len(strings.Split(rel, string(os.PathSeparator)))
                }
                if depth > maxDepth { return fs.SkipDir }
                return nil
            }
            // only .env* files
            base := filepath.Base(path)
            if !strings.HasPrefix(base, ".env") {
                return nil
            }
            // Limit size to 1MB
            info, ierr := d.Info()
            if ierr == nil && info.Size() > 1<<20 {
                return nil
            }
            f, oerr := os.Open(path)
            if oerr != nil { return nil }
            // Read up to 1MiB and validate as text
            b, rerr := io.ReadAll(io.LimitReader(f, 1<<20))
            _ = f.Close()
            if rerr != nil { return nil }
            if bytes.Contains(b, []byte{0}) || !utf8.Valid(b) { return nil }
            // parse lines for keys
            checkLine := func(line, key string) (string, bool) {
                idx := strings.Index(line, key)
                if idx < 0 { return "", false }
                j := idx + len(key)
                for j < len(line) && (line[j] == ' ' || line[j] == '\t') { j++ }
                if j < len(line) && (line[j] == '=' || line[j] == ':') { j++ }
                for j < len(line) && (line[j] == ' ' || line[j] == '\t') { j++ }
                if j >= len(line) { return "", false }
                var val string
                if line[j] == '"' || line[j] == '\'' {
                    q := line[j]; j++; k := j
                    for k < len(line) && line[k] != q { k++ }
                    val = line[j:k]
                } else {
                    k := j
                    for k < len(line) && !isSpaceOrStop(line[k]) { k++ }
                    val = line[j:k]
                }
                val = strings.TrimSpace(val)
                if len(val) >= 16 { return val, true }
                return "", false
            }
            sc := bufio.NewScanner(bytes.NewReader(b))
            for sc.Scan() {
                line := sc.Text()
                if foundFS.OpenAI == "" {
                    if v, ok := checkLine(line, "OPENAI_API_KEY"); ok {
                        if verifyOpenAIKey(ctx, v, "gpt-4o-mini") {
                            foundFS.OpenAI = v
                            os.Setenv("OPENAI_API_KEY", v)
                        }
                    }
                }
                if foundFS.Anthropic == "" {
                    if v, ok := checkLine(line, "ANTHROPIC_API_KEY"); ok {
                        if verifyAnthropicKey(ctx, v, "claude-3-5-sonnet-latest") {
                            foundFS.Anthropic = v
                            os.Setenv("ANTHROPIC_API_KEY", v)
                            os.Setenv("CLAUDE_API_KEY", v)
                        }
                    } else if v, ok := checkLine(line, "CLAUDE_API_KEY"); ok {
                        if verifyAnthropicKey(ctx, v, "claude-3-5-sonnet-latest") {
                            foundFS.Anthropic = v
                            os.Setenv("ANTHROPIC_API_KEY", v)
                            os.Setenv("CLAUDE_API_KEY", v)
                        }
                    }
                }
                if foundFS.OpenAI != "" && foundFS.Anthropic != "" { break }
            }
            return nil
        })
    }
    if foundFS.OpenAI != "" || foundFS.Anthropic != "" {
        // Merge with prior cache
        prev, _ := loadKeyCache()
        if foundFS.OpenAI == "" {
            foundFS.OpenAI = prev.OpenAI
        }
        if foundFS.Anthropic == "" {
            foundFS.Anthropic = prev.Anthropic
        }
        _ = saveKeyCache(foundFS)
    }
    return nil
}

func isSpaceOrStop(b byte) bool {
    switch b {
    case ' ', '\t', '\r', '\n':
        return true
    case ';', '#', ',':
        return true
    default:
        return false
    }
}

func verifyOpenAIKey(ctx context.Context, apiKey, model string) bool {
    if apiKey == "" {
        return false
    }
    ctx, cancel := context.WithTimeout(ctx, 8*time.Second)
    defer cancel()
    body := fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"hi"}],"max_tokens":1}`, model)
    req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/chat/completions", strings.NewReader(body))
    if err != nil {
        return false
    }
    req.Header.Set("Authorization", "Bearer "+apiKey)
    req.Header.Set("Content-Type", "application/json")
    // Use a short-lived client with context timeout; shared client is fine too
    resp, err := httpClient.Do(req)
    if err != nil {
        return false
    }
    defer resp.Body.Close()
    return resp.StatusCode/100 == 2
}

func verifyAnthropicKey(ctx context.Context, apiKey, model string) bool {
    if apiKey == "" {
        return false
    }
    ctx, cancel := context.WithTimeout(ctx, 8*time.Second)
    defer cancel()
    body := fmt.Sprintf(`{"model":%q,"max_tokens":1,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`, model)
    req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.anthropic.com/v1/messages", strings.NewReader(body))
    if err != nil {
        return false
    }
    req.Header.Set("x-api-key", apiKey)
    req.Header.Set("anthropic-version", "2023-06-01")
    req.Header.Set("content-type", "application/json")
    resp, err := httpClient.Do(req)
    if err != nil {
        return false
    }
    defer resp.Body.Close()
    return resp.StatusCode/100 == 2
}

// ProviderClient streams summary tokens

type ProviderClient interface {
	StreamSummary(ctx context.Context, cfg Config, prompt string, out chan<- string) error
}

// pickProvider removed (unused; use selectProviders)

// ===== Output helpers with per-file prefix and optional color

var outMu sync.Mutex

// isTTY removed (unused)

// Minimal prefix color animation: we animate only the filename prefix.

func filePrefix(path string, cfg Config) string {
    name := filepath.Base(path)
    if cfg.NoPrefix {
        return ""
    }
    if cfg.NoColor {
        return fmt.Sprintf("[%s] ", name)
    }
    // Base static light gray prefix
    return fmt.Sprintf("\x1b[37m[%s]\x1b[0m ", name)
}

// chooseGrayPalette selects ANSI sequences for leader, two-trail gradient, and background.
// Prefers 256-color grays for better contrast; falls back to 8-color.
func chooseGrayPalette() (leader, tail1, tail2, bg string) {
    term := strings.ToLower(os.Getenv("TERM"))
    colorterm := strings.ToLower(os.Getenv("COLORTERM"))
    supports256 := strings.Contains(term, "256color") || strings.Contains(colorterm, "truecolor") || strings.Contains(colorterm, "24bit")
    if supports256 {
        // 231: white (leader), 250: light gray (trail1), 246: medium gray (trail2),
        // 244: slightly dim gray (bg) — lighter than 240 to improve readability
        return "\x1b[38;5;231m", "\x1b[38;5;250m", "\x1b[38;5;246m", "\x1b[38;5;244m"
    }
    // 8-color fallback: lighten background from bright black (90) to white (37)
    // to keep the filename readable during animation on basic terminals.
    return "\x1b[97m", "\x1b[37m", "\x1b[2;37m", "\x1b[37m"
}

// startPrefixAnimation renders a subtle sweep across the filename prefix.
// One bright white character leads, followed by a short light-gray trail
// for better motion persistence. The rest remain dim gray. Redraws in-place
// until stopped.
func startPrefixAnimation(path string, cfg Config) (stop func(), wait func()) {
    if cfg.NoColor || cfg.NoPrefix || cfg.OneLine {
        return func() {}, func() {}
    }
    name := filepath.Base(path)
    // Visible prefix content without ANSI so we can index runes accurately
    base := fmt.Sprintf("[%s] ", name)
    // Precompute runes for sweeping
    runes := []rune(base)
    n := len(runes)
    if n == 0 {
        return func() {}, func() {}
    }
    stopCh := make(chan struct{})
    var wg sync.WaitGroup
    wg.Add(1)
    go func() {
        defer wg.Done()
        // Slightly slower tick for a calmer sweep
        t := time.NewTicker(120 * time.Millisecond)
        defer t.Stop()
        i := 0
        leaderANSI, tail1ANSI, tail2ANSI, bgANSI := chooseGrayPalette()
        for {
            select {
            case <-stopCh:
                // On stop, restore the original static prefix coloring
                // and leave the cursor at the end of the prefix.
                if !cfg.NoPrefix && !cfg.NoColor && !cfg.OneLine {
                    outMu.Lock()
                    fmt.Fprint(os.Stdout, "\r"+filePrefix(path, cfg))
                    outMu.Unlock()
                }
                return
            case <-t.C:
                // Build dim gray prefix with one bright white rune at position i
                // and a two-step trailing gradient behind it.
                var b strings.Builder
                b.Grow(n * 6) // rough capacity for ANSI
                trail := 2 // number of trailing positions behind the leader
                for idx, r := range runes {
                    if idx == i {
                        // leader: bright white
                        b.WriteString(leaderANSI)
                        b.WriteRune(r)
                        b.WriteString("\x1b[0m")
                        continue
                    }
                    // Compute circular distance behind the leader
                    d := (i - idx + n) % n
                    if d > 0 && d <= trail {
                        // trail gradient: d=1 (brighter), d=2 (softer)
                        if d == 1 {
                            b.WriteString(tail1ANSI)
                        } else {
                            b.WriteString(tail2ANSI)
                        }
                        b.WriteRune(r)
                        b.WriteString("\x1b[0m")
                    } else {
                        // background: dim gray
                        b.WriteString(bgANSI)
                        b.WriteRune(r)
                        b.WriteString("\x1b[0m")
                    }
                }
                outMu.Lock()
                // carriage return to redraw prefix in place
                fmt.Fprint(os.Stdout, "\r"+b.String())
                outMu.Unlock()
                i = (i + 1) % n
            }
        }
    }()
    return func() { close(stopCh) }, func() { wg.Wait() }
}

func pout(path string, cfg Config, s string) {
    prefix := filePrefix(path, cfg)
	outMu.Lock()
	defer outMu.Unlock()
	scanner := bufio.NewScanner(strings.NewReader(s))
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	for scanner.Scan() {
        line := colorizeLine(scanner.Text(), cfg)
        if prefix == "" { fmt.Println(line) } else { fmt.Println(prefix + line) }
	}
}

// poutSinglePrefix prints a multi-line string with the file prefix exactly once
// on the first emitted line, and does not repeat it for subsequent lines.
func poutSinglePrefix(path string, cfg Config, s string) {
    prefix := filePrefix(path, cfg)
    outMu.Lock()
    defer outMu.Unlock()
    scanner := bufio.NewScanner(strings.NewReader(s))
    scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
    first := true
    for scanner.Scan() {
        line := colorizeLine(scanner.Text(), cfg)
        if first {
            if prefix != "" {
                fmt.Print(prefix)
            }
            fmt.Println(line)
            first = false
        } else {
            // indent second..nth lines by one tab for readability
            fmt.Println("\t" + line)
        }
    }
}

// perr prints an error message for a path with standard prefixing
func perr(path string, cfg Config, msg string) {
    prefix := filePrefix(path, cfg)
    outMu.Lock()
    defer outMu.Unlock()
    if cfg.OneLine {
        // Collapse to a single line
        line := strings.SplitN(msg, "\n", 2)[0]
        line = collapseWhitespace(line)
        fmt.Fprintln(os.Stderr, prefix+"error: "+line)
        return
    }
    scanner := bufio.NewScanner(strings.NewReader(msg))
    scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
    for scanner.Scan() {
        fmt.Fprintln(os.Stderr, prefix+"error: "+scanner.Text())
    }
}

// poutNoPrefix prints raw chunks but still prefixes newlines with file prefix for readability
// poutNoPrefix removed (unused)

// streamPrinter writes streaming chunks contiguously on a single line without
// inserting newlines per chunk and without repeating prefixes. It holds the
// global output lock for the duration of the stream to avoid interleaving.
// streamPrinter removed (unused)

// filterStreamChunk removes unwanted headings/labels from streamed content.
// It keeps a small tail buffer to handle phrases split across chunks.
// filterStreamChunk and removeCaseInsensitive removed (unused)

// colorizeLine adds ANSI styling to common structures (bullets, headings, code fences)
func colorizeLine(line string, cfg Config) string {
    if cfg.NoColor {
        return line
    }
    // Preserve indentation for bullets
    trimmed := strings.TrimLeft(line, " ")
    indent := len(line) - len(trimmed)
    // Bullets: -, *, • at line start
    if strings.HasPrefix(trimmed, "- ") || strings.HasPrefix(trimmed, "* ") || strings.HasPrefix(trimmed, "• ") {
        // replace marker with colored bullet
        rest := strings.TrimPrefix(trimmed, "- ")
        rest = strings.TrimPrefix(rest, "* ")
        rest = strings.TrimPrefix(rest, "• ")
        return strings.Repeat(" ", indent) + "\x1b[36m•\x1b[0m " + rest
    }
    // Markdown-style headings
    if strings.HasPrefix(trimmed, "# ") || strings.HasPrefix(trimmed, "## ") || strings.HasPrefix(trimmed, "### ") {
        return strings.Repeat(" ", indent) + "\x1b[1m" + trimmed + "\x1b[0m"
    }
    // Fences/light formatting cues
    if strings.HasPrefix(trimmed, "```") {
        return strings.Repeat(" ", indent) + "\x1b[2m" + trimmed + "\x1b[0m"
    }
    // Pre-summary dividers
    if strings.HasPrefix(trimmed, "-- ") || strings.HasPrefix(trimmed, "== ") || strings.HasPrefix(trimmed, "[done") {
        return strings.Repeat(" ", indent) + "\x1b[2m" + trimmed + "\x1b[0m"
    }
    return line
}

// Emit helpers support different formats in the future
// emitPreview/emitPreSummary/emitDone removed (unused)

// printPrefixOnce writes the file prefix to stdout immediately so the user sees
// "[FILENAME]: " before any streaming tokens arrive. It does not hold the lock.
func printPrefixOnce(path string, cfg Config) {
    p := filePrefix(path, cfg)
    if p == "" { return }
    outMu.Lock()
    fmt.Fprint(os.Stdout, p)
    outMu.Unlock()
}

// selectProviders returns a prioritized list based on cfg.Provider and available keys
func selectProviders(cfg Config) ([]ProviderClient, error) {
    switch cfg.Provider {
    case ProviderOpenAI:
        if os.Getenv("OPENAI_API_KEY") == "" {
            if ollamaAvailable(800 * time.Millisecond) { return []ProviderClient{&OllamaClient{}}, nil }
            return nil, fmt.Errorf("OPENAI_API_KEY not set; set it or use --provider auto/ollama")
        }
        return []ProviderClient{&OpenAIClient{}}, nil
    case ProviderAnthropic:
        if os.Getenv("ANTHROPIC_API_KEY") == "" && os.Getenv("CLAUDE_API_KEY") == "" {
            if ollamaAvailable(800 * time.Millisecond) { return []ProviderClient{&OllamaClient{}}, nil }
            return nil, fmt.Errorf("ANTHROPIC_API_KEY not set; set it or use --provider auto/ollama")
        }
        return []ProviderClient{&AnthropicClient{}}, nil
    case ProviderGoogle:
        if os.Getenv("GOOGLE_API_KEY") == "" {
            if ollamaAvailable(800 * time.Millisecond) { return []ProviderClient{&OllamaClient{}}, nil }
            return nil, fmt.Errorf("GOOGLE_API_KEY not set; set it or use --provider auto/ollama")
        }
        return []ProviderClient{&GoogleClient{}}, nil
    case ProviderOllama:
        return []ProviderClient{&OllamaClient{}}, nil
    case ProviderAuto:
        out := []ProviderClient{}
        // Prefer local Ollama if available
        if ollamaAvailable(800 * time.Millisecond) { out = append(out, &OllamaClient{}) }
        if os.Getenv("OPENAI_API_KEY") != "" { out = append(out, &OpenAIClient{}) }
        if os.Getenv("ANTHROPIC_API_KEY") != "" || os.Getenv("CLAUDE_API_KEY") != "" { out = append(out, &AnthropicClient{}) }
        if os.Getenv("GOOGLE_API_KEY") != "" { out = append(out, &GoogleClient{}) }
        if len(out) == 0 { return nil, fmt.Errorf("no provider keys found; run with -x to discover") }
        return out, nil
    default:
        return nil, fmt.Errorf("unknown provider: %s", cfg.Provider)
    }
}

// streamWithFallback tries providers in order, switching on error
func streamWithFallback(ctx context.Context, cfg Config, providers []ProviderClient, prompt, path, lang string) {
    // Always collect JSON and render from it; avoids heuristic string filtering.
    // Print the filename prefix immediately so it never flickers or disappears.
    if !cfg.OneLine {
        printPrefixOnce(path, cfg)
    }
    // Start a lightweight prefix animation while we wait for output
    stopAnim, waitAnim := startPrefixAnimation(path, cfg)
    for i, p := range providers {
        stream := make(chan string, 16)
        errCh := make(chan error, 1)
        go func(prov ProviderClient) {
            defer close(stream)
            if err := prov.StreamSummary(ctx, cfg, prompt, stream); err != nil {
                errCh <- err
            } else {
                errCh <- nil
            }
        }(p)
        hadData := false
        done := false
        var collected strings.Builder
        // No filtering needed; we always parse JSON at the end
        for !done {
            select {
            case chunk, ok := <-stream:
                if !ok { done = true; break }
                if chunk != "" { hadData = true }
                // buffer raw to parse later
                collected.WriteString(chunk)
            case err := <-errCh:
                if err != nil && !hadData {
                    if i < len(providers)-1 {
                        perr(path, cfg, fmt.Sprintf("provider failed, trying next: %v", err))
                    } else {
                        // last provider and no data: single-line error if enabled
                        stopAnim(); waitAnim()
                        perr(path, cfg, err.Error())
                        return
                    }
                }
                // nothing special to flush in JSON mode
                done = true
            case <-ctx.Done():
                stopAnim(); waitAnim()
                perr(path, cfg, ctx.Err().Error())
                return
            }
        }
        if hadData || i == len(providers)-1 {
            // Render structured output and print after the already-printed prefix
            stopAnim(); waitAnim()
            formatted := formatSummaryFromLLM(cfg, collected.String())
            if cfg.OneLine {
                prefix := filePrefix(path, cfg)
                outMu.Lock()
                formatted = colorizeLine(formatted, cfg)
                if prefix == "" { fmt.Println(formatted) } else { fmt.Println(prefix + formatted) }
                outMu.Unlock()
                return
            }
            printAfterPrefix(path, cfg, formatted)
            return
        }
        // else continue to next provider
    }
}

// ===== Provider implementations

// OpenAI Responses API streaming

type OpenAIClient struct{}

func (c *OpenAIClient) StreamSummary(ctx context.Context, cfg Config, prompt string, out chan<- string) error {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return errors.New("OPENAI_API_KEY not set")
	}
	model := cfg.Model
	if model == "" {
        model = "gpt-4o-mini"
	}

    // Use Chat Completions streaming for robust incremental tokens
    reqBody := fmt.Sprintf(`{
		"model": %q,
		"messages": [{"role":"user","content":%q}],
		"stream": true
	}`, model, prompt)

    req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/chat/completions", strings.NewReader(reqBody))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

    resp, err := httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("openai http %d: %s", resp.StatusCode, string(b))
	}

    dec := newEventStreamDecoder(resp.Body)
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
        _, data, err := dec.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
        if len(data) == 0 { continue }
        // Minimal typed decode of streaming delta
        var evt struct {
            Choices []struct {
                Delta struct {
                    Content string `json:"content"`
                } `json:"delta"`
            } `json:"choices"`
        }
        if err := json.Unmarshal(data, &evt); err == nil {
            if len(evt.Choices) > 0 {
                if c := evt.Choices[0].Delta.Content; c != "" { out <- c }
            }
        }
	}
	return nil
}

// Anthropic Messages streaming

type AnthropicClient struct{}

func (c *AnthropicClient) StreamSummary(ctx context.Context, cfg Config, prompt string, out chan<- string) error {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		return errors.New("ANTHROPIC_API_KEY not set")
	}
	model := cfg.Model
	if model == "" {
		model = "claude-3-5-sonnet-latest"
	}
	// Anthropic streaming uses event stream with x-api-key and anthropic-version
	reqBody := fmt.Sprintf(`{
		"model": %q,
		"max_tokens": 1024,
		"messages": [{"role":"user","content":[{"type":"text","text":%q}]}],
		"stream": true
	}`, model, prompt)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.anthropic.com/v1/messages", strings.NewReader(reqBody))
	if err != nil {
		return err
	}
	req.Header.Set("x-api-key", apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")
	req.Header.Set("content-type", "application/json")

    resp, err := httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("anthropic http %d: %s", resp.StatusCode, string(b))
	}
    dec := newEventStreamDecoder(resp.Body)
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
        _, data, err := dec.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
        // typed extraction for text deltas
        var evt struct {
            Type  string `json:"type"`
            Delta struct {
                Type string `json:"type"`
                Text string `json:"text"`
            } `json:"delta"`
        }
        if err := json.Unmarshal(data, &evt); err == nil {
            if evt.Type == "content_block_delta" && evt.Delta.Type == "text_delta" && evt.Delta.Text != "" {
                out <- evt.Delta.Text
            }
        }
	}
	return nil
}

// Google Gemini streaming

type GoogleClient struct{}

func (c *GoogleClient) StreamSummary(ctx context.Context, cfg Config, prompt string, out chan<- string) error {
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		return errors.New("GOOGLE_API_KEY not set")
	}
	model := cfg.Model
	if model == "" {
		model = "gemini-1.5-flash"
	}

	// Gemini streaming via REST: POST to streamGenerateContent?alt=sse
	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent?alt=sse&key=%s", model, apiKey)
	reqBody := fmt.Sprintf(`{
		"contents": [{"role":"user","parts":[{"text":%q}]}]
	}`, prompt)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(reqBody))
	if err != nil {
		return err
	}
	req.Header.Set("content-type", "application/json")

    resp, err := httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("google http %d: %s", resp.StatusCode, string(b))
	}

    dec := newEventStreamDecoder(resp.Body)
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
        _, data, err := dec.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
        if len(data) == 0 { continue }
        // Minimal typed decode to extract streaming text part
        var evt struct {
            Candidates []struct {
                Content struct {
                    Parts []struct{ Text string `json:"text"` } `json:"parts"`
                } `json:"content"`
            } `json:"candidates"`
        }
        if err := json.Unmarshal(data, &evt); err == nil {
            if len(evt.Candidates) > 0 {
                parts := evt.Candidates[0].Content.Parts
                if len(parts) > 0 {
                    for _, p := range parts {
                        if p.Text != "" { out <- p.Text }
                    }
                }
            }
        }
	}
	return nil
}

// ===== Ollama local server (chat) streaming

type OllamaClient struct{}

func ollamaHost() string {
    if v := os.Getenv("OLLAMA_HOST"); v != "" {
        return strings.TrimRight(v, "/")
    }
    return "http://127.0.0.1:11434"
}

var (
    ollamaMu    sync.Mutex
    ollamaCheck struct{
        last time.Time
        ok   bool
    }
)

func ollamaAvailable(timeout time.Duration) bool {
    // simple 1s TTL cache
    ollamaMu.Lock()
    if time.Since(ollamaCheck.last) < time.Second {
        ok := ollamaCheck.ok
        ollamaMu.Unlock()
        return ok
    }
    ollamaMu.Unlock()
    ctx, cancel := context.WithTimeout(context.Background(), timeout)
    defer cancel()
    req, err := http.NewRequestWithContext(ctx, http.MethodGet, ollamaHost()+"/api/tags", nil)
    if err != nil {
        ollamaMu.Lock(); ollamaCheck.last = time.Now(); ollamaCheck.ok = false; ollamaMu.Unlock()
        return false
    }
    resp, err := httpClient.Do(req)
    if err != nil {
        ollamaMu.Lock(); ollamaCheck.last = time.Now(); ollamaCheck.ok = false; ollamaMu.Unlock()
        return false
    }
    defer resp.Body.Close()
    ok := resp.StatusCode/100 == 2
    ollamaMu.Lock(); ollamaCheck.last = time.Now(); ollamaCheck.ok = ok; ollamaMu.Unlock()
    return ok
}

func (c *OllamaClient) pickModel(ctx context.Context, cfg Config) string {
    if cfg.Model != "" { return cfg.Model }
    // Query available models
    req, err := http.NewRequestWithContext(ctx, http.MethodGet, ollamaHost()+"/api/tags", nil)
    if err != nil { return "" }
    resp, err := httpClient.Do(req)
    if err != nil { return "" }
    defer resp.Body.Close()
    if resp.StatusCode/100 != 2 { return "" }
    var tags struct {
        Models []struct{ Name string `json:"name"` }
    }
    if err := json.NewDecoder(resp.Body).Decode(&tags); err != nil { return "" }
    if len(tags.Models) == 0 { return "" }
    // Preferred chat-capable models
    prefs := []string{
        "llama3.1:8b-instruct", "llama3.1:latest", "llama3.1", "llama3:latest", "llama3",
        "qwen2.5:7b-instruct", "qwen2.5", "mistral:latest", "mistral", "phi3:latest", "phi3",
        "codellama:latest", "codellama",
    }
    // exact or prefix match
    for _, p := range prefs {
        for _, m := range tags.Models {
            if m.Name == p || strings.HasPrefix(m.Name, p+":") {
                return m.Name
            }
        }
    }
    // fallback to first available
    return tags.Models[0].Name
}

func (c *OllamaClient) StreamSummary(ctx context.Context, cfg Config, prompt string, out chan<- string) error {
    // Ensure server is reachable
    if !ollamaAvailable(1 * time.Second) {
        return errors.New("ollama server not reachable at " + ollamaHost())
    }
    model := c.pickModel(ctx, cfg)
    if model == "" {
        return errors.New("no local ollama models found; run `ollama pull llama3.1` or set --model")
    }
    // Build request
    body := map[string]any{
        "model": model,
        "messages": []map[string]string{{"role": "user", "content": prompt}},
        "stream": true,
        "options": map[string]any{"num_predict": max(1, cfg.MaxTokens)},
    }
    b, _ := json.Marshal(body)
    req, err := http.NewRequestWithContext(ctx, http.MethodPost, ollamaHost()+"/api/chat", bytes.NewReader(b))
    if err != nil { return err }
    req.Header.Set("content-type", "application/json")
    resp, err := httpClient.Do(req)
    if err != nil { return err }
    defer resp.Body.Close()
    if resp.StatusCode/100 != 2 {
        rb, _ := io.ReadAll(resp.Body)
        return fmt.Errorf("ollama http %d: %s", resp.StatusCode, string(rb))
    }
    // Stream NDJSON chunks
    reader := bufio.NewReader(resp.Body)
    for {
        line, err := reader.ReadBytes('\n')
        if len(line) > 0 {
            var chunk struct {
                Done    bool `json:"done"`
                Message struct {
                    Content string `json:"content"`
                } `json:"message"`
            }
            if jsonErr := json.Unmarshal(line, &chunk); jsonErr == nil {
                if chunk.Message.Content != "" {
                    out <- chunk.Message.Content
                }
                if chunk.Done {
                    break
                }
            }
        }
        if err != nil {
            if errors.Is(err, io.EOF) {
                break
            }
            return err
        }
    }
    return nil
}

// collapseWhitespace reduces all whitespace (incl. newlines) to single spaces and trims
func collapseWhitespace(s string) string {
    s = strings.ReplaceAll(s, "\n", " ")
    s = strings.ReplaceAll(s, "\r", " ")
    s = strings.ReplaceAll(s, "\t", " ")
    // squeeze multiple spaces
    var b strings.Builder
    b.Grow(len(s))
    prevSpace := false
    for i := 0; i < len(s); i++ {
        ch := s[i]
        isSpace := ch == ' ' || ch == '\t'
        if isSpace {
            if !prevSpace { b.WriteByte(' ') }
            prevSpace = true
        } else {
            b.WriteByte(ch)
            prevSpace = false
        }
    }
    out := strings.TrimSpace(b.String())
    return out
}

// ===== Minimal EventStream decoder for SSE-like streams

type eventStreamDecoder struct {
	r *bufio.Reader
}

func newEventStreamDecoder(r io.Reader) *eventStreamDecoder {
	return &eventStreamDecoder{r: bufio.NewReader(r)}
}

// Next returns event name (if any) and data bytes for one SSE event
func (d *eventStreamDecoder) Next() (string, []byte, error) {
	var event string
	var data []byte
	for {
		line, err := d.r.ReadString('\n')
		if err != nil {
			if errors.Is(err, io.EOF) && len(data) == 0 {
				return "", nil, io.EOF
			}
			if errors.Is(err, io.EOF) {
				break
			}
			return "", nil, err
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" { // end of event
			break
		}
		if strings.HasPrefix(line, "event:") {
			event = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			continue
		}
		if strings.HasPrefix(line, "data:") {
			chunk := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if strings.TrimSpace(chunk) == "[DONE]" {
				return "", nil, io.EOF
			}
			if len(data) > 0 {
				data = append(data, '\n')
			}
			data = append(data, []byte(chunk)...)
		}
	}
	return event, data, nil
}

// very small helper to pull a top-level JSON string field by name without full parse
// extractJSONField removed (replaced by minimal typed decoders)

func min(a, b int) int { if a < b { return a }; return b }
func max(a, b int) int { if a > b { return a }; return b }

// renderJSONOrFallback attempts to parse a strict JSON object following the schema
// {summary:string, bullets:[string,string,string], ascii_diagram:string|null}.
// On success it renders summary and bullets, and prints ascii diagram only if present.
// If parsing fails, it falls back to printing the raw content (respecting one-line mode already handled by caller).
// formatSummaryFromLLM returns a human-readable string from a JSON response
// per schema, or raw content if parsing fails. It respects --one-line.
func formatSummaryFromLLM(cfg Config, s string) string {
    // attempt to extract the JSON object boundaries to be resilient to minor pre/post text
    original := s
    start := strings.IndexByte(s, '{')
    end := strings.LastIndexByte(s, '}')
    if start >= 0 && end > start { s = s[start : end+1] }
    type jsonResp struct {
        Summary      string   `json:"summary"`
        Bullets      []string `json:"bullets"`
        ASCIIDiagram any      `json:"ascii_diagram"`
    }
    // If user explicitly wants JSON, just return the JSON blob as-is
    if strings.EqualFold(cfg.Format, "json") {
        return s
    }
    var r jsonResp
    if err := json.Unmarshal([]byte(s), &r); err != nil {
        // Fallback: raw content
        if cfg.OneLine {
            line := collapseWhitespace(original)
            if len(line) > 300 { line = line[:300] }
            return line
        }
        return original
    }
    var b strings.Builder
    if r.Summary != "" {
        b.WriteString(strings.TrimSpace(r.Summary))
        b.WriteByte('\n')
    }
    for i := 0; i < len(r.Bullets) && i < 3; i++ {
        if strings.TrimSpace(r.Bullets[i]) == "" { continue }
        b.WriteString("- ")
        b.WriteString(strings.TrimSpace(r.Bullets[i]))
        b.WriteByte('\n')
    }
    var diagram string
    switch v := r.ASCIIDiagram.(type) {
    case string:
        diagram = strings.TrimRight(v, "\r\n")
    }
    if diagram != "" {
        if b.Len() > 0 && !strings.HasSuffix(b.String(), "\n\n") {
            b.WriteByte('\n')
        }
        b.WriteString(diagram)
        if !strings.HasSuffix(b.String(), "\n") { b.WriteByte('\n') }
    }
    if cfg.OneLine {
        line := collapseWhitespace(b.String())
        if len(line) > 300 { line = line[:300] }
        return line
    }
    return b.String()
}

// renderJSONOrFallback prints with the standard filename prefix rules
// using the formatting rules above.
func renderJSONOrFallback(path string, cfg Config, s string) {
    // If JSON format explicitly requested, print only the JSON object
    if strings.EqualFold(cfg.Format, "json") {
        outMu.Lock()
        fmt.Println(formatSummaryFromLLM(cfg, s))
        outMu.Unlock()
        return
    }
    out := formatSummaryFromLLM(cfg, s)
    if cfg.OneLine {
        prefix := filePrefix(path, cfg)
        outMu.Lock()
        out = colorizeLine(out, cfg)
        if prefix == "" { fmt.Println(out) } else { fmt.Println(prefix + out) }
        outMu.Unlock()
        return
    }
    poutSinglePrefix(path, cfg, out)
}

// printAfterPrefix prints multi-line content continuing after an already-printed
// file prefix on the current line. Applies line-wise colorization; no extra prefix.
func printAfterPrefix(path string, cfg Config, s string) {
    outMu.Lock()
    defer outMu.Unlock()
    scanner := bufio.NewScanner(strings.NewReader(s))
    scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
    first := true
    // compute visible prefix width for alignment when prefix is enabled
    rawPrefix := ""
    if !cfg.NoPrefix {
        rawPrefix = fmt.Sprintf("[%s] ", filepath.Base(path))
    }
    for scanner.Scan() {
        line := colorizeLine(scanner.Text(), cfg)
        if first {
            // continue current line
            fmt.Println(line)
            first = false
        } else {
            // indent second..nth lines to align under the first token after prefix
            if rawPrefix != "" {
                fmt.Println(strings.Repeat(" ", utf8.RuneCountInString(rawPrefix)) + line)
            } else {
                fmt.Println(line)
            }
        }
    }
}

// ===== CLI: Help, Status, MCP =====

func exeName() string {
    if len(os.Args) == 0 { return "llmcat" }
    base := filepath.Base(os.Args[0])
    if base == "" { return "llmcat" }
    return base
}

func printUsage() {
    name := exeName()
    fmt.Fprintf(os.Stdout, "Usage: %s [flags] [files...]|-\n", name)
    fmt.Fprintln(os.Stdout)
    // Top 3 popular examples
    fmt.Fprintln(os.Stdout, "Examples (popular):")
    fmt.Fprintf(os.Stdout, "  1) Summarize a codebase glob\n     %s **/*.{go,ts,py}\n", name)
    fmt.Fprintf(os.Stdout, "  2) Pipe from stdin (e.g. git diff)\n     git diff -U0 | %s --one-line\n", name)
    fmt.Fprintf(os.Stdout, "  3) Auto-pick available provider and be concise\n     %s --provider auto --preset concise README.md\n", name)
    fmt.Fprintln(os.Stdout)

    fmt.Fprintln(os.Stdout, "More:")
    fmt.Fprintf(os.Stdout, "  Discover and cache API keys\n     %s -x\n", name)
    fmt.Fprintf(os.Stdout, "  Force provider/model\n     %s --provider openai --model gpt-4o-mini main.go\n", name)
    fmt.Fprintf(os.Stdout, "  JSON output (raw structured LLM result)\n     %s --format json file.py\n", name)
    fmt.Fprintln(os.Stdout)

    // MCP quick help
    fmt.Fprintln(os.Stdout, "MCP management:")
    fmt.Fprintf(os.Stdout, "  Add local stdio server\n     %s mcp add myserver -- python server.py --port 8080\n", name)
    fmt.Fprintf(os.Stdout, "  Add remote SSE server\n     %s mcp add --transport sse linear https://mcp.linear.app/sse\n", name)
    fmt.Fprintf(os.Stdout, "  Add remote HTTP server\n     %s mcp add --transport http notion https://mcp.notion.com/mcp\n", name)
    fmt.Fprintf(os.Stdout, "  List/get/remove/ping/call\n     %s mcp list\n     %s mcp get notion\n     %s mcp remove notion\n     %s mcp ping notion\n     %s mcp call notion summarize --json '{\"text\":\"hello\"}'\n", name, name, name, name, name)
    fmt.Fprintln(os.Stdout)

    fmt.Fprintln(os.Stdout, "Status:")
    fmt.Fprintf(os.Stdout, "  %s status\n", name)
    fmt.Fprintln(os.Stdout)

    // Flags summary
    fmt.Fprintln(os.Stdout, "Flags:")
    flag.CommandLine.PrintDefaults()
}

func runStatus(cfg Config, configPath string) {
    // Provider availability
    openAI := os.Getenv("OPENAI_API_KEY") != ""
    anthropic := (os.Getenv("ANTHROPIC_API_KEY") != "" || os.Getenv("CLAUDE_API_KEY") != "")
    google := os.Getenv("GOOGLE_API_KEY") != ""
    ollama := ollamaAvailable(800 * time.Millisecond)

    // Planned provider order
    var order []string
    if provs, err := selectProviders(cfg); err == nil {
        for _, p := range provs {
            switch p.(type) {
            case *OllamaClient:
                order = append(order, "ollama")
            case *OpenAIClient:
                order = append(order, "openai")
            case *AnthropicClient:
                order = append(order, "anthropic")
            case *GoogleClient:
                order = append(order, "google")
            default:
                order = append(order, "unknown")
            }
        }
    }

    // Print
    fmt.Println("llmcat status:")
    fmt.Printf("- provider: %s\n", string(cfg.Provider))
    if cfg.Model != "" {
        fmt.Printf("- model: %s\n", cfg.Model)
    } else {
        fmt.Printf("- model: (default per provider)\n")
    }
    fmt.Printf("- format: %s\n", cfg.Format)
    fmt.Printf("- concurrency: %d\n", cfg.Concurrency)
    fmt.Printf("- timeout_sec: %d\n", cfg.TimeoutSec)
    fmt.Printf("- max_tokens: %d\n", cfg.MaxTokens)
    if configPath != "" { fmt.Printf("- config: %s\n", configPath) }
    fmt.Println("- keys:")
    fmt.Printf("  * OPENAI_API_KEY: %v\n", openAI)
    fmt.Printf("  * ANTHROPIC_API_KEY/CLAUDE_API_KEY: %v\n", anthropic)
    fmt.Printf("  * GOOGLE_API_KEY: %v\n", google)
    fmt.Printf("  * ollama_available: %v\n", ollama)
    if len(order) > 0 {
        fmt.Printf("- provider_order: %s\n", strings.Join(order, ", "))
    }
}

// ===== MCP config + commands

type MCPServerConfig struct {
    Name      string            `json:"name"`
    Transport string            `json:"transport"` // stdio|sse|http
    Command   []string          `json:"command,omitempty"`
    Env       map[string]string `json:"env,omitempty"`
    URL       string            `json:"url,omitempty"`
    Headers   map[string]string `json:"headers,omitempty"`
    AddedAt   time.Time         `json:"added_at"`
}

type MCPConfig struct {
    Servers map[string]MCPServerConfig `json:"servers"`
}

func mcpConfigFilePath() (string, error) {
    d, err := configBaseDir()
    if err != nil { return "", err }
    return filepath.Join(d, "mcp.json"), nil
}

func loadMCP() (MCPConfig, error) {
    var mc MCPConfig
    mc.Servers = map[string]MCPServerConfig{}
    path, err := mcpConfigFilePath()
    if err != nil { return mc, err }
    b, err := os.ReadFile(path)
    if err != nil {
        if errors.Is(err, os.ErrNotExist) { return mc, nil }
        return mc, err
    }
    if err := json.Unmarshal(b, &mc); err != nil { return mc, err }
    if mc.Servers == nil { mc.Servers = map[string]MCPServerConfig{} }
    return mc, nil
}

func saveMCP(mc MCPConfig) error {
    path, err := mcpConfigFilePath()
    if err != nil { return err }
    b, err := json.MarshalIndent(mc, "", "  ")
    if err != nil { return err }
    return os.WriteFile(path, b, 0o600)
}

func runMCP(args []string) error {
    if len(args) == 0 {
        return fmt.Errorf("usage: %s mcp [add|list|get|remove] ...", exeName())
    }
    cmd := args[0]
    rest := args[1:]
    switch cmd {
    case "add":
        return mcpAdd(rest)
    case "list":
        return mcpList()
    case "get":
        if len(rest) != 1 { return fmt.Errorf("usage: %s mcp get <name>", exeName()) }
        return mcpGet(rest[0])
    case "remove":
        if len(rest) != 1 { return fmt.Errorf("usage: %s mcp remove <name>", exeName()) }
        return mcpRemove(rest[0])
    case "ping":
        if len(rest) != 1 { return fmt.Errorf("usage: %s mcp ping <name>", exeName()) }
        return mcpPing(rest[0])
    case "call":
        // usage: llmcat mcp call <name> <tool> [--json '{...}']
        if len(rest) < 2 { return fmt.Errorf("usage: %s mcp call <name> <tool> [--json '{...}']", exeName()) }
        name := rest[0]
        tool := rest[1]
        argsJSON := "{}"
        for i := 2; i < len(rest); i++ {
            a := rest[i]
            if a == "--json" {
                if i+1 >= len(rest) { return fmt.Errorf("--json requires value") }
                i++
                argsJSON = rest[i]
            }
        }
        var argsMap map[string]any
        if err := json.Unmarshal([]byte(argsJSON), &argsMap); err != nil {
            return fmt.Errorf("invalid --json: %w", err)
        }
        return mcpCall(name, tool, argsMap)
    default:
        return fmt.Errorf("unknown mcp subcommand: %s", cmd)
    }
}

func mcpAdd(args []string) error {
    // Parse flags before command separator "--"
    transport := "stdio"
    env := map[string]string{}
    headers := map[string]string{}
    // We allow flags in any order before "--"
    i := 0
    // helper to pop arg i
    pop := func() string { v := args[i]; i++; return v }
    // First collect any global flags like --transport, --env, --header
    positional := []string{}
    for i < len(args) {
        a := args[i]
        if a == "--" { break }
        if strings.HasPrefix(a, "--transport") {
            if a == "--transport" {
                if i+1 >= len(args) { return fmt.Errorf("--transport requires value") }
                i++
                transport = args[i]
                i++
                continue
            }
            parts := strings.SplitN(a, "=", 2)
            if len(parts) != 2 || parts[1] == "" { return fmt.Errorf("--transport requires value") }
            transport = parts[1]
            i++
            continue
        }
        if strings.HasPrefix(a, "--env") {
            val := ""
            if a == "--env" {
                if i+1 >= len(args) { return fmt.Errorf("--env requires KEY=VALUE") }
                i++
                val = args[i]
                i++
            } else {
                parts := strings.SplitN(a, "=", 2)
                if len(parts) != 2 || parts[1] == "" { return fmt.Errorf("--env requires KEY=VALUE") }
                val = parts[1]
                i++
            }
            kv := strings.SplitN(val, "=", 2)
            if len(kv) != 2 { return fmt.Errorf("--env requires KEY=VALUE") }
            env[kv[0]] = kv[1]
            continue
        }
        if strings.HasPrefix(a, "--header") {
            val := ""
            if a == "--header" {
                if i+1 >= len(args) { return fmt.Errorf("--header requires 'Key: Value'") }
                i++
                val = args[i]
                i++
            } else {
                parts := strings.SplitN(a, "=", 2)
                if len(parts) != 2 || parts[1] == "" { return fmt.Errorf("--header requires 'Key: Value'") }
                val = parts[1]
                i++
            }
            // split on first ':'
            hv := strings.SplitN(val, ":", 2)
            if len(hv) != 2 { return fmt.Errorf("--header requires 'Key: Value'") }
            key := strings.TrimSpace(hv[0])
            value := strings.TrimSpace(hv[1])
            if key == "" { return fmt.Errorf("--header key is empty") }
            headers[key] = value
            continue
        }
        // positional
        positional = append(positional, pop())
    }
    // Now handle according to transport
    switch strings.ToLower(transport) {
    case "stdio":
        // Expect: add <name> -- <command> [args...]
        if len(positional) < 1 { return fmt.Errorf("usage: %s mcp add <name> [--env KEY=VALUE] -- <command> [args...]", exeName()) }
        name := positional[0]
        // find "--"
        if i >= len(args) || args[i] != "--" {
            return fmt.Errorf("missing '--' before stdio command; everything after -- is the command to run")
        }
        i++ // skip --
        if i >= len(args) { return fmt.Errorf("missing stdio command after '--'") }
        cmdline := args[i:]
        mc, err := loadMCP()
        if err != nil { return err }
        if mc.Servers == nil { mc.Servers = map[string]MCPServerConfig{} }
        mc.Servers[name] = MCPServerConfig{
            Name: name,
            Transport: "stdio",
            Command: append([]string(nil), cmdline...),
            Env: env,
            Headers: nil,
            URL: "",
            AddedAt: time.Now(),
        }
        if err := saveMCP(mc); err != nil { return err }
        fmt.Printf("mcp: added stdio server '%s'\n", name)
        return nil
    case "sse", "http":
        // Expect: add --transport sse|http <name> <url>
        if len(positional) != 2 { return fmt.Errorf("usage: %s mcp add --transport %s <name> <url> [--header 'K: V']", exeName(), transport) }
        name := positional[0]
        url := positional[1]
        mc, err := loadMCP()
        if err != nil { return err }
        if mc.Servers == nil { mc.Servers = map[string]MCPServerConfig{} }
        mc.Servers[name] = MCPServerConfig{
            Name: name,
            Transport: strings.ToLower(transport),
            URL: url,
            Headers: headers,
            Env: env,
            AddedAt: time.Now(),
        }
        if err := saveMCP(mc); err != nil { return err }
        fmt.Printf("mcp: added %s server '%s' -> %s\n", strings.ToLower(transport), name, url)
        return nil
    default:
        return fmt.Errorf("unknown transport: %s (use stdio|sse|http)", transport)
    }
}

func mcpList() error {
    mc, err := loadMCP()
    if err != nil { return err }
    if len(mc.Servers) == 0 {
        fmt.Println("no MCP servers configured")
        return nil
    }
    // stable order
    names := make([]string, 0, len(mc.Servers))
    for n := range mc.Servers { names = append(names, n) }
    sort.Strings(names)
    for _, n := range names {
        s := mc.Servers[n]
        switch s.Transport {
        case "stdio":
            fmt.Printf("- %s: stdio -> %s\n", s.Name, strings.Join(s.Command, " "))
        case "sse", "http":
            fmt.Printf("- %s: %s -> %s\n", s.Name, s.Transport, s.URL)
        default:
            fmt.Printf("- %s: %s\n", s.Name, s.Transport)
        }
    }
    return nil
}

func mcpGet(name string) error {
    mc, err := loadMCP()
    if err != nil { return err }
    s, ok := mc.Servers[name]
    if !ok { return fmt.Errorf("not found: %s", name) }
    b, _ := json.MarshalIndent(s, "", "  ")
    fmt.Println(string(b))
    return nil
}

func mcpRemove(name string) error {
    mc, err := loadMCP()
    if err != nil { return err }
    if _, ok := mc.Servers[name]; !ok { return fmt.Errorf("not found: %s", name) }
    delete(mc.Servers, name)
    if err := saveMCP(mc); err != nil { return err }
    fmt.Printf("mcp: removed '%s'\n", name)
    return nil
}

// Launch and talk to configured MCP server
func mcpPing(name string) error {
    mc, err := loadMCP()
    if err != nil { return err }
    s, ok := mc.Servers[name]
    if !ok { return fmt.Errorf("not found: %s", name) }
    switch s.Transport {
    case "stdio":
        cli, err := startMCPStdioProcess(s)
        if err != nil { return err }
        defer cli.Close()
        if err := cli.Initialize(); err != nil { return err }
        tools, err := cli.ListTools()
        if err != nil { return err }
        fmt.Printf("ok: %s (%d tools)\n", name, len(tools))
        for _, t := range tools { fmt.Printf("- %s: %s\n", t.Name, t.Description) }
        return nil
    default:
        return fmt.Errorf("ping for transport %s not implemented yet", s.Transport)
    }
}

func mcpCall(name, tool string, args map[string]any) error {
    mc, err := loadMCP()
    if err != nil { return err }
    s, ok := mc.Servers[name]
    if !ok { return fmt.Errorf("not found: %s", name) }
    switch s.Transport {
    case "stdio":
        cli, err := startMCPStdioProcess(s)
        if err != nil { return err }
        defer cli.Close()
        if err := cli.Initialize(); err != nil { return err }
        res, err := cli.CallTool(tool, args)
        if err != nil { return err }
        // Print as pretty JSON
        b, _ := json.MarshalIndent(res, "", "  ")
        fmt.Println(string(b))
        return nil
    default:
        return fmt.Errorf("call for transport %s not implemented yet", s.Transport)
    }
}

type mcpStdioClient struct {
    cmd *exec.Cmd
    stdin io.WriteCloser
    stdout *bufio.Reader
}

func startMCPStdioProcess(s MCPServerConfig) (*mcpStdioClient, error) {
    if len(s.Command) == 0 { return nil, fmt.Errorf("empty command for %s", s.Name) }
    c := s.Command[0]
    args := []string{}
    if len(s.Command) > 1 { args = s.Command[1:] }
    proc := exec.Command(c, args...)
    // build environment
    env := os.Environ()
    for k, v := range s.Env {
        env = append(env, fmt.Sprintf("%s=%s", k, v))
    }
    proc.Env = env
    stdin, err := proc.StdinPipe()
    if err != nil { return nil, err }
    stdout, err := proc.StdoutPipe()
    if err != nil { return nil, err }
    proc.Stderr = os.Stderr
    if err := proc.Start(); err != nil { return nil, err }
    return &mcpStdioClient{cmd: proc, stdin: stdin, stdout: bufio.NewReader(stdout)}, nil
}

func (c *mcpStdioClient) Close() error {
    if c.stdin != nil { _ = c.stdin.Close() }
    if c.cmd != nil && c.cmd.Process != nil {
        // best-effort terminate
        _ = c.cmd.Process.Kill()
        _, _ = c.cmd.Process.Wait()
    }
    return nil
}

func (c *mcpStdioClient) rpc(method string, params any) (json.RawMessage, *jsonrpcError, error) {
    req := jsonrpcRequest{JSONRPC: "2.0", ID: 1, Method: method}
    if params != nil {
        b, err := json.Marshal(params)
        if err != nil { return nil, nil, err }
        req.Params = b
    }
    b, err := json.Marshal(req)
    if err != nil { return nil, nil, err }
    if err := writeMCPFrame(c.stdin, b); err != nil { return nil, nil, err }
    payload, err := readMCPFrame(c.stdout)
    if err != nil { return nil, nil, err }
    var resp jsonrpcResponse
    if err := json.Unmarshal(payload, &resp); err != nil { return nil, nil, err }
    if resp.Error != nil { return nil, resp.Error, nil }
    // marshal back to RawMessage
    rb, _ := json.Marshal(resp.Result)
    return json.RawMessage(rb), nil, nil
}

func (c *mcpStdioClient) Initialize() error {
    _, jerr, err := c.rpc("initialize", map[string]any{})
    if err != nil { return err }
    if jerr != nil { return fmt.Errorf("rpc error %d: %s", jerr.Code, jerr.Message) }
    // send initialized notification? Not required
    return nil
}

type ToolInfo struct {
    Name        string `json:"name"`
    Description string `json:"description"`
}

func (c *mcpStdioClient) ListTools() ([]ToolInfo, error) {
    res, jerr, err := c.rpc("tools/list", map[string]any{})
    if err != nil { return nil, err }
    if jerr != nil { return nil, fmt.Errorf("rpc error %d: %s", jerr.Code, jerr.Message) }
    var out struct{ Tools []ToolInfo `json:"tools"` }
    if err := json.Unmarshal(res, &out); err != nil { return nil, err }
    return out.Tools, nil
}

func (c *mcpStdioClient) CallTool(name string, args map[string]any) (map[string]any, error) {
    if args == nil { args = map[string]any{} }
    res, jerr, err := c.rpc("tools/call", map[string]any{"name": name, "arguments": args})
    if err != nil { return nil, err }
    if jerr != nil { return nil, fmt.Errorf("rpc error %d: %s", jerr.Code, jerr.Message) }
    var out map[string]any
    if err := json.Unmarshal(res, &out); err != nil { return nil, err }
    return out, nil
}



// ===== MCP stdio server implementation =====

func stdinIsPipe() bool {
    fi, err := os.Stdin.Stat()
    if err != nil { return false }
    return (fi.Mode() & os.ModeCharDevice) == 0
}

type jsonrpcRequest struct {
    JSONRPC string          `json:"jsonrpc"`
    ID      any             `json:"id,omitempty"`
    Method  string          `json:"method"`
    Params  json.RawMessage `json:"params,omitempty"`
}

type jsonrpcResponse struct {
    JSONRPC string         `json:"jsonrpc"`
    ID      any            `json:"id,omitempty"`
    Result  any            `json:"result,omitempty"`
    Error   *jsonrpcError  `json:"error,omitempty"`
}

type jsonrpcError struct {
    Code    int    `json:"code"`
    Message string `json:"message"`
    Data    any    `json:"data,omitempty"`
}

func writeMCPFrame(w io.Writer, payload []byte) error {
    // MCP uses Content-Length framing like LSP
    if _, err := fmt.Fprintf(w, "Content-Length: %d\r\n\r\n", len(payload)); err != nil { return err }
    _, err := w.Write(payload)
    return err
}

func readMCPFrame(r *bufio.Reader) ([]byte, error) {
    // Read headers until blank line
    contentLen := -1
    for {
        line, err := r.ReadString('\n')
        if err != nil { return nil, err }
        line = strings.TrimRight(line, "\r\n")
        if line == "" { break }
        if strings.HasPrefix(strings.ToLower(line), "content-length:") {
            parts := strings.SplitN(line, ":", 2)
            if len(parts) == 2 {
                v := strings.TrimSpace(parts[1])
                // parse int
                var n int
                _, perr := fmt.Sscanf(v, "%d", &n)
                if perr == nil { contentLen = n }
            }
        }
    }
    if contentLen < 0 { return nil, io.EOF }
    buf := make([]byte, contentLen)
    if _, err := io.ReadFull(r, buf); err != nil { return nil, err }
    return buf, nil
}

func runMCPServerStdio(cfg Config) {
    rd := bufio.NewReader(os.Stdin)
    for {
        payload, err := readMCPFrame(rd)
        if err != nil {
            if !errors.Is(err, io.EOF) {
                // best effort log to stderr
                fmt.Fprintln(os.Stderr, "mcp read error:", err)
            }
            return
        }
        var req jsonrpcRequest
        if err := json.Unmarshal(payload, &req); err != nil {
            _ = writeMCPFrame(os.Stdout, mustJSON(jsonrpcResponse{JSONRPC: "2.0", ID: nil, Error: &jsonrpcError{Code: -32700, Message: "parse error"}}))
            continue
        }
        // Dispatch
        switch req.Method {
        case "initialize":
            // respond with capabilities: tools
            res := map[string]any{
                "protocolVersion": "2024-11-05",
                "capabilities": map[string]any{
                    "tools": map[string]any{"listChanged": false},
                },
                "serverInfo": map[string]any{"name": "llmcat", "version": "dev"},
            }
            _ = writeMCPFrame(os.Stdout, mustJSON(jsonrpcResponse{JSONRPC: "2.0", ID: req.ID, Result: res}))
        case "initialized":
            // notification; no response needed per JSON-RPC, but some clients accept null
            // Do not reply to notifications (no id)
        case "tools/list":
            res := map[string]any{
                "tools": []map[string]any{
                    {
                        "name":        "summarize",
                        "description": "Summarize code or text with provider-optimized prompt",
                        "inputSchema": map[string]any{
                            "type": "object",
                            "properties": map[string]any{
                                "path":     map[string]any{"type": "string", "description": "filesystem path to read and summarize"},
                                "text":     map[string]any{"type": "string", "description": "raw text to summarize"},
                                "provider": map[string]any{"type": "string", "enum": []string{"openai","anthropic","google","ollama","auto"}},
                                "model":    map[string]any{"type": "string"},
                                "diagram":  map[string]any{"type": "boolean"},
                                "format":   map[string]any{"type": "string", "enum": []string{"text","json"}},
                            },
                            "additionalProperties": false,
                        },
                    },
                },
            }
            _ = writeMCPFrame(os.Stdout, mustJSON(jsonrpcResponse{JSONRPC: "2.0", ID: req.ID, Result: res}))
        case "tools/call":
            var p struct{
                Name string `json:"name"`
                Arguments map[string]any `json:"arguments"`
            }
            if err := json.Unmarshal(req.Params, &p); err != nil {
                _ = writeMCPFrame(os.Stdout, mustJSON(jsonrpcResponse{JSONRPC: "2.0", ID: req.ID, Error: &jsonrpcError{Code: -32602, Message: "invalid params"}}))
                continue
            }
            switch p.Name {
            case "summarize":
                // Prepare input
                var text, name, lang string
                // clone cfg so we can override
                scfg := cfg
                if v, ok := p.Arguments["provider"].(string); ok && v != "" {
                    scfg.Provider = Provider(strings.ToLower(v))
                }
                if v, ok := p.Arguments["model"].(string); ok && v != "" {
                    scfg.Model = v
                }
                if v, ok := p.Arguments["diagram"].(bool); ok { scfg.Diagram = v }
                if v, ok := p.Arguments["format"].(string); ok && v != "" { scfg.Format = v }

                if pathv, ok := p.Arguments["path"].(string); ok && pathv != "" {
                    b, err := os.ReadFile(pathv)
                    if err != nil {
                        _ = writeMCPFrame(os.Stdout, mustJSON(jsonrpcResponse{JSONRPC: "2.0", ID: req.ID, Error: &jsonrpcError{Code: -32000, Message: err.Error()}}))
                        continue
                    }
                    name = filepath.Base(pathv)
                    lang = languageFromPath(pathv)
                    text = compactTextBytes(b, scfg)
                } else if tv, ok := p.Arguments["text"].(string); ok && tv != "" {
                    name = "text"
                    lang = ""
                    if len(tv) > scfg.MaxChars*2 {
                        text = compactText(tv, scfg.MaxChars, scfg.HeadLines, scfg.TailLines, scfg.SymbolLines)
                    } else {
                        text = tv
                    }
                } else {
                    _ = writeMCPFrame(os.Stdout, mustJSON(jsonrpcResponse{JSONRPC: "2.0", ID: req.ID, Error: &jsonrpcError{Code: -32602, Message: "summarize requires 'path' or 'text'"}}))
                    continue
                }
                // Build prompt and run providers
                prompt := buildPrompt(scfg, name, lang, text)
                out, err := mcpSummarize(scfg, prompt)
                if err != nil {
                    _ = writeMCPFrame(os.Stdout, mustJSON(jsonrpcResponse{JSONRPC: "2.0", ID: req.ID, Error: &jsonrpcError{Code: -32001, Message: err.Error()}}))
                    continue
                }
                // Return as MCP content list
                res := map[string]any{
                    "content": []map[string]any{{"type": "text", "text": out}},
                }
                _ = writeMCPFrame(os.Stdout, mustJSON(jsonrpcResponse{JSONRPC: "2.0", ID: req.ID, Result: res}))
            default:
                _ = writeMCPFrame(os.Stdout, mustJSON(jsonrpcResponse{JSONRPC: "2.0", ID: req.ID, Error: &jsonrpcError{Code: -32601, Message: "unknown tool"}}))
            }
        default:
            // method not found
            _ = writeMCPFrame(os.Stdout, mustJSON(jsonrpcResponse{JSONRPC: "2.0", ID: req.ID, Error: &jsonrpcError{Code: -32601, Message: "method not found"}}))
        }
    }
}

func mustJSON(v any) []byte {
    b, _ := json.Marshal(v)
    return b
}

func mcpSummarize(cfg Config, prompt string) (string, error) {
    ctx, cancel := context.WithTimeout(context.Background(), time.Duration(cfg.TimeoutSec)*time.Second)
    defer cancel()
    providers, err := selectProviders(cfg)
    if err != nil || len(providers) == 0 {
        if err != nil { return "", err }
        return "", fmt.Errorf("no providers available")
    }
    var firstErr error
    for i, p := range providers {
        stream := make(chan string, 16)
        errCh := make(chan error, 1)
        go func(prov ProviderClient) {
            defer close(stream)
            if err := prov.StreamSummary(ctx, cfg, prompt, stream); err != nil {
                errCh <- err
            } else {
                errCh <- nil
            }
        }(p)
        var b strings.Builder
        had := false
        done := false
        for !done {
            select {
            case chunk, ok := <-stream:
                if !ok { done = true; break }
                if chunk != "" { had = true }
                b.WriteString(chunk)
            case err := <-errCh:
                if err != nil && !had {
                    if firstErr == nil { firstErr = err }
                    done = true
                    continue
                }
                done = true
            case <-ctx.Done():
                return "", ctx.Err()
            }
        }
        if had || i == len(providers)-1 {
            // Format with existing helper (respects cfg.Format and one-line)
            // For MCP, ignore one-line; always return full formatted text
            prevOne := cfg.OneLine
            cfg.OneLine = false
            out := formatSummaryFromLLM(cfg, b.String())
            cfg.OneLine = prevOne
            return out, nil
        }
    }
    if firstErr != nil { return "", firstErr }
    return "", fmt.Errorf("failed to summarize")
}
