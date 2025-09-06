package main

import (
    "bufio"
    "encoding/json"
    "flag"
    "fmt"
    "io"
    "os"
    "os/exec"
    "strings"
)

// Minimal MCP JSON-RPC framing helpers
func writeFrame(w io.Writer, payload []byte) error {
    if _, err := fmt.Fprintf(w, "Content-Length: %d\r\n\r\n", len(payload)); err != nil { return err }
    _, err := w.Write(payload)
    return err
}

func readFrame(r *bufio.Reader) ([]byte, error) {
    contentLen := -1
    for {
        line, err := r.ReadString('\n')
        if err != nil { return nil, err }
        line = strings.TrimRight(line, "\r\n")
        if line == "" { break }
        if strings.HasPrefix(strings.ToLower(line), "content-length:") {
            var n int
            fmt.Sscanf(strings.TrimSpace(strings.SplitN(line, ":", 2)[1]), "%d", &n)
            contentLen = n
        }
    }
    if contentLen < 0 { return nil, io.EOF }
    buf := make([]byte, contentLen)
    if _, err := io.ReadFull(r, buf); err != nil { return nil, err }
    return buf, nil
}

type req struct {
    JSONRPC string          `json:"jsonrpc"`
    ID      int             `json:"id,omitempty"`
    Method  string          `json:"method"`
    Params  json.RawMessage `json:"params,omitempty"`
}

type resp struct {
    JSONRPC string         `json:"jsonrpc"`
    ID      int            `json:"id,omitempty"`
    Result  any            `json:"result,omitempty"`
    Error   *struct{ Code int `json:"code"`; Message string `json:"message"` } `json:"error,omitempty"`
}

func main() {
    bin := flag.String("bin", "./llmcat", "path to llmcat binary")
    flag.Parse()
    cmd := exec.Command(*bin)
    stdin, _ := cmd.StdinPipe()
    stdout, _ := cmd.StdoutPipe()
    cmd.Stderr = os.Stderr
    if err := cmd.Start(); err != nil {
        fmt.Fprintln(os.Stderr, "start error:", err)
        os.Exit(1)
    }
    rd := bufio.NewReader(stdout)
    // initialize
    b, _ := json.Marshal(req{JSONRPC: "2.0", ID: 1, Method: "initialize", Params: json.RawMessage(`{}`)})
    _ = writeFrame(stdin, b)
    if payload, err := readFrame(rd); err == nil {
        var r resp; _ = json.Unmarshal(payload, &r)
        fmt.Println("initialize ok")
    } else { fmt.Fprintln(os.Stderr, "initialize read error:", err); os.Exit(1) }
    // tools/list
    b, _ = json.Marshal(req{JSONRPC: "2.0", ID: 2, Method: "tools/list", Params: json.RawMessage(`{}`)})
    _ = writeFrame(stdin, b)
    if payload, err := readFrame(rd); err == nil {
        var r resp; _ = json.Unmarshal(payload, &r)
        jb, _ := json.MarshalIndent(r.Result, "", "  ")
        fmt.Println(string(jb))
    } else { fmt.Fprintln(os.Stderr, "tools/list read error:", err); os.Exit(1) }
    _ = stdin.Close()
    _ = cmd.Process.Kill()
}

