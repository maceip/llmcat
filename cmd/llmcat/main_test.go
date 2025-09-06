package main

import (
    "os"
    "strings"
    "testing"
)

func TestChooseGrayPalette256(t *testing.T) {
    oldTERM := os.Getenv("TERM")
    oldCT := os.Getenv("COLORTERM")
    t.Cleanup(func() { os.Setenv("TERM", oldTERM); os.Setenv("COLORTERM", oldCT) })
    _ = os.Setenv("TERM", "xterm-256color")
    _ = os.Setenv("COLORTERM", "")
    leader, tail1, tail2, bg := chooseGrayPalette()
    if !strings.Contains(leader, "38;5;231") || !strings.Contains(tail1, "38;5;250") || !strings.Contains(tail2, "38;5;246") || !strings.Contains(bg, "38;5;240") {
        t.Fatalf("unexpected 256 palette: %q %q %q %q", leader, tail1, tail2, bg)
    }
}

func TestChooseGrayPalette8Color(t *testing.T) {
    oldTERM := os.Getenv("TERM")
    oldCT := os.Getenv("COLORTERM")
    t.Cleanup(func() { os.Setenv("TERM", oldTERM); os.Setenv("COLORTERM", oldCT) })
    _ = os.Setenv("TERM", "xterm")
    _ = os.Setenv("COLORTERM", "")
    leader, tail1, tail2, bg := chooseGrayPalette()
    if leader != "\x1b[97m" || tail1 != "\x1b[37m" || tail2 != "\x1b[2;37m" || bg != "\x1b[90m" {
        t.Fatalf("unexpected 8-color palette: %q %q %q %q", leader, tail1, tail2, bg)
    }
}
