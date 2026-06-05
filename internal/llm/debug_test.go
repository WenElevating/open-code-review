package llm

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/open-code-review/open-code-review/internal/stdout"
)

func TestOpenAIClientDebugLogsRequestAndToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if auth := r.Header.Get("Authorization"); auth != "Bearer secret-token" {
			t.Fatalf("Authorization header = %q", auth)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "chatcmpl-debug",
			"model": "debug-model",
			"choices": [{
				"message": {
					"role": "assistant",
					"content": "",
					"tool_calls": [{
						"id": "call_1",
						"type": "function",
						"function": {
							"name": "task_done",
							"arguments": "{\"ok\":true}"
						}
					}]
				},
				"finish_reason": "tool_calls"
			}]
		}`))
	}))
	defer server.Close()

	var out bytes.Buffer
	restore := stdout.SetWriterForTest(&out)
	defer restore()

	client := NewOpenAIClient(ClientConfig{
		URL:            server.URL,
		APIKey:         "secret-token",
		Model:          "debug-model",
		AutoAppendPath: boolPtr(false),
		Debug:          true,
	})

	_, err := client.Completions(ChatRequest{
		Messages: []Message{NewTextMessage("user", "hello")},
		Tools: []ToolDef{{
			Type: "function",
			Function: FunctionDef{
				Name:        "task_done",
				Description: "finish",
				Parameters:  map[string]any{"type": "object"},
			},
		}},
	})
	if err != nil {
		t.Fatalf("Completions returned error: %v", err)
	}

	log := out.String()
	for _, want := range []string{
		"[llm-debug] request openai POST " + server.URL,
		`"model": "debug-model"`,
		`"tools"`,
		`[llm-debug] response openai status=200`,
		`"name": "task_done"`,
		`"arguments": "{\"ok\":true}"`,
	} {
		if !strings.Contains(log, want) {
			t.Fatalf("debug log missing %q:\n%s", want, log)
		}
	}
	if strings.Contains(log, "secret-token") || strings.Contains(log, "Authorization") {
		t.Fatalf("debug log leaked secret header/token:\n%s", log)
	}
}
