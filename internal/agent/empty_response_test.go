package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/open-code-review/open-code-review/internal/config/template"
	"github.com/open-code-review/open-code-review/internal/llm"
	"github.com/open-code-review/open-code-review/internal/session"
)

type emptyResponseLLM struct {
	calls int
}

func (c *emptyResponseLLM) Completions(req llm.ChatRequest) (*llm.ChatResponse, error) {
	return c.CompletionsWithCtx(context.Background(), req)
}

func (c *emptyResponseLLM) CompletionsWithCtx(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
	c.calls++
	return &llm.ChatResponse{
		Choices: []llm.Choice{{
			Message: llm.ResponseMessage{Role: "assistant"},
		}},
		Usage: &llm.UsageInfo{PromptTokens: 10, TotalTokens: 10},
	}, nil
}

func (c *emptyResponseLLM) StreamCompletion(req llm.ChatRequest, cb func(chunk []byte) error) error {
	return nil
}

func TestPerformLlmCodeReviewFailsAfterEmptyResponseRetries(t *testing.T) {
	client := &emptyResponseLLM{}
	agent := New(Args{
		RepoDir: ".",
		Template: template.Template{
			MaxToolRequestTimes: 10,
			MaxTokens:           4096,
		},
		LLMClient: client,
		Session:   session.New("", "", "test-model", session.SessionOptions{}),
	})

	oldSleep := sleepEmptyLLMResponseRetry
	sleepEmptyLLMResponseRetry = func(context.Context, int) error { return nil }
	defer func() { sleepEmptyLLMResponseRetry = oldSleep }()

	err := agent.performLlmCodeReview(context.Background(), []llm.Message{
		llm.NewTextMessage("user", "review this diff"),
	}, "file.go")

	if err == nil {
		t.Fatal("expected error")
	}
	if client.calls != maxEmptyLLMResponseRetries {
		t.Fatalf("expected %d retries, got %d", maxEmptyLLMResponseRetries, client.calls)
	}
	if !strings.Contains(err.Error(), "empty LLM response") {
		t.Fatalf("expected empty response error, got %q", err.Error())
	}
	if !strings.Contains(err.Error(), "provider may not support") {
		t.Fatalf("expected provider compatibility guidance, got %q", err.Error())
	}
}

func TestFirstWarningOfType(t *testing.T) {
	warnings := []AgentWarning{
		{Type: "token_threshold_exceeded", File: "large.go", Message: "too large"},
		{Type: "subtask_error", File: "file.go", Message: "empty LLM response"},
		{Type: "subtask_error", File: "other.go", Message: "another error"},
	}

	warning := firstWarningOfType(warnings, "subtask_error")
	if warning == nil {
		t.Fatal("expected warning")
	}
	if warning.File != "file.go" || warning.Message != "empty LLM response" {
		t.Fatalf("unexpected warning: %#v", warning)
	}

	if warning := firstWarningOfType(warnings, "missing"); warning != nil {
		t.Fatalf("expected nil for missing warning type, got %#v", warning)
	}
}
