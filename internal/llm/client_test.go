package llm

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

func TestNewOpenAIClient_URLNormalization(t *testing.T) {
	tests := []struct {
		name     string
		inputURL string
		wantURL  string
	}{
		{
			name:     "base URL without trailing slash",
			inputURL: "https://api.example.com/v1",
			wantURL:  "https://api.example.com/v1/chat/completions",
		},
		{
			name:     "base URL with trailing slash",
			inputURL: "https://api.example.com/v1/",
			wantURL:  "https://api.example.com/v1/chat/completions",
		},
		{
			name:     "full URL already has chat/completions",
			inputURL: "https://api.example.com/v1/chat/completions",
			wantURL:  "https://api.example.com/v1/chat/completions",
		},
		{
			name:     "full URL with trailing slash",
			inputURL: "https://api.example.com/v1/chat/completions/",
			wantURL:  "https://api.example.com/v1/chat/completions/",
		},
		{
			name:     "bare host",
			inputURL: "https://api.example.com",
			wantURL:  "https://api.example.com/chat/completions",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := NewOpenAIClient(ClientConfig{URL: tt.inputURL})
			if client.cfg.URL != tt.wantURL {
				t.Errorf("got URL %q, want %q", client.cfg.URL, tt.wantURL)
			}
		})
	}
}

func TestNewOpenAIClient_ExactEndpointDoesNotAppendChatCompletions(t *testing.T) {
	client := NewOpenAIClient(ClientConfig{
		URL:            "https://proxy.example.com/custom-openai",
		AutoAppendPath: boolPtr(false),
	})

	if client.cfg.URL != "https://proxy.example.com/custom-openai" {
		t.Errorf("got URL %q, want exact endpoint", client.cfg.URL)
	}
}

func TestNewAnthropicClient_URLNormalization(t *testing.T) {
	tests := []struct {
		name     string
		inputURL string
		wantURL  string
	}{
		{
			name:     "bare host",
			inputURL: "https://api.anthropic.com",
			wantURL:  "https://api.anthropic.com/v1/messages",
		},
		{
			name:     "bare host with trailing slash",
			inputURL: "https://api.anthropic.com/",
			wantURL:  "https://api.anthropic.com/v1/messages",
		},
		{
			name:     "full URL already has /v1/messages",
			inputURL: "https://api.anthropic.com/v1/messages",
			wantURL:  "https://api.anthropic.com/v1/messages",
		},
		{
			name:     "full URL with trailing slash",
			inputURL: "https://api.anthropic.com/v1/messages/",
			wantURL:  "https://api.anthropic.com/v1/messages/",
		},
		{
			name:     "custom proxy base URL",
			inputURL: "https://proxy.example.com/anthropic",
			wantURL:  "https://proxy.example.com/anthropic/v1/messages",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := NewAnthropicClient(ClientConfig{URL: tt.inputURL})
			if client.cfg.URL != tt.wantURL {
				t.Errorf("got URL %q, want %q", client.cfg.URL, tt.wantURL)
			}
		})
	}
}

func TestNewAnthropicClient_ExactEndpointDoesNotAppendMessagesPath(t *testing.T) {
	client := NewAnthropicClient(ClientConfig{
		URL:            "https://open.bigmodel.cn/api/anthropic",
		AutoAppendPath: boolPtr(false),
	})

	if client.cfg.URL != "https://open.bigmodel.cn/api/anthropic" {
		t.Errorf("got URL %q, want exact endpoint", client.cfg.URL)
	}
}

func TestRetryWithCtxIncludesLastErrorWhenContextExpires(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()

	err := retryWithCtx(ctx, func() error {
		return errors.New("request failed: dial tcp 127.0.0.1:443: connectex: No connection could be made")
	})

	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "context deadline exceeded") {
		t.Fatalf("expected context deadline in error, got %q", err.Error())
	}
	if !strings.Contains(err.Error(), "dial tcp 127.0.0.1:443") {
		t.Fatalf("expected last retry error in error, got %q", err.Error())
	}
}
