package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type phaseResult struct {
	name               string
	totalRequests      int
	successfulRequests int64
	failedRequests     int64
	latencies          []time.Duration
	elapsed            time.Duration
}

func (p phaseResult) avgLatencyMs() float64 {
	if p.successfulRequests <= 0 {
		return 0
	}
	var total time.Duration
	for _, d := range p.latencies {
		total += d
	}
	return float64(total) / float64(p.successfulRequests) / float64(time.Millisecond)
}

func (p phaseResult) throughputRPS() float64 {
	if p.elapsed.Seconds() <= 0 {
		return 0
	}
	return float64(p.successfulRequests) / p.elapsed.Seconds()
}

func printPhase(p phaseResult) {
	fmt.Printf("\n=== %s Results:\n", p.name)
	fmt.Printf("   Total: %d requests (%d success, %d failed)\n", p.totalRequests, p.successfulRequests, p.failedRequests)
	fmt.Printf("   Elapsed: %.2fs\n", p.elapsed.Seconds())
	fmt.Printf("   Avg latency: %.2fms\n", p.avgLatencyMs())
	fmt.Printf("   Throughput: %.2f req/s\n", p.throughputRPS())
}

func parseHeaderArgs(headerParts []string) map[string]string {
	out := make(map[string]string)
	for _, part := range headerParts {
		var key, value string
		if strings.Contains(part, ":") {
			parts := strings.SplitN(part, ":", 2)
			key = strings.TrimSpace(parts[0])
			value = strings.TrimSpace(parts[1])
		} else if strings.Contains(part, "=") {
			parts := strings.SplitN(part, "=", 2)
			key = strings.TrimSpace(parts[0])
			value = strings.TrimSpace(parts[1])
		} else {
			continue
		}
		out[key] = value
	}
	return out
}

func mergeHTTPHeaders(cliHeaders []string, authToken string) http.Header {
	headers := parseHeaderArgs(cliHeaders)
	result := make(http.Header)
	for k, v := range headers {
		result.Set(k, v)
	}

	if authToken != "" {
		if result.Get("Authorization") == "" {
			result.Set("Authorization", "Bearer "+authToken)
		}
	}

	envAuth := os.Getenv("MCP_AUTHORIZATION")
	if envAuth == "" {
		envAuth = os.Getenv("AUTHORIZATION")
	}
	if envAuth != "" && result.Get("Authorization") == "" {
		result.Set("Authorization", strings.TrimSpace(envAuth))
	}

	envBearer := os.Getenv("MCP_AUTH_TOKEN")
	if envBearer == "" {
		envBearer = os.Getenv("MCP_BEARER_TOKEN")
	}
	if envBearer != "" && result.Get("Authorization") == "" {
		result.Set("Authorization", "Bearer "+strings.TrimSpace(envBearer))
	}

	return result
}

// multiString implements flag.Value for collecting multiple -H flags
type multiString struct {
	values []string
}

func (m *multiString) String() string {
	return strings.Join(m.values, ", ")
}

func (m *multiString) Set(value string) error {
	m.values = append(m.values, value)
	return nil
}

func main() {
	serverURL := flag.String("s", "http://localhost:8000/mcp", "MCP server URL")
	runs := flag.Int("r", 100, "Tool call runs per user (total cycles)")
	users := flag.Int("u", 1, "Number of concurrent clients (processes)")
	toolName := flag.String("t", "say_hello", "Tool name to call")
	argsStr := flag.String("a", "{}", "Tool arguments as JSON object")
	authToken := flag.String("auth-token", "", "Shortcut for Authorization: Bearer <token>")
	callsPerCycle := flag.Int("calls-per-cycle", 100, "Number of call_tool operations per cycle")

	// Collect -H flags using a custom Value type
	var headerList multiString
	flag.Var(&headerList, "H", "Extra HTTP header, e.g. -H \"Authorization: Bearer <token>\"")
	flag.Parse()

	if r := os.Getenv("RUNS"); r != "" {
		if n, err := strconv.Atoi(r); err == nil && n > 0 {
			*runs = n
		}
	}

	// Parse tool arguments
	var toolArguments map[string]interface{}
	argsJSON := strings.TrimSpace(*argsStr)
	if argsJSON != "" {
		if err := json.Unmarshal([]byte(argsJSON), &toolArguments); err != nil {
			fmt.Fprintf(os.Stderr, "Error: Invalid JSON in arguments: %v\n", err)
			os.Exit(1)
		}
	}

	httpHeaders := mergeHTTPHeaders(headerList.values, *authToken)
	baseHeaders := http.Header{
		"Content-Type": []string{"application/json; charset=utf-8"},
		"Accept":       []string{"application/json, application/x-ndjson, text/event-stream"},
	}
	for k, v := range httpHeaders {
		for _, val := range v {
			baseHeaders.Add(k, val)
		}
	}

	ctx := context.Background()
	client := &http.Client{Timeout: 30 * time.Second}

	url := *serverURL
	initReq := `{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"demo","version":"0.0.1"}}}`
	notifyReq := `{"jsonrpc": "2.0","method": "notifications/initialized"}`
	listReq := `{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}`
	callReq := fmt.Sprintf(`{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"%s","arguments":%s}}`, *toolName, *argsStr)

	fmt.Printf("🔌 MCP Streamable HTTP Benchmark\n")
	fmt.Printf("   Transport: Streamable HTTP\n")
	fmt.Printf("   Users: %d, Runs: %d, Calls per cycle: %d\n", *users, *runs, *callsPerCycle)
	fmt.Printf("   Server: %s\n", url)

	// Best-effort tool verification
	fmt.Printf("   Verifying tool '%s' exists...\n", *toolName)
	if err := verifyToolExists(ctx, client, url, baseHeaders, initReq, notifyReq, listReq, *toolName); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("   ✅ Tool '%s' found\n", *toolName)

	// Cyclic benchmark: each cycle is init -> list_tools -> N call_tool
	var cyclicTotalCycles int64
	var cyclicTotalCalls int64
	var cyclicInitSuccess, cyclicInitFail int64
	var cyclicListSuccess, cyclicListFail int64
	var cyclicCallSuccess, cyclicCallFail int64
	var cyclicInitLatencies, cyclicListLatencies, cyclicCallLatencies []time.Duration
	var cyclicMu sync.Mutex
	var cyclicElapsed time.Duration

	benchStart := time.Now()

	var wg sync.WaitGroup
	for i := 0; i < *users; i++ {
		wg.Add(1)
		go func(userIdx int) {
			defer wg.Done()

			var localInitLatencies, localListLatencies, localCallLatencies []time.Duration
			var localInitSuccess, localInitFail, localListSuccess, localListFail, localCallSuccess, localCallFail int64
			var localCycles int64

			for {
				currentTotalCalls := atomic.LoadInt64(&cyclicTotalCalls)
				if currentTotalCalls >= int64(*runs) {
					break
				}

				remainingCalls := int64(*runs) - currentTotalCalls
				callsThisCycle := int64(*callsPerCycle)
				if callsThisCycle > remainingCalls {
					callsThisCycle = remainingCalls
				}

				if callsThisCycle <= 0 {
					break
				}

				// Phase 1: init
				t0 := time.Now()
				sessionID, ok := doInit(ctx, client, url, baseHeaders, initReq, notifyReq)
				if ok {
					localInitLatencies = append(localInitLatencies, time.Since(t0))
					localInitSuccess++
				} else {
					localInitFail++
				}

				// Phase 2: list_tools
				if sessionID != "" {
					t0 = time.Now()
					userHeaders := baseHeaders.Clone()
					userHeaders.Set("Mcp-Session-Id", sessionID)
					ok = doListTools(ctx, client, url, userHeaders, listReq)
					if ok {
						localListLatencies = append(localListLatencies, time.Since(t0))
						localListSuccess++
					} else {
						localListFail++
					}
				}

				// Phase 3: call_tool N times
				if sessionID != "" {
					userHeaders := baseHeaders.Clone()
					userHeaders.Set("Mcp-Session-Id", sessionID)
					for j := int64(0); j < callsThisCycle; j++ {
						t0 = time.Now()
						ok = doCallTool(ctx, client, url, userHeaders, callReq)
						if ok {
							localCallLatencies = append(localCallLatencies, time.Since(t0))
							localCallSuccess++
						} else {
							localCallFail++
						}
					}
				}

				localCycles++
				atomic.AddInt64(&cyclicTotalCalls, callsThisCycle)
			}

			cyclicMu.Lock()
			cyclicInitLatencies = append(cyclicInitLatencies, localInitLatencies...)
			cyclicListLatencies = append(cyclicListLatencies, localListLatencies...)
			cyclicCallLatencies = append(cyclicCallLatencies, localCallLatencies...)
			cyclicInitSuccess += localInitSuccess
			cyclicInitFail += localInitFail
			cyclicListSuccess += localListSuccess
			cyclicListFail += localListFail
			cyclicCallSuccess += localCallSuccess
			cyclicCallFail += localCallFail
			cyclicTotalCycles += localCycles
			cyclicMu.Unlock()
		}(i)
	}
	wg.Wait()
	cyclicElapsed = time.Since(benchStart)

	fmt.Printf("\n=== Cyclic Benchmark Results:\n")
	fmt.Printf("   Total cycles: %d\n", cyclicTotalCycles)
	fmt.Printf("   Total operations: %d inits, %d lists, %d calls\n", cyclicInitSuccess+cyclicInitFail, cyclicListSuccess+cyclicListFail, cyclicCallSuccess+cyclicCallFail)
	fmt.Printf("   Elapsed: %.2fs\n", cyclicElapsed.Seconds())
	fmt.Println()
	fmt.Println("   Init stats:")
	fmt.Printf("      %d success, %d failed\n", cyclicInitSuccess, cyclicInitFail)
	if cyclicInitSuccess > 0 {
		var total time.Duration
		for _, d := range cyclicInitLatencies {
			total += d
		}
		avg := float64(total) / float64(cyclicInitSuccess) / float64(time.Millisecond)
		fmt.Printf("      Avg latency: %.2fms\n", avg)
		fmt.Printf("      Throughput: %.2f req/s\n", float64(cyclicInitSuccess)/cyclicElapsed.Seconds())
	}
	fmt.Println()
	fmt.Println("   Tool/list stats:")
	fmt.Printf("      %d success, %d failed\n", cyclicListSuccess, cyclicListFail)
	if cyclicListSuccess > 0 {
		var total time.Duration
		for _, d := range cyclicListLatencies {
			total += d
		}
		avg := float64(total) / float64(cyclicListSuccess) / float64(time.Millisecond)
		fmt.Printf("      Avg latency: %.2fms\n", avg)
		fmt.Printf("      Throughput: %.2f req/s\n", float64(cyclicListSuccess)/cyclicElapsed.Seconds())
	}
	fmt.Println()
	fmt.Println("   Tool/call stats:")
	fmt.Printf("      %d success, %d failed\n", cyclicCallSuccess, cyclicCallFail)
	if cyclicCallSuccess > 0 {
		var total time.Duration
		for _, d := range cyclicCallLatencies {
			total += d
		}
		avg := float64(total) / float64(cyclicCallSuccess) / float64(time.Millisecond)
		fmt.Printf("      Avg latency: %.2fms\n", avg)
		fmt.Printf("      Throughput: %.2f req/s\n", float64(cyclicCallSuccess)/cyclicElapsed.Seconds())
	}
}

func doInit(ctx context.Context, client *http.Client, url string, baseHeaders http.Header, initReq, notifyReq string) (string, bool) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBufferString(initReq))
	if err != nil {
		return "", false
	}
	req.Header = baseHeaders.Clone()
	resp, err := client.Do(req)
	if err != nil {
		return "", false
	}
	sessionID := resp.Header.Get("Mcp-Session-Id")
	_ = resp.Body.Close()
	if sessionID == "" {
		return "", false
	}

	req, err = http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBufferString(notifyReq))
	if err != nil {
		return "", false
	}
	req.Header = baseHeaders.Clone()
	req.Header.Set("Mcp-Session-Id", sessionID)
	resp, err = client.Do(req)
	if err != nil {
		return "", false
	}
	_ = resp.Body.Close()

	return sessionID, true
}

func doListTools(ctx context.Context, client *http.Client, url string, headers http.Header, listReq string) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBufferString(listReq))
	if err != nil {
		return false
	}
	req.Header = headers.Clone()
	resp, err := client.Do(req)
	if err != nil {
		return false
	}
	_ = resp.Body.Close()
	return true
}

func doCallTool(ctx context.Context, client *http.Client, url string, headers http.Header, callReq string) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBufferString(callReq))
	if err != nil {
		return false
	}
	req.Header = headers.Clone()
	resp, err := client.Do(req)
	if err != nil {
		return false
	}
	_ = resp.Body.Close()
	return true
}

func verifyToolExists(
	ctx context.Context,
	client *http.Client,
	url string,
	baseHeaders http.Header,
	initReq string,
	notifyReq string,
	listReq string,
	toolName string,
) error {
	sessionID, ok := doInit(ctx, client, url, baseHeaders, initReq, notifyReq)
	if !ok {
		return fmt.Errorf("tool verification: init failed")
	}

	userHeaders := baseHeaders.Clone()
	userHeaders.Set("Mcp-Session-Id", sessionID)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBufferString(listReq))
	if err != nil {
		return err
	}
	req.Header = userHeaders.Clone()
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	body, err := readSSE(resp.Body)
	if err != nil {
		return err
	}

	// Parse the JSON response to extract tool list
	var listResponse struct {
		Result struct {
			Tools []struct {
				Name string `json:"name"`
			} `json:"tools"`
		} `json:"result"`
	}

	if err := json.Unmarshal(body, &listResponse); err != nil {
		return fmt.Errorf("Tool '%s' not found (failed to parse response: %v)", toolName, err)
	}

	// Check if the requested tool exists
	for _, tool := range listResponse.Result.Tools {
		if tool.Name == toolName {
			return nil
		}
	}

	// Tool not found - list all available tools
	var availableTools []string
	for _, tool := range listResponse.Result.Tools {
		availableTools = append(availableTools, tool.Name)
	}

	if len(availableTools) == 0 {
		return fmt.Errorf("Tool '%s' not found. No tools available.", toolName)
	}

	return fmt.Errorf("Tool '%s' not found. Available tools: %s", toolName, strings.Join(availableTools, ", "))
}

func readSSE(r io.Reader) ([]byte, error) {
	var buf bytes.Buffer
	data := make([]byte, 1024)
	for {
		n, err := r.Read(data)
		if n > 0 {
			buf.Write(data[:n])
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return buf.Bytes(), err
		}
	}
	content := buf.String()
	lines := strings.Split(content, "\n")
	var result strings.Builder
	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") {
			result.WriteString(strings.TrimPrefix(line, "data: "))
			result.WriteString("\n")
		}
	}
	return []byte(result.String()), nil
}
