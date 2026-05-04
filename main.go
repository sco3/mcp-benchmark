package main

import (
	"bufio"
	"bytes"
	"context"
	"flag"
	"fmt"
	"io"
	"net"
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
	fmt.Printf("\n📊 %s Results:\n", p.name)
	fmt.Printf("   Total: %d requests (%d success, %d failed)\n", p.totalRequests, p.successfulRequests, p.failedRequests)
	fmt.Printf("   Elapsed: %.2fs\n", p.elapsed.Seconds())
	fmt.Printf("   Avg latency: %.2fms\n", p.avgLatencyMs())
	fmt.Printf("   Throughput: %.2f req/s\n", p.throughputRPS())
}

func main() {
	serverURL := flag.String("s", "http://localhost:8000/mcp", "server URL")
	runs := flag.Int("r", 100, "number of requests")
	initRuns := flag.Int("i", 0, "number of runs for init and tools/list (default: same as -r)")
	toolName := flag.String("t", "say_hello", "tool name to call")
	args := flag.String("a", "{}", "arguments in JSON format")
	users := flag.Int("u", 1, "number of virtual users")
	flag.Parse()

	if r := os.Getenv("RUNS"); r != "" {
		if n, err := strconv.Atoi(r); err == nil && n > 0 {
			*runs = n
		}
	}
	if *initRuns <= 0 {
		*initRuns = *runs
	}

	url := *serverURL
	initReq := `{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"demo","version":"0.0.1"}}}`
	notifyReq := `{"jsonrpc": "2.0","method": "notifications/initialized"}`
	listReq := `{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}`
	callReq := fmt.Sprintf(`{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"%s","arguments":%s}}`, *toolName, *args)

	baseHeaders := http.Header{
		"Content-Type": []string{"application/json; charset=utf-8"},
		"Accept":       []string{"application/json, application/x-ndjson, text/event-stream"},
	}

	ctx := context.Background()
	client := &http.Client{Timeout: 30 * time.Second}

	fmt.Printf("🔌 MCP Streamable HTTP Benchmark\n")
	fmt.Printf("   Transport: Streamable HTTP\n")
	fmt.Printf("   Users: %d, Init runs: %d, Tool call runs: %d\n", *users, *initRuns, *runs)
	fmt.Printf("   Server: %s\n", url)

	// Best-effort tool verification (matches sdk-benchmark output shape)
	fmt.Printf("   Verifying tool '%s' exists...\n", *toolName)
	if err := verifyToolExists(ctx, client, url, baseHeaders, initReq, notifyReq, listReq, *toolName); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("   ✅ Tool '%s' found\n", *toolName)

	// Create sessions for each virtual user
	type userSession struct {
		sessionID string
	}

	// Phase 1: Init benchmark (initialize + notifications/initialized; new session each run)
	var initPhase phaseResult
	{
		totalRuns := (*initRuns) * (*users)
		fmt.Printf("\n🚀 Phase 1 - Init Benchmark: %d clients × %d runs = %d total requests\n", *users, *initRuns, totalRuns)
		fmt.Printf("   Server: %s\n", url)

		var allLatencies []time.Duration
		var successfulRequests atomic.Int64
		var failedRequests atomic.Int64
		var mu sync.Mutex
		benchStart := time.Now()

		var wg sync.WaitGroup
		for i := 0; i < *users; i++ {
			wg.Add(1)
			go func(userIdx int) {
				defer wg.Done()
				var localLatencies []time.Duration
				var localSuccess int64
				var localFailed int64

				for j := 0; j < *initRuns; j++ {
					t0 := time.Now()

					req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBufferString(initReq))
					if err != nil {
						if j == 0 {
							fmt.Fprintf(os.Stderr, "user %d create init request: %v\n", userIdx, err)
						}
						localFailed++
						continue
					}
					req.Header = baseHeaders.Clone()
					resp, err := client.Do(req)
					if err != nil {
						if j == 0 {
							fmt.Fprintf(os.Stderr, "user %d init request %d: %v\n", userIdx, j, err)
						}
						localFailed++
						continue
					}

					sessionID := resp.Header.Get("Mcp-Session-Id")
					_ = resp.Body.Close()
					if sessionID == "" {
						if j == 0 {
							fmt.Fprintf(os.Stderr, "user %d init request %d: no session ID in response\n", userIdx, j)
						}
						localFailed++
						continue
					}

					req, err = http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBufferString(notifyReq))
					if err != nil {
						localFailed++
						continue
					}
					req.Header = baseHeaders.Clone()
					req.Header.Set("Mcp-Session-Id", sessionID)
					resp, err = client.Do(req)
					if err != nil {
						if j == 0 {
							fmt.Fprintf(os.Stderr, "user %d notify request %d: %v\n", userIdx, j, err)
						}
						localFailed++
						continue
					}
					_ = resp.Body.Close()

					localLatencies = append(localLatencies, time.Since(t0))
					localSuccess++
				}

				mu.Lock()
				allLatencies = append(allLatencies, localLatencies...)
				mu.Unlock()
				successfulRequests.Add(localSuccess)
				failedRequests.Add(localFailed)
			}(i)
		}
		wg.Wait()
		totalElapsed := time.Since(benchStart)
		initPhase = phaseResult{
			name:               "Init",
			totalRequests:      totalRuns,
			successfulRequests: successfulRequests.Load(),
			failedRequests:     failedRequests.Load(),
			latencies:          allLatencies,
			elapsed:            totalElapsed,
		}
		printPhase(initPhase)
	}

	// Create sessions once for tools/list and tools/call phases
	sessions := make([]*userSession, *users)
	for i := 0; i < *users; i++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBufferString(initReq))
		if err != nil {
			fmt.Fprintf(os.Stderr, "user %d create init request: %v\n", i, err)
			os.Exit(1)
		}
		req.Header = baseHeaders.Clone()

		resp, err := client.Do(req)
		if err != nil {
			fmt.Fprintf(os.Stderr, "user %d init request: %v\n", i, err)
			os.Exit(1)
		}

		sessionID := resp.Header.Get("Mcp-Session-Id")
		_ = resp.Body.Close()
		if sessionID == "" {
			fmt.Fprintf(os.Stderr, "user %d: no session ID in response\n", i)
			os.Exit(1)
		}

		req, _ = http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBufferString(notifyReq))
		req.Header = baseHeaders.Clone()
		req.Header.Set("Mcp-Session-Id", sessionID)
		resp, err = client.Do(req)
		if err != nil {
			fmt.Fprintf(os.Stderr, "user %d notify request: %v\n", i, err)
			os.Exit(1)
		}
		_ = resp.Body.Close()

		sessions[i] = &userSession{sessionID: sessionID}
	}

	// Phase 2: tools/list benchmark (reuse sessions)
	var listPhase phaseResult
	{
		totalRuns := (*initRuns) * (*users)
		fmt.Printf("\n🚀 Phase 2 - Tool/list Benchmark: %d clients × %d runs = %d total requests\n", *users, *initRuns, totalRuns)
		fmt.Printf("   Server: %s\n", url)

		var allLatencies []time.Duration
		var successfulRequests atomic.Int64
		var failedRequests atomic.Int64
		var mu sync.Mutex
		benchStart := time.Now()

		var wg sync.WaitGroup
		for i := 0; i < *users; i++ {
			wg.Add(1)
			go func(userIdx int, session *userSession) {
				defer wg.Done()
				userHeaders := baseHeaders.Clone()
				userHeaders.Set("Mcp-Session-Id", session.sessionID)

				var localLatencies []time.Duration
				for j := 0; j < *initRuns; j++ {
					req, _ := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBufferString(listReq))
					req.Header = userHeaders.Clone()

					latency, err := measureRequest(client, req)
					if err != nil {
						if j == 0 {
							fmt.Fprintf(os.Stderr, "user %d tools/list request %d: %v\n", userIdx, j, err)
						}
						failedRequests.Add(1)
						continue
					}
					localLatencies = append(localLatencies, latency)
					successfulRequests.Add(1)
				}

				mu.Lock()
				allLatencies = append(allLatencies, localLatencies...)
				mu.Unlock()
			}(i, sessions[i])
		}
		wg.Wait()
		totalElapsed := time.Since(benchStart)
		listPhase = phaseResult{
			name:               "Tool/list",
			totalRequests:      totalRuns,
			successfulRequests: successfulRequests.Load(),
			failedRequests:     failedRequests.Load(),
			latencies:          allLatencies,
			elapsed:            totalElapsed,
		}
		printPhase(listPhase)
	}

	// Benchmark tool calls with all users in parallel
	{
		totalRuns := (*runs) * (*users)
		fmt.Printf("\n🚀 Phase 3 - Tool Call Benchmark: %d clients × %d runs = %d total requests\n", *users, *runs, totalRuns)
		fmt.Printf("   Server: %s\n", url)
		fmt.Printf("   Tool: %s\n", *toolName)
		if *args != "" {
			fmt.Printf("   Arguments: %s\n", *args)
		}
	}

	var allLatencies []time.Duration
	var successfulRequests atomic.Int64
	var failedRequests atomic.Int64
	var mu sync.Mutex
	benchStart := time.Now()

	var wg sync.WaitGroup
	for i := 0; i < *users; i++ {
		wg.Add(1)
		go func(userIdx int, session *userSession) {
			defer wg.Done()
			userHeaders := baseHeaders.Clone()
			userHeaders.Set("Mcp-Session-Id", session.sessionID)

			var localLatencies []time.Duration
			for j := 0; j < *runs; j++ {
				req, _ := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBufferString(callReq))
				req.Header = userHeaders.Clone()

				latency, err := measureRequest(client, req)
				if err != nil {
					fmt.Fprintf(os.Stderr, "user %d request %d: %v\n", userIdx, j, err)
					failedRequests.Add(1)
					continue
				}
				localLatencies = append(localLatencies, latency)
				successfulRequests.Add(1)
			}

			mu.Lock()
			allLatencies = append(allLatencies, localLatencies...)
			mu.Unlock()
		}(i, sessions[i])
	}
	wg.Wait()
	totalElapsed := time.Since(benchStart)
	totalRuns := *runs * *users
	toolPhase := phaseResult{
		name:               "Tool call",
		totalRequests:      totalRuns,
		successfulRequests: successfulRequests.Load(),
		failedRequests:     failedRequests.Load(),
		latencies:          allLatencies,
		elapsed:            totalElapsed,
	}
	printPhase(toolPhase)

	fmt.Printf("\n📈 Summary:\n")
	fmt.Printf("   Init:      %.2f req/s,  %.2fms avg latency\n", initPhase.throughputRPS(), initPhase.avgLatencyMs())
	fmt.Printf("   Tool/list: %.2f req/s,  %.2fms avg latency\n", listPhase.throughputRPS(), listPhase.avgLatencyMs())
	fmt.Printf("   Tool call: %.2f req/s,  %.2fms avg latency\n", toolPhase.throughputRPS(), toolPhase.avgLatencyMs())
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
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBufferString(initReq))
	if err != nil {
		return err
	}
	req.Header = baseHeaders.Clone()
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	sessionID := resp.Header.Get("Mcp-Session-Id")
	_ = resp.Body.Close()
	if sessionID == "" {
		return fmt.Errorf("tool verification: no session ID")
	}

	req, err = http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBufferString(notifyReq))
	if err != nil {
		return err
	}
	req.Header = baseHeaders.Clone()
	req.Header.Set("Mcp-Session-Id", sessionID)
	resp, err = client.Do(req)
	if err != nil {
		return err
	}
	_ = resp.Body.Close()

	req, err = http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBufferString(listReq))
	if err != nil {
		return err
	}
	req.Header = baseHeaders.Clone()
	req.Header.Set("Mcp-Session-Id", sessionID)
	resp, err = client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	body, err := readSSE(resp.Body)
	if err != nil {
		return err
	}
	if !strings.Contains(string(body), toolName) {
		return fmt.Errorf("Tool '%s' not found", toolName)
	}
	return nil
}

func measureRequest(client *http.Client, req *http.Request) (time.Duration, error) {
	conn, err := net.Dial("tcp", req.URL.Host)
	if err != nil {
		return 0, err
	}
	defer conn.Close()

	start := time.Now()
	_, err = fmt.Fprintf(conn, "%s %s HTTP/1.1\r\n", req.Method, req.URL.Path)
	if err != nil {
		return 0, err
	}
	_, err = fmt.Fprintf(conn, "Host: %s\r\n", req.URL.Host)
	if err != nil {
		return 0, err
	}
	for k, vs := range req.Header {
		for _, v := range vs {
			_, err = fmt.Fprintf(conn, "%s: %s\r\n", k, v)
			if err != nil {
				return 0, err
			}
		}
	}
	body, _ := io.ReadAll(req.Body)
	_, err = fmt.Fprintf(conn, "Content-Length: %d\r\n\r\n%s", len(body), body)
	if err != nil {
		return 0, err
	}

	reader := bufio.NewReader(conn)
	var latency time.Duration
	for {
		line, err := reader.ReadBytes('\n')
		if err != nil {
			return 0, err
		}
		if bytes.HasPrefix(line, []byte("data: ")) && len(bytes.TrimSpace(line)) > 6 {
			latency = time.Since(start)
			break
		}
	}

	return latency, nil
}

func percentile(data []time.Duration, p int) time.Duration {
	if len(data) == 0 {
		return 0
	}
	sorted := make([]time.Duration, len(data))
	copy(sorted, data)
	for i := 0; i < len(sorted)-1; i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}
	idx := len(sorted) * p / 100
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
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
