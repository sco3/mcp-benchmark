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
	fmt.Printf("\n=== %s Results:\n", p.name)
	fmt.Printf("   Total: %d requests (%d success, %d failed)\n", p.totalRequests, p.successfulRequests, p.failedRequests)
	fmt.Printf("   Elapsed: %.2fs\n", p.elapsed.Seconds())
	fmt.Printf("   Avg latency: %.2fms\n", p.avgLatencyMs())
	fmt.Printf("   Throughput: %.2f req/s\n", p.throughputRPS())
}

func main() {
	serverURL := flag.String("s", "http://localhost:8000/mcp", "server URL")
	runs := flag.Int("r", 100, "number of tool call runs (total across all cycles)")
	callsPerCycle := flag.Int("calls-per-cycle", 100, "number of tool calls per cycle (default: 100)")
	toolName := flag.String("t", "say_hello", "tool name to call")
	args := flag.String("a", "{}", "arguments in JSON format")
	users := flag.Int("u", 1, "number of virtual users")
	flag.Parse()

	if r := os.Getenv("RUNS"); r != "" {
		if n, err := strconv.Atoi(r); err == nil && n > 0 {
			*runs = n
		}
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

	fmt.Printf("MCP Streamable HTTP Benchmark\n")
	fmt.Printf("   Transport: Streamable HTTP\n")
	fmt.Printf("   Users: %d, Tool call runs: %d\n", *users, *runs)
	fmt.Printf("   Server: %s\n", url)
	fmt.Printf("   Scenario per cycle: 1 init -> 1 list_tools -> %d call_tool\n", *callsPerCycle)

	// Best-effort tool verification (matches sdk-benchmark output shape)
	fmt.Printf("   Verifying tool '%s' exists...\n", *toolName)
	if err := verifyToolExists(ctx, client, url, baseHeaders, initReq, notifyReq, listReq, *toolName); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("   [OK] Tool '%s' found\n", *toolName)

	// Create sessions for each virtual user
	type userSession struct {
		sessionID string
	}

	// Cyclic benchmark: each cycle is 1 init -> 1 list -> N calls
	// Total cycles = ceil(total_calls / calls_per_cycle) per user
	totalCalls := *runs
	cyclesPerUser := (totalCalls + *callsPerCycle - 1) / *callsPerCycle
	if totalCalls%*callsPerCycle == 0 && totalCalls > 0 {
		cyclesPerUser = totalCalls / *callsPerCycle
	} else if totalCalls == 0 {
		cyclesPerUser = 0
	}

	var cyclicStats phaseResult
	{
		fmt.Printf("\n=== Cyclic Benchmark: %d clients x %d cycles = %d total cycles\n", *users, cyclesPerUser, cyclesPerUser**users)
		fmt.Printf("   Server: %s\n", url)
		fmt.Printf("   Tool: %s\n", *toolName)
		if *args != "" {
			fmt.Printf("   Arguments: %s\n", *args)
		}
		fmt.Printf("   Concurrency: %d processes (one per client)\n", *users)

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

				callsRemaining := totalCalls
				for callsRemaining > 0 {
					callsThisCycle := *callsPerCycle
					if callsRemaining < *callsPerCycle {
						callsThisCycle = callsRemaining
					}

					// 1. Initialize session (new session each cycle)
					req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBufferString(initReq))
					if err != nil {
						localFailed++
						callsRemaining -= callsThisCycle
						continue
					}
					req.Header = baseHeaders.Clone()
					resp, err := client.Do(req)
					if err != nil {
						if userIdx == 0 && callsRemaining == totalCalls {
							fmt.Fprintf(os.Stderr, "user %d init request: %v\n", userIdx, err)
						}
						localFailed++
						callsRemaining -= callsThisCycle
						continue
					}

					sessionID := resp.Header.Get("Mcp-Session-Id")
					_ = resp.Body.Close()
					if sessionID == "" {
						if userIdx == 0 && callsRemaining == totalCalls {
							fmt.Fprintf(os.Stderr, "user %d init request: no session ID in response\n", userIdx)
						}
						localFailed++
						callsRemaining -= callsThisCycle
						continue
					}

					req, err = http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBufferString(notifyReq))
					if err != nil {
						localFailed++
						callsRemaining -= callsThisCycle
						continue
					}
					req.Header = baseHeaders.Clone()
					req.Header.Set("Mcp-Session-Id", sessionID)
					resp, err = client.Do(req)
					if err != nil {
						if userIdx == 0 && callsRemaining == totalCalls {
							fmt.Fprintf(os.Stderr, "user %d notify request: %v\n", userIdx, err)
						}
						localFailed++
						callsRemaining -= callsThisCycle
						continue
					}
					_ = resp.Body.Close()

					// 2. List tools
					req, err = http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBufferString(listReq))
					if err != nil {
						localFailed++
						callsRemaining -= callsThisCycle
						continue
					}
					req.Header = baseHeaders.Clone()
					req.Header.Set("Mcp-Session-Id", sessionID)
					resp, err = client.Do(req)
					if err != nil {
						if userIdx == 0 && callsRemaining == totalCalls {
							fmt.Fprintf(os.Stderr, "user %d tools/list request: %v\n", userIdx, err)
						}
						localFailed++
						callsRemaining -= callsThisCycle
						continue
					}
					_ = resp.Body.Close()

					// 3. Call tool(s)
					for j := 0; j < callsThisCycle; j++ {
						req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBufferString(callReq))
						if err != nil {
							localFailed++
							continue
						}
						req.Header = baseHeaders.Clone()
						req.Header.Set("Mcp-Session-Id", sessionID)

						latency, err := measureRequest(client, req)
						if err != nil {
							if userIdx == 0 && callsRemaining == totalCalls && j == 0 {
								fmt.Fprintf(os.Stderr, "user %d tool call request: %v\n", userIdx, err)
							}
							localFailed++
							continue
						}
						localLatencies = append(localLatencies, latency)
						localSuccess++
					}

					callsRemaining -= callsThisCycle
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
		cyclicStats = phaseResult{
			name:               "Cyclic",
			totalRequests:      totalCalls * *users,
			successfulRequests: successfulRequests.Load(),
			failedRequests:     failedRequests.Load(),
			latencies:          allLatencies,
			elapsed:            totalElapsed,
		}
		printPhase(cyclicStats)
	}

	fmt.Printf("\n=== Summary:\n")
	fmt.Printf("   Cyclic: %.2f req/s,  %.2fms avg latency\n", cyclicStats.throughputRPS(), cyclicStats.avgLatencyMs())
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
