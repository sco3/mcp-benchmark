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

func main() {
	serverURL := flag.String("s", "http://localhost:8000/mcp", "server URL")
	runs := flag.Int("r", 100, "number of requests")
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
	callReq := fmt.Sprintf(`{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"%s","arguments":%s}}`, *toolName, *args)

	baseHeaders := http.Header{
		"Content-Type": []string{"application/json; charset=utf-8"},
		"Accept":       []string{"application/json, application/x-ndjson, text/event-stream"},
	}

	ctx := context.Background()
	client := &http.Client{Timeout: 30 * time.Second}

	// Create sessions for each virtual user
	type userSession struct {
		sessionID string
	}

	sessions := make([]*userSession, *users)
	for i := 0; i < *users; i++ {
		// Step 1: Initialize
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
		if sessionID == "" {
			fmt.Fprintf(os.Stderr, "user %d: no session ID in response\n", i)
			os.Exit(1)
		}
		resp.Body.Close()

		// Step 2: Notify
		req, _ = http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBufferString(notifyReq))
		req.Header = baseHeaders.Clone()
		req.Header.Set("Mcp-Session-Id", sessionID)
		resp, err = client.Do(req)
		if err != nil {
			fmt.Fprintf(os.Stderr, "user %d notify request: %v\n", i, err)
			os.Exit(1)
		}
		resp.Body.Close()

		sessions[i] = &userSession{sessionID: sessionID}
	}

	// Benchmark tool calls with all users in parallel
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
	benchEnd := time.Now()
	totalElapsed := benchEnd.Sub(benchStart)
	totalRuns := *runs * *users
	avgElapsed := totalElapsed / time.Duration(totalRuns)

	var total time.Duration
	var min, max time.Duration
	for i, lat := range allLatencies {
		total += lat
		if i == 0 || lat < min {
			min = lat
		}
		if i == 0 || lat > max {
			max = lat
		}
	}
	avg := total / time.Duration(len(allLatencies))
	rps := 1e9 / float64(avg.Nanoseconds())
	throughputRps := float64(totalRuns) / totalElapsed.Seconds()

	fmt.Printf("Virtual users: %d\n", *users)
	fmt.Printf("Runs per user: %d\n", *runs)
	fmt.Printf("Total requests: %d\n", totalRuns)
	fmt.Printf("Total elapsed: %v\n", totalElapsed)
	fmt.Printf("Avg latency (individual): %v\n", avg)
	fmt.Printf("Avg latency (throughput): %v\n", avgElapsed)
	fmt.Printf("Min latency: %v\n", min)
	fmt.Printf("Max latency: %v\n", max)
	fmt.Printf("P50 latency: %v\n", percentile(allLatencies, 50))
	fmt.Printf("P95 latency: %v\n", percentile(allLatencies, 95))
	fmt.Printf("P99 latency: %v\n", percentile(allLatencies, 99))
	fmt.Printf("RPS (individual): %.2f\n", rps)
	fmt.Printf("RPS (throughput): %.2f\n", throughputRps)
	fmt.Printf("\nRequests:\n")
	fmt.Printf("  Successful: %d\n", successfulRequests.Load())
	fmt.Printf("  Failed: %d\n", failedRequests.Load())
	successRate := 0.0
	if totalRuns > 0 {
		successRate = float64(successfulRequests.Load()) / float64(totalRuns) * 100.0
	}
	fmt.Printf("  Success Rate: %.2f%%\n", successRate)
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
