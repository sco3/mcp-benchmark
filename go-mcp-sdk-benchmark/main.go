package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
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
	var toolArguments json.RawMessage
	argsJSON := strings.TrimSpace(*argsStr)
	if argsJSON != "" {
		toolArguments = json.RawMessage(argsJSON)
	}

	// Build headers map for custom HTTP client
	headers := parseHeaderArgs(headerList.values)
	if *authToken != "" {
		if headers["Authorization"] == "" {
			headers["Authorization"] = "Bearer " + *authToken
		}
	}

	envAuth := os.Getenv("MCP_AUTHORIZATION")
	if envAuth == "" {
		envAuth = os.Getenv("AUTHORIZATION")
	}
	if envAuth != "" && headers["Authorization"] == "" {
		headers["Authorization"] = strings.TrimSpace(envAuth)
	}

	envBearer := os.Getenv("MCP_AUTH_TOKEN")
	if envBearer == "" {
		envBearer = os.Getenv("MCP_BEARER_TOKEN")
	}
	if envBearer != "" && headers["Authorization"] == "" {
		headers["Authorization"] = "Bearer " + strings.TrimSpace(envBearer)
	}

	fmt.Printf("MCP SDK Benchmark (using go official mcp sdk)\n")
	fmt.Printf("   Transport: Streamable HTTP\n")
	fmt.Printf("   Users: %d, Runs: %d, Calls per cycle: %d\n", *users, *runs, *callsPerCycle)
	fmt.Printf("   Server: %s\n", *serverURL)

	ctx := context.Background()

	// Best-effort tool verification
	fmt.Printf("   Verifying tool '%s' exists...\n", *toolName)
	httpClient := createHTTPClient(headers)
	if err := verifyToolExists(ctx, *serverURL, httpClient, *toolName); err != nil {
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

				// Create a new client session for each cycle with custom HTTP client
				client := mcp.NewClient(&mcp.Implementation{Name: "go-mcp-sdk-benchmark", Version: "1.0.0"}, nil)
				cs, err := client.Connect(ctx, &mcp.StreamableClientTransport{
					Endpoint:   *serverURL,
					HTTPClient: httpClient,
				}, nil)
				if err == nil && cs != nil {
					localInitLatencies = append(localInitLatencies, time.Since(benchStart))
					localInitSuccess++

					// Phase 2: list_tools
					t0 := time.Now()
					_, listErr := cs.ListTools(ctx, &mcp.ListToolsParams{})
					if listErr == nil {
						localListLatencies = append(localListLatencies, time.Since(t0))
						localListSuccess++

						// Phase 3: call_tool N times
						for j := int64(0); j < callsThisCycle; j++ {
							t0 = time.Now()
							_, callErr := cs.CallTool(ctx, &mcp.CallToolParams{
								Name:      *toolName,
								Arguments: toolArguments,
							})
							if callErr == nil {
								localCallLatencies = append(localCallLatencies, time.Since(t0))
								localCallSuccess++
							} else {
								localCallFail++
							}
						}
					} else {
						localListFail++
					}

					// Close the session
					cs.Close()
				} else {
					localInitFail++
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

func createHTTPClient(headers map[string]string) *http.Client {
	if len(headers) == 0 {
		return http.DefaultClient
	}
	
	// Create a custom transport that adds headers
	baseTransport := http.DefaultTransport.(*http.Transport).Clone()
	customTransport := &headerAddingTransport{
		base:    baseTransport,
		headers: headers,
	}
	
	return &http.Client{
		Transport: customTransport,
	}
}

type headerAddingTransport struct {
	base    http.RoundTripper
	headers map[string]string
}

func (h *headerAddingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// Clone the request to avoid modifying the original
	reqCopy := req.Clone(req.Context())
	for key, value := range h.headers {
		reqCopy.Header.Set(key, value)
	}
	return h.base.RoundTrip(reqCopy)
}

func verifyToolExists(ctx context.Context, serverURL string, httpClient *http.Client, toolName string) error {
	client := mcp.NewClient(&mcp.Implementation{Name: "go-mcp-sdk-benchmark", Version: "1.0.0"}, nil)
	cs, err := client.Connect(ctx, &mcp.StreamableClientTransport{
		Endpoint:   serverURL,
		HTTPClient: httpClient,
	}, nil)
	if err != nil || cs == nil {
		return fmt.Errorf("tool verification: connect failed: %v", err)
	}
	defer cs.Close()

	resp, err := cs.ListTools(ctx, &mcp.ListToolsParams{})
	if err != nil {
		return fmt.Errorf("tool verification: list tools failed: %v", err)
	}

	// Check if the requested tool exists
	for _, tool := range resp.Tools {
		if tool.Name == toolName {
			return nil
		}
	}

	// Tool not found - list all available tools
	var availableTools []string
	for _, tool := range resp.Tools {
		availableTools = append(availableTools, tool.Name)
	}

	if len(availableTools) == 0 {
		return fmt.Errorf("Tool '%s' not found. No tools available.", toolName)
	}

	return fmt.Errorf("Tool '%s' not found. Available tools: %s", toolName, strings.Join(availableTools, ", "))
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