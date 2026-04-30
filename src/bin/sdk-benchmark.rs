//! MCP Streamable HTTP Benchmark
//! Uses rmcp with Streamable HTTP transport
//!
//! Measures three phases:
//! - Init: Session initialization RPS and latency
//! - Tool/list: Tool listing RPM and latency
//! - Tool call: Tool call RPS and latency

use clap::Parser;
use rmcp::{model::*, ServiceExt};
use serde_json::Value;
use std::env;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tokio::task::JoinSet;

#[derive(Parser, Debug)]
#[command(name = "sdk-benchmark")]
#[command(about = "MCP Streamable HTTP Benchmark")]
struct Args {
    /// Server URL
    #[arg(short = 's', default_value = "http://localhost:8000/mcp")]
    server_url: String,

    /// Number of runs per user for tool calls
    #[arg(short = 'r', default_value = "100")]
    runs: usize,

    /// Number of runs for init and list (default: same as runs)
    #[arg(short = 'i', default_value = None)]
    init_runs: Option<usize>,

    /// Tool name to call for benchmark
    #[arg(short = 't', default_value = "say_hello")]
    tool_name: String,

    /// JSON arguments to pass to the tool
    #[arg(short = 'a', default_value = "{}")]
    arguments: String,

    /// Number of concurrent clients
    #[arg(short = 'u', default_value = "1")]
    users: usize,
}

struct BenchmarkStats {
    total_requests: usize,
    successful_requests: usize,
    failed_requests: usize,
    total_latency: Duration,
}

impl BenchmarkStats {
    fn new() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            total_latency: Duration::ZERO,
        }
    }
}

struct PhaseStats {
    name: &'static str,
    total_requests: usize,
    successful_requests: usize,
    failed_requests: usize,
    total_latency: Duration,
    elapsed: Duration,
}

impl Clone for PhaseStats {
    fn clone(&self) -> Self {
        Self {
            name: self.name,
            total_requests: self.total_requests,
            successful_requests: self.successful_requests,
            failed_requests: self.failed_requests,
            total_latency: self.total_latency,
            elapsed: self.elapsed,
        }
    }
}

impl PhaseStats {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            total_latency: Duration::ZERO,
            elapsed: Duration::ZERO,
        }
    }

    fn throughput(&self) -> f64 {
        if self.elapsed.as_secs_f64() > 0.0 {
            self.successful_requests as f64 / self.elapsed.as_secs_f64()
        } else {
            0.0
        }
    }

    fn avg_latency_ms(&self) -> f64 {
        if self.successful_requests > 0 {
            self.total_latency.as_secs_f64() / self.successful_requests as f64 * 1000.0
        } else {
            0.0
        }
    }

    fn print_results(&self) {
        println!("\n📊 {} Results:", self.name);
        println!(
            "   Total: {} requests ({} success, {} failed)",
            self.total_requests, self.successful_requests, self.failed_requests
        );
        println!("   Elapsed: {:.2?}", self.elapsed);
        println!("   Avg latency: {:.2}ms", self.avg_latency_ms());
        println!("   Throughput: {:.2} req/s", self.throughput());
    }
}

async fn run_init_benchmark(
    server_url: &str,
    concurrent_clients: usize,
    runs_per_client: usize,
) -> PhaseStats {
    let stats = Arc::new(Mutex::new(PhaseStats::new("Init")));
    let mut tasks = JoinSet::new();

    let total_requests = concurrent_clients * runs_per_client;
    println!(
        "\n🚀 Phase 1 - Init Benchmark: {} clients × {} runs = {} total requests",
        concurrent_clients, runs_per_client, total_requests
    );
    println!("   Server: {}", server_url);

    let start_time = Instant::now();

    for client_id in 0..concurrent_clients {
        let stats_clone = Arc::clone(&stats);
        let server_url = server_url.to_string();

        tasks.spawn(async move {
            let client_stats = benchmark_init(client_id, server_url, runs_per_client).await;

            let mut stats = stats_clone.lock().await;
            stats.total_requests += client_stats.total_requests;
            stats.successful_requests += client_stats.successful_requests;
            stats.failed_requests += client_stats.failed_requests;
            stats.total_latency += client_stats.total_latency;
        });
    }

    // Wait for all tasks to complete
    while let Some(result) = tasks.join_next().await {
        if let Err(e) = result {
            eprintln!("Task error: {}", e);
        }
    }

    let mut final_stats = stats.lock().await;
    final_stats.elapsed = start_time.elapsed();
    final_stats.clone()
}

async fn run_list_benchmark(
    server_url: &str,
    concurrent_clients: usize,
    runs_per_client: usize,
) -> PhaseStats {
    let stats = Arc::new(Mutex::new(PhaseStats::new("Tool/list")));
    let mut tasks = JoinSet::new();

    let total_requests = concurrent_clients * runs_per_client;
    println!(
        "\n🚀 Phase 2 - Tool/list Benchmark: {} clients × {} runs = {} total requests",
        concurrent_clients, runs_per_client, total_requests
    );
    println!("   Server: {}", server_url);

    let start_time = Instant::now();

    for client_id in 0..concurrent_clients {
        let stats_clone = Arc::clone(&stats);
        let server_url = server_url.to_string();

        tasks.spawn(async move {
            let client_stats = benchmark_list_tools(client_id, server_url, runs_per_client).await;

            let mut stats = stats_clone.lock().await;
            stats.total_requests += client_stats.total_requests;
            stats.successful_requests += client_stats.successful_requests;
            stats.failed_requests += client_stats.failed_requests;
            stats.total_latency += client_stats.total_latency;
        });
    }

    // Wait for all tasks to complete
    while let Some(result) = tasks.join_next().await {
        if let Err(e) = result {
            eprintln!("Task error: {}", e);
        }
    }

    let mut final_stats = stats.lock().await;
    final_stats.elapsed = start_time.elapsed();
    final_stats.clone()
}

async fn run_tool_call_benchmark(
    server_url: &str,
    concurrent_clients: usize,
    runs_per_client: usize,
    tool_name: &str,
    arguments: Option<Value>,
) -> PhaseStats {
    let stats = Arc::new(Mutex::new(PhaseStats::new("Tool call")));
    let mut tasks = JoinSet::new();

    let total_requests = concurrent_clients * runs_per_client;
    println!(
        "\n🚀 Phase 3 - Tool Call Benchmark: {} clients × {} runs = {} total requests",
        concurrent_clients, runs_per_client, total_requests
    );
    println!("   Server: {}", server_url);
    println!("   Tool: {}", tool_name);
    if let Some(ref args) = arguments {
        println!("   Arguments: {}", args);
    }

    let start_time = Instant::now();

    for client_id in 0..concurrent_clients {
        let stats_clone = Arc::clone(&stats);
        let server_url = server_url.to_string();
        let tool_name = tool_name.to_string();
        let arguments = arguments.clone();

        tasks.spawn(async move {
            let client_stats =
                benchmark_tool_call(client_id, server_url, runs_per_client, tool_name, arguments)
                    .await;

            let mut stats = stats_clone.lock().await;
            stats.total_requests += client_stats.total_requests;
            stats.successful_requests += client_stats.successful_requests;
            stats.failed_requests += client_stats.failed_requests;
            stats.total_latency += client_stats.total_latency;
        });
    }

    // Wait for all tasks to complete
    while let Some(result) = tasks.join_next().await {
        if let Err(e) = result {
            eprintln!("Task error: {}", e);
        }
    }

    let mut final_stats = stats.lock().await;
    final_stats.elapsed = start_time.elapsed();
    final_stats.clone()
}

async fn benchmark_init(client_id: usize, base_url: String, num_requests: usize) -> BenchmarkStats {
    let mut stats = BenchmarkStats::new();

    for i in 0..num_requests {
        let start = Instant::now();

        let transport = rmcp::transport::StreamableHttpClientTransport::from_uri(base_url.as_str());
        let handler = ();

        match handler.serve(transport).await {
            Ok(mut running_service) => {
                // Explicitly close the session and wait for cleanup
                if let Err(e) = running_service.close().await {
                    stats.failed_requests += 1;
                    if i == 0 {
                        eprintln!("   Client {} session close error: {}", client_id, e);
                    }
                    stats.total_requests += 1;
                    stats.total_latency += start.elapsed();
                    continue;
                }
                stats.successful_requests += 1;
            }
            Err(e) => {
                stats.failed_requests += 1;
                if i == 0 {
                    eprintln!("   Client {} init error: {}", client_id, e);
                }
            }
        }

        stats.total_requests += 1;
        stats.total_latency += start.elapsed();
    }

    stats
}

async fn benchmark_list_tools(
    client_id: usize,
    base_url: String,
    num_requests: usize,
) -> BenchmarkStats {
    let mut stats = BenchmarkStats::new();

    let transport = rmcp::transport::StreamableHttpClientTransport::from_uri(base_url.as_str());
    let handler = ();

    let client_peer = match handler.serve(transport).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("   Client {} failed to initialize: {}", client_id, e);
            stats.failed_requests = num_requests;
            stats.total_requests = num_requests;
            return stats;
        }
    };

    // Reuse session for all list_tools calls
    for i in 0..num_requests {
        let start = Instant::now();

        match client_peer.list_tools(None).await {
            Ok(_) => {
                stats.successful_requests += 1;
            }
            Err(e) => {
                stats.failed_requests += 1;
                if i == 0 {
                    eprintln!("   Client {} list_tools error: {}", client_id, e);
                }
            }
        }

        let elapsed = start.elapsed();
        eprintln!("   [list] Request {}: {:.3}ms", i + 1, elapsed.as_secs_f64() * 1000.0);
        stats.total_requests += 1;
        stats.total_latency += elapsed;
    }

    stats
}

async fn benchmark_tool_call(
    client_id: usize,
    base_url: String,
    num_requests: usize,
    tool_name: String,
    arguments: Option<Value>,
) -> BenchmarkStats {
    let mut stats = BenchmarkStats::new();

    let transport = rmcp::transport::StreamableHttpClientTransport::from_uri(base_url.as_str());
    let handler = ();

    let client_peer = match handler.serve(transport).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("   Client {} failed to initialize: {}", client_id, e);
            stats.failed_requests = num_requests;
            stats.total_requests = num_requests;
            return stats;
        }
    };

    // Reuse session for all tool calls
    for i in 0..num_requests {
        let start = Instant::now();

        // Build tool call parameters using owned String
        let mut params = CallToolRequestParams::new(tool_name.to_owned());
        if let Some(ref args) = arguments {
            if let Value::Object(obj) = args {
                params = params.with_arguments(obj.clone());
            }
        }

        // Call specified tool
        match client_peer.call_tool(params).await {
            Ok(result) => {
                // Check if the tool call returned an error (is_error flag)
                if result.is_error.unwrap_or(false) {
                    stats.failed_requests += 1;
                    if i == 0 {
                        // Extract error message from content
                        let error_msg = result
                            .content
                            .iter()
                            .find_map(|c| {
                                if let RawContent::Text(text) = &**c {
                                    Some(text.text.clone())
                                } else {
                                    None
                                }
                            })
                            .unwrap_or_else(|| "Tool call failed".to_string());
                        eprintln!("   Client {} tool error: {}", client_id, error_msg);
                    }
                } else if !result.content.is_empty() {
                    stats.successful_requests += 1;
                } else {
                    stats.failed_requests += 1;
                    if i == 0 {
                        eprintln!("   Client {} error: Empty response", client_id);
                    }
                }
            }
            Err(e) => {
                stats.failed_requests += 1;
                if i == 0 {
                    eprintln!("   Client {} error: {}", client_id, e);
                }
            }
        }

        let elapsed = start.elapsed();
        //eprintln!("   [tool] Request {}: {:.3}ms", i + 1, elapsed.as_secs_f64() * 1000.0);
        stats.total_requests += 1;
        stats.total_latency += elapsed;
    }

    stats
}

/// Verify that the tool exists on the server, otherwise print available tools
async fn verify_tool_exists(
    base_url: &str,
    tool_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let transport = rmcp::transport::StreamableHttpClientTransport::from_uri(base_url);
    let handler = ();

    let client_peer = handler
        .serve(transport)
        .await
        .map_err(|e| format!("Failed to connect to server: {}", e))?;

    // Try to list tools
    let tools_result = client_peer
        .list_tools(None)
        .await
        .map_err(|e| format!("Failed to list tools: {}", e))?;

    let tool_names: Vec<&str> = tools_result.tools.iter().map(|t| t.name.as_ref()).collect();

    if !tool_names.contains(&tool_name) {
        eprintln!("\n❌ Tool '{}' not found on server", tool_name);
        eprintln!("\nAvailable tools:");
        for name in tool_names {
            eprintln!("  - {}", name);
        }
        return Err(format!("Tool '{}' not found", tool_name).into());
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = Args::parse();

    // Override runs from environment if set
    if let Ok(r) = env::var("RUNS") {
        if let Ok(n) = r.parse::<usize>() {
            if n > 0 {
                args.runs = n;
            }
        }
    }

    let init_runs = args.init_runs.unwrap_or(args.runs);

    println!("🔌 MCP Streamable HTTP Benchmark");
    println!("   Transport: Streamable HTTP");
    println!(
        "   Users: {}, Init runs: {}, Tool call runs: {}",
        args.users, init_runs, args.runs
    );
    println!("   Server: {}", args.server_url);

    // Parse JSON arguments if provided
    let tool_arguments = if args.arguments.is_empty() {
        None
    } else {
        // Parse as Value
        let value: Value = serde_json::from_str(&args.arguments)
            .map_err(|e| format!("Failed to parse JSON arguments: {}", e))?;

        // Ensure it's an object
        match value {
            Value::Object(_) => Some(value),
            _ => {
                return Err("JSON arguments must be an object (e.g., {\"key\": \"value\"})".into())
            }
        }
    };

    // Verify tool exists on server
    println!("   Verifying tool '{}' exists...", args.tool_name);
    verify_tool_exists(&args.server_url, &args.tool_name).await?;
    println!("   ✅ Tool '{}' found", args.tool_name);

    // Phase 1: Init benchmark
    let init_stats = run_init_benchmark(&args.server_url, args.users, init_runs).await;
    init_stats.print_results();

    // Phase 2: Tool/list benchmark
    let list_stats = run_list_benchmark(&args.server_url, args.users, init_runs).await;
    list_stats.print_results();

    // Phase 3: Tool call benchmark
    let tool_call_stats = run_tool_call_benchmark(
        &args.server_url,
        args.users,
        args.runs,
        &args.tool_name,
        tool_arguments,
    )
    .await;
    tool_call_stats.print_results();

    // Summary
    println!("\n📈 Summary:");
    println!(
        "   Init:      {:.2} req/s,  {:.2}ms avg latency",
        init_stats.throughput(),
        init_stats.avg_latency_ms()
    );
    println!(
        "   Tool/list: {:.2} req/s,  {:.2}ms avg latency",
        list_stats.throughput(),
        list_stats.avg_latency_ms()
    );
    println!(
        "   Tool call: {:.2} req/s,  {:.2}ms avg latency",
        tool_call_stats.throughput(),
        tool_call_stats.avg_latency_ms()
    );

    Ok(())
}
