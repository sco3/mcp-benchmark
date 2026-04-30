use clap::Parser;
use reqwest::header::{HeaderMap, HeaderValue, ACCEPT, CONTENT_TYPE};
use std::env;
use std::io::{self, BufRead, Write};
use std::net::TcpStream;
use std::os::linux::net::TcpStreamExt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(name = "test-with-rust")]
#[command(about = "MCP benchmark tool")]
struct Args {
    #[arg(short = 's', default_value = "http://localhost:8000/mcp")]
    server_url: String,

    #[arg(short = 'r', default_value = "100")]
    runs: usize,

    /// Number of runs for init and tools/list (default: same as runs)
    #[arg(short = 'i', default_value = None)]
    init_runs: Option<usize>,

    #[arg(short = 't', default_value = "say_hello")]
    tool_name: String,

    #[arg(short = 'a', default_value = "{}")]
    args: String,

    #[arg(short = 'u', default_value = "1")]
    users: usize,
}

struct UserSession {
    session_id: String,
}

#[derive(Clone)]
struct PhaseStats {
    name: &'static str,
    total_requests: usize,
    successful_requests: usize,
    failed_requests: usize,
    total_latency: Duration,
    elapsed: Duration,
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

    fn avg_latency_ms(&self) -> f64 {
        if self.successful_requests > 0 {
            self.total_latency.as_secs_f64() / self.successful_requests as f64 * 1000.0
        } else {
            0.0
        }
    }

    fn throughput(&self) -> f64 {
        if self.elapsed.as_secs_f64() > 0.0 {
            self.successful_requests as f64 / self.elapsed.as_secs_f64()
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = Args::parse();

    // Override runs from environment if set
    if let Ok(r) = env::var("RUNS") {
        if let Ok(n) = r.parse::<usize>() {
            if n > 0 {
                args.runs = n;
            }
        }
    }

    let url = &args.server_url;
    let init_runs = args.init_runs.unwrap_or(args.runs);
    let init_req = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"demo","version":"0.0.1"}}}"#;
    let notify_req = r#"{"jsonrpc": "2.0","method": "notifications/initialized"}"#;
    let list_req = r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#;
    let call_req = format!(
        r#"{{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{{"name":"{}","arguments":{}}}}}"#,
        args.tool_name, args.args
    );

    let mut base_headers = HeaderMap::new();
    base_headers.insert(
        CONTENT_TYPE,
        HeaderValue::from_static("application/json; charset=utf-8"),
    );
    base_headers.insert(
        ACCEPT,
        HeaderValue::from_static("application/json, application/x-ndjson, text/event-stream"),
    );

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;

    println!("🔌 MCP Streamable HTTP Benchmark");
    println!("   Transport: Streamable HTTP");
    println!(
        "   Users: {}, Init runs: {}, Tool call runs: {}",
        args.users, init_runs, args.runs
    );
    println!("   Server: {}", args.server_url);

    // Best-effort tool verification (to match sdk-benchmark output shape)
    println!("   Verifying tool '{}' exists...", args.tool_name);
    verify_tool_exists(
        &client,
        url,
        &base_headers,
        init_req,
        notify_req,
        list_req,
        &args.tool_name,
    )?;
    println!("   ✅ Tool '{}' found", args.tool_name);

    // Phase 1: Init benchmark (initialize + notifications/initialized)
    println!(
        "\n🚀 Phase 1 - Init Benchmark: {} clients × {} runs = {} total requests",
        args.users,
        init_runs,
        args.users * init_runs
    );
    println!("   Server: {}", args.server_url);
    let init_phase = run_init_phase(
        &client,
        url,
        &base_headers,
        init_req,
        notify_req,
        args.users,
        init_runs,
    );
    init_phase.print_results();

    // Create sessions for list/tools and tool/call phases
    let sessions = create_sessions(
        &client,
        url,
        &base_headers,
        init_req,
        notify_req,
        args.users,
    )?;

    // Phase 2: Tool/list benchmark (reuse sessions)
    println!(
        "\n🚀 Phase 2 - Tool/list Benchmark: {} clients × {} runs = {} total requests",
        args.users,
        init_runs,
        args.users * init_runs
    );
    println!("   Server: {}", args.server_url);
    let list_phase = run_list_phase(&client, url, &base_headers, &sessions, list_req, init_runs)?;
    list_phase.print_results();

    // Phase 3: Tool call benchmark (reuse sessions)
    println!(
        "\n🚀 Phase 3 - Tool Call Benchmark: {} clients × {} runs = {} total requests",
        args.users,
        args.runs,
        args.users * args.runs
    );
    println!("   Server: {}", args.server_url);
    println!("   Tool: {}", args.tool_name);
    let tool_call_phase =
        run_tool_call_phase(&client, url, &base_headers, &sessions, &call_req, args.runs)?;
    tool_call_phase.print_results();

    println!("\n📈 Summary:");
    println!(
        "   Init:      {:.2} req/s,  {:.2}ms avg latency",
        init_phase.throughput(),
        init_phase.avg_latency_ms()
    );
    println!(
        "   Tool/list: {:.2} req/s,  {:.2}ms avg latency",
        list_phase.throughput(),
        list_phase.avg_latency_ms()
    );
    println!(
        "   Tool call: {:.2} req/s,  {:.2}ms avg latency",
        tool_call_phase.throughput(),
        tool_call_phase.avg_latency_ms()
    );

    Ok(())
}

fn verify_tool_exists(
    client: &reqwest::blocking::Client,
    url: &str,
    base_headers: &HeaderMap,
    init_req: &str,
    notify_req: &str,
    list_req: &str,
    tool_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let sessions = create_sessions(client, url, base_headers, init_req, notify_req, 1)?;
    let session_id = &sessions[0].session_id;

    let mut headers = base_headers.clone();
    headers.insert("Mcp-Session-Id", HeaderValue::from_str(session_id)?);

    let resp = client
        .post(url)
        .body(list_req.to_string())
        .headers(headers)
        .send()?;
    let body = resp.text()?;

    // Best-effort match: rmcp/go servers typically include tool names in JSON content.
    if body.contains(tool_name) {
        Ok(())
    } else {
        Err(format!("Tool '{}' not found", tool_name).into())
    }
}

fn create_sessions(
    client: &reqwest::blocking::Client,
    url: &str,
    base_headers: &HeaderMap,
    init_req: &str,
    notify_req: &str,
    users: usize,
) -> Result<Vec<UserSession>, Box<dyn std::error::Error>> {
    let mut sessions: Vec<UserSession> = Vec::with_capacity(users);

    for i in 0..users {
        // Step 1: Initialize
        let resp = client
            .post(url)
            .body(init_req.to_string())
            .headers(base_headers.clone())
            .send()?;

        let session_id = resp
            .headers()
            .get("Mcp-Session-Id")
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| format!("user {}: no session ID in response", i))?
            .to_string();

        resp.text()?; // consume response body

        // Step 2: Notify
        let mut notify_headers = base_headers.clone();
        notify_headers.insert("Mcp-Session-Id", HeaderValue::from_str(&session_id)?);
        let resp = client
            .post(url)
            .body(notify_req.to_string())
            .headers(notify_headers)
            .send()?;
        resp.text()?; // consume response body

        sessions.push(UserSession { session_id });
    }

    Ok(sessions)
}

fn run_init_phase(
    client: &reqwest::blocking::Client,
    url: &str,
    base_headers: &HeaderMap,
    init_req: &str,
    notify_req: &str,
    users: usize,
    runs_per_user: usize,
) -> PhaseStats {
    let result = Arc::new(Mutex::new(PhaseStats::new("Init")));
    let start = Instant::now();

    let mut handles = Vec::new();
    for user_idx in 0..users {
        let client = client.clone();
        let url = url.to_string();
        let base_headers = base_headers.clone();
        let init_req = init_req.to_string();
        let notify_req = notify_req.to_string();
        let result = Arc::clone(&result);

        let handle = std::thread::spawn(move || {
            let mut local_success = 0usize;
            let mut local_failed = 0usize;
            let mut local_total_latency = Duration::ZERO;

            for j in 0..runs_per_user {
                let t0 = Instant::now();

                let resp = match client
                    .post(&url)
                    .body(init_req.clone())
                    .headers(base_headers.clone())
                    .send()
                {
                    Ok(r) => r,
                    Err(e) => {
                        if j == 0 {
                            eprintln!("user {} init request {}: {}", user_idx, j, e);
                        }
                        local_failed += 1;
                        continue;
                    }
                };

                let session_id = match resp
                    .headers()
                    .get("Mcp-Session-Id")
                    .and_then(|v| v.to_str().ok())
                {
                    Some(v) if !v.is_empty() => v.to_string(),
                    _ => {
                        if j == 0 {
                            eprintln!("user {} init request {}: no session id", user_idx, j);
                        }
                        local_failed += 1;
                        continue;
                    }
                };

                if resp.text().is_err() {
                    local_failed += 1;
                    continue;
                }

                let mut notify_headers = base_headers.clone();
                if let Ok(v) = HeaderValue::from_str(&session_id) {
                    notify_headers.insert("Mcp-Session-Id", v);
                } else {
                    local_failed += 1;
                    continue;
                }

                let resp = match client
                    .post(&url)
                    .body(notify_req.clone())
                    .headers(notify_headers)
                    .send()
                {
                    Ok(r) => r,
                    Err(e) => {
                        if j == 0 {
                            eprintln!("user {} notify request {}: {}", user_idx, j, e);
                        }
                        local_failed += 1;
                        continue;
                    }
                };

                if resp.text().is_err() {
                    local_failed += 1;
                    continue;
                }

                local_success += 1;
                local_total_latency += t0.elapsed();
            }

            let mut r = result.lock().unwrap();
            r.total_requests += runs_per_user;
            r.successful_requests += local_success;
            r.failed_requests += local_failed;
            r.total_latency += local_total_latency;
        });

        handles.push(handle);
    }

    for h in handles {
        let _ = h.join();
    }

    let mut r = result.lock().unwrap();
    r.elapsed = start.elapsed();
    r.clone()
}

fn run_list_phase(
    client: &reqwest::blocking::Client,
    url: &str,
    base_headers: &HeaderMap,
    sessions: &[UserSession],
    list_req: &str,
    runs_per_user: usize,
) -> Result<PhaseStats, Box<dyn std::error::Error>> {
    let result = Arc::new(Mutex::new(PhaseStats::new("Tool/list")));
    let start = Instant::now();

    // Prepare request bytes once per user (includes Mcp-Session-Id)
    let mut prepared: Vec<(Vec<u8>, String, u16)> = Vec::with_capacity(sessions.len());
    for session in sessions {
        let mut req = client.post(url).body(list_req.to_string()).headers({
            let mut h = base_headers.clone();
            h.insert(
                "Mcp-Session-Id",
                HeaderValue::from_str(&session.session_id)?,
            );
            h
        });
        let (bytes, host, port) = prepare_request_bytes(&mut req)?;
        prepared.push((bytes, host, port));
    }

    let mut handles = Vec::new();
    for (user_idx, (bytes, host, port)) in prepared.into_iter().enumerate() {
        let result = Arc::clone(&result);
        let handle = std::thread::spawn(move || {
            let mut local_success = 0usize;
            let mut local_failed = 0usize;
            let mut local_total_latency = Duration::ZERO;

            for j in 0..runs_per_user {
                match measure_request(&bytes, &host, port) {
                    Ok(lat) => {
                        local_total_latency += lat;
                        local_success += 1;
                    }
                    Err(e) => {
                        if j == 0 {
                            eprintln!("user {} tools/list request {}: {}", user_idx, j, e);
                        }
                        local_failed += 1;
                    }
                }
            }

            let mut r = result.lock().unwrap();
            r.total_requests += runs_per_user;
            r.successful_requests += local_success;
            r.failed_requests += local_failed;
            r.total_latency += local_total_latency;
        });
        handles.push(handle);
    }

    for h in handles {
        let _ = h.join();
    }

    let mut r = result.lock().unwrap();
    r.elapsed = start.elapsed();
    Ok(r.clone())
}

fn run_tool_call_phase(
    client: &reqwest::blocking::Client,
    url: &str,
    base_headers: &HeaderMap,
    sessions: &[UserSession],
    call_req: &str,
    runs_per_user: usize,
) -> Result<PhaseStats, Box<dyn std::error::Error>> {
    let result = Arc::new(Mutex::new(PhaseStats::new("Tool call")));
    let start = Instant::now();

    // Prepare request bytes once per user (includes Mcp-Session-Id)
    let mut prepared: Vec<(Vec<u8>, String, u16)> = Vec::with_capacity(sessions.len());
    for session in sessions {
        let mut req = client.post(url).body(call_req.to_string()).headers({
            let mut h = base_headers.clone();
            h.insert(
                "Mcp-Session-Id",
                HeaderValue::from_str(&session.session_id)?,
            );
            h
        });
        let (bytes, host, port) = prepare_request_bytes(&mut req)?;
        prepared.push((bytes, host, port));
    }

    let mut handles = Vec::new();
    for (user_idx, (bytes, host, port)) in prepared.into_iter().enumerate() {
        let result = Arc::clone(&result);
        let handle = std::thread::spawn(move || {
            let mut local_success = 0usize;
            let mut local_failed = 0usize;
            let mut local_total_latency = Duration::ZERO;

            for j in 0..runs_per_user {
                match measure_request(&bytes, &host, port) {
                    Ok(lat) => {
                        local_total_latency += lat;
                        local_success += 1;
                    }
                    Err(e) => {
                        if j == 0 {
                            eprintln!("user {} tool call request {}: {}", user_idx, j, e);
                        }
                        local_failed += 1;
                    }
                }
            }

            let mut r = result.lock().unwrap();
            r.total_requests += runs_per_user;
            r.successful_requests += local_success;
            r.failed_requests += local_failed;
            r.total_latency += local_total_latency;
        });
        handles.push(handle);
    }

    for h in handles {
        let _ = h.join();
    }

    let mut r = result.lock().unwrap();
    r.elapsed = start.elapsed();
    Ok(r.clone())
}

fn prepare_request_bytes(
    req: &mut reqwest::blocking::RequestBuilder,
) -> Result<(Vec<u8>, String, u16), Box<dyn std::error::Error>> {
    let prepared = req.try_clone().ok_or("cannot clone request")?;
    let built = prepared.build()?;

    let url = built.url();
    let host = url.host_str().ok_or("no host in URL")?;
    let port = url.port_or_known_default().ok_or("no port in URL")?;

    let method = built.method().as_str();
    let path = url.path();
    let query = url.query().unwrap_or("");
    let path_with_query = if query.is_empty() {
        path.to_string()
    } else {
        format!("{}?{}", path, query)
    };

    // Собираем весь HTTP-пакет в один буфер
    let mut buffer = Vec::with_capacity(512);
    write!(buffer, "{} {} HTTP/1.1\r\n", method, path_with_query)?;
    write!(buffer, "Host: {}:{}\r\n", host, port)?;

    for (name, value) in built.headers() {
        write!(buffer, "{}: {}\r\n", name, value.to_str().unwrap_or(""))?;
    }

    let body = built.body().and_then(|b| b.as_bytes()).unwrap_or(&[]);
    write!(buffer, "Content-Length: {}\r\n\r\n", body.len())?;
    buffer.extend_from_slice(body);

    Ok((buffer, host.to_string(), port))
}

fn measure_request(
    prepared_bytes: &[u8],
    host: &str,
    port: u16,
) -> Result<Duration, Box<dyn std::error::Error>> {
    let start = Instant::now();

    let mut stream = TcpStream::connect(format!("{}:{}", host, port))?;
    let _ = stream.set_nodelay(true); // Disable Nagle's algorithm
    let _ = stream.set_quickack(true).ok(); // Immediate ACK (if supported by OS)
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;

    // Send everything in a single write_all call
    stream.write_all(prepared_bytes)?;

    let mut reader = io::BufReader::new(stream);
    let mut line = String::new();

    loop {
        line.clear();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            return Err("connection closed".into());
        }
        if line.starts_with("data: ") && line.trim().len() > 6 {
            return Ok(start.elapsed());
        }
    }
}

// (percentile helper removed; sdk-benchmark-style output doesn't use it)
