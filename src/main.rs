use clap::Parser;
use reqwest::header::{HeaderMap, HeaderValue, ACCEPT, CONTENT_TYPE};
use std::env;
use std::io::{self, BufRead, Write};
use std::net::TcpStream;
use std::os::linux::net::TcpStreamExt;
use std::sync::atomic::{AtomicUsize, Ordering};
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
    let init_req = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"demo","version":"0.0.1"}}}"#;
    let notify_req = r#"{"jsonrpc": "2.0","method": "notifications/initialized"}"#;
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

    // Create sessions for each virtual user
    let mut sessions: Vec<UserSession> = Vec::with_capacity(args.users);

    for i in 0..args.users {
        // Step 1: Initialize
        let mut req = client
            .post(url)
            .body(init_req.to_string())
            .headers(base_headers.clone());
        let resp = req.send()?;

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
        req = client
            .post(url)
            .body(notify_req.to_string())
            .headers(notify_headers);
        let resp = req.send()?;
        resp.text()?; // consume response body

        sessions.push(UserSession { session_id });
    }

    // Benchmark tool calls with all users in parallel
    let all_latencies = Arc::new(Mutex::new(Vec::new()));
    let successful_requests = Arc::new(AtomicUsize::new(0));
    let failed_requests = Arc::new(AtomicUsize::new(0));
    let bench_start = Instant::now();

    // Prepare request bytes once before the loop
    let mut prepared_requests: Vec<(Vec<u8>, String, u16)> = Vec::with_capacity(args.users);
    for i in 0..args.users {
        let mut req = client.post(url).body(call_req.clone()).headers({
            let mut h = base_headers.clone();
            h.insert(
                "Mcp-Session-Id",
                HeaderValue::from_str(&sessions[i].session_id).unwrap(),
            );
            h
        });
        let (bytes, host, port) = prepare_request_bytes(&mut req)?;
        prepared_requests.push((bytes, host, port));
    }

    let mut handles = Vec::new();
    for i in 0..args.users {
        let (prepared_bytes, host, port) = prepared_requests.remove(0);
        let runs = args.runs;
        let all_latencies = Arc::clone(&all_latencies);
        let successful_requests = Arc::clone(&successful_requests);
        let failed_requests = Arc::clone(&failed_requests);

        let handle = std::thread::spawn(move || {
            let mut local_latencies = Vec::new();
            let mut local_success = 0;
            let mut local_failed = 0;
            for j in 0..runs {
                match measure_request(&prepared_bytes, &host, port) {
                    Ok(latency) => {
                        local_latencies.push(latency);
                        local_success += 1;
                    }
                    Err(e) => {
                        eprintln!("user {} request {}: {}", i, j, e);
                        local_failed += 1;
                    }
                }
            }

            let mut latencies = all_latencies.lock().unwrap();
            latencies.extend(local_latencies);

            successful_requests.fetch_add(local_success, Ordering::Relaxed);
            failed_requests.fetch_add(local_failed, Ordering::Relaxed);
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let bench_end = Instant::now();
    let total_elapsed = bench_end.duration_since(bench_start);
    let total_runs = args.runs * args.users;
    let avg_elapsed = total_elapsed / total_runs as u32;

    let latencies = all_latencies.lock().unwrap();
    let success_count = successful_requests.load(Ordering::Relaxed);
    let failed_count = failed_requests.load(Ordering::Relaxed);
    let (total, min, max) = if latencies.is_empty() {
        (Duration::ZERO, Duration::ZERO, Duration::ZERO)
    } else {
        let total = latencies.iter().sum::<Duration>();
        let min = *latencies.iter().min().unwrap();
        let max = *latencies.iter().max().unwrap();
        (total, min, max)
    };

    let avg = if latencies.is_empty() {
        Duration::ZERO
    } else {
        total / latencies.len() as u32
    };

    let rps = if avg.as_nanos() > 0 {
        1e9 / avg.as_nanos() as f64
    } else {
        0.0
    };

    let throughput_rps = if total_elapsed.as_secs_f64() > 0.0 {
        total_runs as f64 / total_elapsed.as_secs_f64()
    } else {
        0.0
    };

    println!("Virtual users: {}", args.users);
    println!("Runs per user: {}", args.runs);
    println!("Total requests: {}", total_runs);
    println!("Total elapsed: {:?}", total_elapsed);
    println!("Avg latency (individual): {:?}", avg);
    println!("Avg latency (throughput): {:?}", avg_elapsed);
    println!("Min latency: {:?}", min);
    println!("Max latency: {:?}", max);
    println!("P50 latency: {:?}", percentile(&latencies, 50));
    println!("P95 latency: {:?}", percentile(&latencies, 95));
    println!("P99 latency: {:?}", percentile(&latencies, 99));
    println!("RPS (individual): {:.2}", rps);
    println!("RPS (throughput): {:.2}", throughput_rps);
    println!("\nRequests:");
    println!("  Successful: {}", success_count);
    println!("  Failed: {}", failed_count);
    let success_rate = if total_runs > 0 {
        (success_count as f64 / total_runs as f64) * 100.0
    } else {
        0.0
    };
    println!("  Success Rate: {:.2}%", success_rate);

    Ok(())
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

fn percentile(data: &[Duration], p: usize) -> Duration {
    if data.is_empty() {
        return Duration::ZERO;
    }
    let mut sorted: Vec<Duration> = data.to_vec();
    sorted.sort();
    let idx = sorted.len() * p / 100;
    let idx = idx.min(sorted.len() - 1);
    sorted[idx]
}
