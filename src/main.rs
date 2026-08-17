use std::process::ExitCode;
use tracing_subscriber::{EnvFilter, fmt};

fn main() -> ExitCode {
    let _guard = init_telemetry();

    if let Err(err) = minecraft::run() {
        // Log via tracing, but fallback to stderr in case tracing setup failed
        tracing::error!("Fatal Engine Error: {err:#}");
        eprintln!("[FATAL ENGINE ERROR]: {err:#}");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

fn init_telemetry() -> impl Drop {
    // 1. Install safe panic hook FIRST to catch panics during telemetry setup
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        let payload = panic_info
            .payload()
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| panic_info.payload().downcast_ref::<String>().map(|s| s.as_str()))
            .unwrap_or("Box<Any>");

        let location = panic_info
            .location()
            .map_or_else(|| "unknown location".to_string(), |l| format!("{}:{}:{}", l.file(), l.line(), l.column()));

        eprintln!("[FATAL ENGINE PANIC] '{payload}' at {location}");

        default_hook(panic_info);
    }));

    // 2. Build environment filter with sensible default overrides
    let default_filter = "info,wgpu_core=warn,wgpu_hal=warn,naga=warn,winit=warn";
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_filter));

    // 3. Initialize subscriber idiomatic try_init
    if let Err(err) = fmt::Subscriber::builder()
        .with_env_filter(env_filter)
        .with_target(true)
        .try_init()
    {
        eprintln!("Warning: Telemetry subscriber failed to initialize: {err}");
    }

    // Guard guarantees stdio flush on exit or unwinding
    FlushGuard
}

struct FlushGuard;

impl Drop for FlushGuard {
    fn drop(&mut self) {
        use std::io::Write;
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
    }
}
