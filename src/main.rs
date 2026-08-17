use std::process::ExitCode;
use tracing_subscriber::{EnvFilter, fmt};

fn main() -> ExitCode {
    let _guard = init_telemetry();

    if let Err(err) = minecraft::run() {
        tracing::error!("Fatal Engine Error: {err:#}");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

fn init_telemetry() -> impl Drop {
    // 1. Build environment filter with sensible default overrides for noisy dependencies
    let default_filter = "info,wgpu_core=warn,wgpu_hal=warn,naga=warn,winit=warn";
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_filter));

    // 2. Initialize tracing subscriber
    let subscriber = fmt::Subscriber::builder()
        .with_env_filter(env_filter)
        .with_target(true)
        .finish();

    let logger_initialized = tracing::subscriber::set_global_default(subscriber).is_ok();
    if !logger_initialized {
        eprintln!("Warning: Global telemetry subscriber was already initialized.");
    }

    // 3. Install safe panic hook (bypasses logger locks to prevent deadlocks)
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        let payload = panic_info.payload().downcast_ref::<&str>().map_or_else(
            || {
                panic_info
                    .payload()
                    .downcast_ref::<String>()
                    .map_or("Box<Any>", |s| s.as_str())
            },
            |s| *s,
        );

        let location = panic_info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "unknown location".to_string());

        eprintln!("[FATAL ENGINE PANIC] '{payload}' at {location}");

        default_hook(panic_info);
    }));

    // RAII guard to guarantee stdio flush on exit
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
