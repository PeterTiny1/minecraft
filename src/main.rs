use minecraft::AppState;
use winit::event_loop::EventLoop;

// 2. The main function is now just a launcher.
fn main() -> Result<(), impl std::error::Error> {
    env_logger::init();

    // Parse args (this is an "application" concern)
    let mut save = false;
    let mut args = std::env::args();
    let _path = args.next().unwrap();
    if let Some(arg) = args.next() {
        match &*arg {
            "-save" | "-s" => save = true,
            _ => println!("Invalid argument {arg}!"),
        }
    }

    // 3. Initialize the event loop (this is an "application" concern)
    let event_loop = EventLoop::new().unwrap();

    // 4. Create the state from the library
    let mut state = AppState::new(save);

    // 5. Run the app
    event_loop.run_app(&mut state)
}
