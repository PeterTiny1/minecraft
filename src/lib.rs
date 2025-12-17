// --- MODULES ---
// We make them public `pub mod` so they form the library's API
pub mod block;
pub mod camera;
pub mod chunk;
pub mod mesh_gen;
pub mod player;
pub mod ray;
pub mod renderer;
pub mod texture;
pub mod ui;
pub mod world_gen;

// --- IMPORTS ---
use itertools::Itertools;
use player::Player;
use std::{
    collections::{hash_map::Entry, HashMap}, // HashMap is used by ChunkDataStorage
    env,
    fs::File, // Needed for save_all_chunks
    path::Path,
    sync::mpsc,
    time::Instant,
};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, KeyEvent, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    keyboard::{Key, NamedKey},
    window::Window,
};

use chunk::{ChunkManager, CHUNK_DEPTH_I32, CHUNK_WIDTH_I32};
use vek::Vec3;

// --- CRATE-LEVEL PUBLIC EXPORTS ---
pub use block::BlockType;
pub use chunk::ChunkData; // Exporting this to fix HashMap type
pub use renderer::RenderContext;

// --- CONSTANTS ---
pub const RENDER_DISTANCE: f32 = 768.0;
pub const SEED: u32 = 0;
pub type ChunkDataStorage = HashMap<[i32; 2], chunk::ChunkData>;

// --- LOCAL STRUCTS ---

#[derive(Default)]
struct InputState {
    left_pressed: bool,
    right_pressed: bool,
}

// This is the "World" or "Orchestrator"
pub struct AppState<'a> {
    // Systems
    window: Option<&'static Window>,
    render_context: Option<renderer::RenderContext<'a>>,
    chunk_manager: ChunkManager,
    camera: Option<camera::Camera>,
    camera_controller: camera::PlayerController,
    ui: Option<ui::State>,
    player: Player,
    input: InputState,

    // State
    last_update_time: Instant,
    last_break_time: Instant,
    save_on_exit: bool,
}

impl AppState<'_> {
    #[must_use]
    pub fn new(save: bool) -> Self {
        let camera_controller = camera::PlayerController::new(10.0, 0.05);
        let player = Player::new(Vec3::new(0.0, 100.0, 0.0));
        let chunk_manager = ChunkManager::default();

        Self {
            window: None,
            render_context: None,
            ui: None,
            camera: None,

            chunk_manager,
            player,
            camera_controller,

            input: InputState::default(),
            last_update_time: Instant::now(),
            last_break_time: Instant::now(),
            save_on_exit: save,
        }
    }

    /// The main game logic update tick.
    /// This is called by `RedrawRequested` *after* all systems
    /// are confirmed to be initialized.
    fn update(&mut self, dt: std::time::Duration) {
        let camera = self.camera.as_mut().unwrap();
        let render_context = self.render_context.as_mut().unwrap();

        // --- 1. Lock Shared Data ---
        let generated_chunkdata = self.chunk_manager.generated_data.read().unwrap();

        // --- 2. Update Camera, Player, and Uniforms ---

        // A. Update camera rotation from mouse input
        // (This assumes `update_camera` is in `PlayerController` and applies the mouse delta)
        self.camera_controller.update_camera(&mut camera.data, dt);

        // B. Update player physics & get block player is looking at
        // (This assumes `update_physics` now takes the controller to get input)
        self.player.update_physics(
            dt.as_secs_f32(),
            &generated_chunkdata,
            &self.camera_controller,
            &camera.data, // Pass camera data for raycasting
        );

        // C. Camera follows player's new position
        camera.data.position = self.player.get_camera_position();

        // D. NOW update the uniforms with the final camera state
        render_context.uniforms.update_view_proj(camera);
        render_context.write_uniforms();
        drop(generated_chunkdata);
        let mut generated_chunkdata = self.chunk_manager.generated_data.write().unwrap();
        // --- 3. Update World (Chunk Loading) ---
        if let Some(chunk_location) = chunk::nearest_visible_unloaded(&generated_chunkdata, camera)
        {
            if let Entry::Vacant(entry) = generated_chunkdata.entry(chunk_location) {
                let location = &format!("{}.bin", chunk_location.iter().join(","));
                let path = Path::new(location);
                // Pass the noise generator from the chunk_manager
                self.chunk_manager.load_chunk(path, entry, chunk_location);
            }
        }

        // --- 4. Handle Block Breaking/Placing ---
        // (We now get `looking_at_block` from the player)
        if let Some((location, previous_step)) = self.player.looking_at_block {
            let now = Instant::now();

            if self.input.right_pressed {
                let location = location
                    + match previous_step {
                        0 => Vec3 { x: 1, y: 0, z: 0 },
                        1 => Vec3 { x: -1, y: 0, z: 0 },
                        2 => Vec3 { x: 0, y: 1, z: 0 },
                        3 => Vec3 { x: 0, y: -1, z: 0 },
                        4 => Vec3 { x: 0, y: 0, z: 1 },
                        5 => Vec3 { x: 0, y: 0, z: -1 },
                        _ => unreachable!(),
                    };
                if location.y < 256 && location.y > -1 {
                    let chunk_x = location.x.div_euclid(CHUNK_WIDTH_I32);
                    let chunk_z = location.z.div_euclid(CHUNK_DEPTH_I32);
                    if let Some(chunk) = generated_chunkdata.get_mut(&[chunk_x, chunk_z]) {
                        let local_x = location.x.rem_euclid(CHUNK_WIDTH_I32) as usize;
                        let local_z = location.z.rem_euclid(CHUNK_DEPTH_I32) as usize;
                        if chunk.contents[local_x][location.y as usize][local_z]
                            == block::BlockType::Air
                        {
                            chunk.contents[local_x][location.y as usize][local_z] =
                                block::BlockType::Stone; // Or your "held item"

                            // Send a remesh request
                            match self.chunk_manager.sender.try_send([chunk_x, chunk_z]) {
                                Ok(()) => {}
                                Err(mpsc::TrySendError::Disconnected(_)) => {
                                    panic!("Got disconnected!")
                                }
                                Err(mpsc::TrySendError::Full(_)) => todo!(),
                            }
                        }
                    }
                }
            } else if self.input.left_pressed && (now - self.last_break_time).as_millis() > 250 {
                self.last_break_time = now;
                let chunk_x = location.x.div_euclid(CHUNK_WIDTH_I32);
                let chunk_z = location.z.div_euclid(CHUNK_DEPTH_I32);
                let local_x = location.x.rem_euclid(CHUNK_WIDTH_I32) as usize;
                let local_z = location.z.rem_euclid(CHUNK_DEPTH_I32) as usize;

                if let Some(chunk) = generated_chunkdata.get_mut(&[chunk_x, chunk_z]) {
                    chunk.contents[local_x][location.y as usize][local_z] = block::BlockType::Air;

                    // Send a remesh request
                    match self.chunk_manager.sender.try_send([chunk_x, chunk_z]) {
                        Ok(()) => {}
                        Err(mpsc::TrySendError::Disconnected(_)) => panic!("Got disconnected!"),
                        Err(mpsc::TrySendError::Full(_)) => todo!(),
                    }
                }
            }
        }

        // --- 5. Finalize ---
        // (Drop the lock *before* inserting new meshes)
        drop(generated_chunkdata);
        self.chunk_manager.insert_chunk(render_context);
    }

    fn save_all_chunks(&self) {
        let generated_chunkdata = self.chunk_manager.generated_data.read().unwrap();
        for (chunk_location, data) in generated_chunkdata.iter() {
            let location = format!("{}.bin", chunk_location.iter().join(","));
            let path = Path::new(&location);
            if let Ok(mut file) = File::create(path) {
                bincode::encode_into_std_write(data, &mut file, bincode::config::standard())
                    .unwrap();
            }
        }
    }
}

// We make the ApplicationHandler public so main.rs can use it
impl ApplicationHandler for AppState<'_> {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = Box::leak(Box::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Blockcraft")
                        .with_fullscreen(Some(winit::window::Fullscreen::Borderless(None))),
                )
                .unwrap(),
        ));
        window
            .set_cursor_grab(winit::window::CursorGrabMode::Confined)
            .unwrap();
        window.set_cursor_visible(false);
        self.window = Some(window);

        let size = window.inner_size();
        let mut render_context = renderer::RenderContext::new(window, size);

        let camera_data = camera::CameraData::new(
            self.player.position.into_tuple(), // Start camera at player's head
            -45.0_f32.to_radians(),
            -20.0_f32.to_radians(),
        );
        let projection = camera::Projection::new(
            size.width,
            size.height,
            90.0_f32.to_radians(),
            0.05,
            RENDER_DISTANCE,
        );
        let camera = camera::Camera {
            data: camera_data,
            projection,
        };

        render_context.uniforms.update_view_proj(&camera);
        self.ui = Some(ui::init_state(&render_context, size));
        self.render_context = Some(render_context);
        self.camera = Some(camera);
        self.last_update_time = Instant::now(); // Reset update timer
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        // --- These events DON'T need the app to be fully initialized ---
        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        ..
                    },
                ..
            } => {
                if self.save_on_exit {
                    // Call the new, correct method
                    self.save_all_chunks();
                }
                event_loop.exit();
            }

            WindowEvent::KeyboardInput { event, .. } => {
                self.camera_controller
                    .process_keyboard(event.physical_key, event.state.is_pressed());
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
                };
                self.camera_controller.process_scroll(scroll);
            }

            WindowEvent::MouseInput { state, button, .. } => match button {
                winit::event::MouseButton::Left => self.input.left_pressed = state.is_pressed(),
                winit::event::MouseButton::Right => self.input.right_pressed = state.is_pressed(),
                _ => {}
            },

            // --- These events DO need the app to be fully initialized ---
            WindowEvent::Resized(new_size) => {
                // Only resize if all systems are ready
                if let (Some(render_context), Some(camera)) =
                    (self.render_context.as_mut(), self.camera.as_mut())
                {
                    if new_size.height > 0 && new_size.width > 0 {
                        render_context.resize(new_size);
                        camera.projection.resize(new_size.width, new_size.height);
                        // ui.resize(new_size, &render_context.queue); // TODO: Implement ui.resize
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                let dt = now - self.last_update_time;
                self.last_update_time = now;

                // --- 1. CALL UPDATE FIRST ---
                // Update *all* state. This will borrow `self` mutably,
                // but the borrow ends immediately.
                self.update(dt);

                // --- 2. THEN, DO THE BORROWS FOR RENDER ---
                // This is a *new* set of borrows, which is fine.
                if let (Some(render_context), Some(camera), Some(ui)) = (
                    self.render_context.as_mut(),
                    self.camera.as_mut(),
                    self.ui.as_mut(),
                ) {
                    // Now, we just pass the borrowed values to render
                    match render_context.render(&self.chunk_manager, camera, ui) {
                        Ok(()) => {}
                        Err(wgpu::SurfaceError::Lost) => render_context.resize(render_context.size),
                        Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                        Err(e) => eprintln!("{e:?}"),
                    }
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            // This just updates the controller's internal state
            self.camera_controller.process_mouse(delta.0, delta.1);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(window) = self.window {
            window.request_redraw();
        }
    }
}

/// # Errors
///
/// Will return Err if something goes wrong
///
/// # Panics
///
/// Will panic if there somehow isn't a first argument
pub fn run() -> Result<(), impl std::error::Error> {
    env_logger::init();
    let mut save = false;
    let mut args = env::args();
    let _path = args.next().unwrap();
    if let Some(arg) = args.next() {
        match &*arg {
            "-save" | "-s" => save = true,
            _ => println!("Invalid argument {arg}!"),
        }
    }
    let event_loop = EventLoop::new().unwrap();

    let mut state = AppState::new(save);
    event_loop.run_app(&mut state)
}
