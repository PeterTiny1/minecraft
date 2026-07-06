pub(crate) mod block;
pub(crate) mod camera;
pub(crate) mod chunk;
pub(crate) mod mesh_gen;
pub(crate) mod player;
pub(crate) mod ray;
pub(crate) mod renderer;
pub(crate) mod texture;
pub(crate) mod ui;
pub(crate) mod world_gen;

// --- IMPORTS ---
use player::Player;
use std::{
    collections::HashMap, // HashMap is used by ChunkDataStorage
    env,
    fs::File, // Needed for save_all_chunks
    path::Path,
    sync::Arc,
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

use crate::chunk::{CHUNK_DEPTH, CHUNK_HEIGHT, CHUNK_WIDTH};

// --- CONSTANTS ---
pub const RENDER_DISTANCE: f32 = 768.0;
pub const SEED: u32 = 0;
pub type ChunkDataStorage = HashMap<[i32; 2], chunk::ChunkData>;

pub const DIRECTION_OFFSETS: [Vec3<i32>; 6] = [
    Vec3 { x: -1, y: 0, z: 0 },
    Vec3 { x: 1, y: 0, z: 0 },
    Vec3 { x: 0, y: -1, z: 0 },
    Vec3 { x: 0, y: 1, z: 0 },
    Vec3 { x: 0, y: 0, z: -1 },
    Vec3 { x: 0, y: 0, z: 1 },
];

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

        // --- 1. Physics & Camera (Requires Read Lock) ---
        {
            let world_data = &self.chunk_manager.generated_data;
            self.camera_controller.update_camera(&mut camera.data, dt);
            self.player.update_physics(
                dt.as_secs_f32(),
                world_data,
                &self.camera_controller,
                &camera.data,
            );
        } // Read lock drops here automatically!

        camera.data.position = self.player.get_camera_position();
        render_context.uniforms.update_view_proj(camera);
        render_context.write_uniforms();

        // A. Chunk Loading
        if let Some(chunk_loc) =
            chunk::nearest_visible_unloaded(&self.chunk_manager.generated_data, camera)
        {
            let path_str = format!("{},{}.bin", chunk_loc[0], chunk_loc[1]);

            // 1. Kick off generation/loading internally
            let _center_arc = self
                .chunk_manager
                .load_and_insert_chunk(Path::new(&path_str), chunk_loc);

            let world_data = &self.chunk_manager.generated_data;
            let [chunk_x, chunk_z] = chunk_loc;
            // 2. Queue up the mesh job using our fresh Arc handle
            // and the now-unlocked map reference
            self.chunk_manager.queue_mesh_job(world_data, chunk_loc);
            self.chunk_manager
                .queue_mesh_job(world_data, [chunk_x - 1, chunk_z]);
            self.chunk_manager
                .queue_mesh_job(world_data, [chunk_x + 1, chunk_z]);
            self.chunk_manager
                .queue_mesh_job(world_data, [chunk_x, chunk_z - 1]);
            self.chunk_manager
                .queue_mesh_job(world_data, [chunk_x, chunk_z + 1]);

            // Diagonal corner seams
            self.chunk_manager
                .queue_mesh_job(world_data, [chunk_x - 1, chunk_z - 1]);
            self.chunk_manager
                .queue_mesh_job(world_data, [chunk_x + 1, chunk_z + 1]);
            self.chunk_manager
                .queue_mesh_job(world_data, [chunk_x - 1, chunk_z + 1]);
            self.chunk_manager
                .queue_mesh_job(world_data, [chunk_x + 1, chunk_z - 1]);
        }

        // B. Block Interaction (Breaking / Placing)
        if let Some((location, previous_step)) = self.player.get_looking_at() {
            let now = Instant::now();
            let is_place = self.input.right_pressed;
            let is_break =
                self.input.left_pressed && (now - self.last_break_time).as_millis() > 250;

            if is_place || is_break {
                // Calculate target block position
                let target_pos = if is_place {
                    location - DIRECTION_OFFSETS[previous_step]
                } else {
                    self.last_break_time = now;
                    location
                };

                if target_pos.y >= 0 && target_pos.y < CHUNK_HEIGHT as i32 {
                    let chunk_x = target_pos.x.div_euclid(CHUNK_WIDTH_I32);
                    let chunk_z = target_pos.z.div_euclid(CHUNK_DEPTH_I32);
                    let chunk_loc = [chunk_x, chunk_z];

                    if let Some(chunk_arc) = self.chunk_manager.generated_data.get_mut(&chunk_loc) {
                        let local_x = target_pos.x.rem_euclid(CHUNK_WIDTH_I32) as usize;
                        let local_z = target_pos.z.rem_euclid(CHUNK_DEPTH_I32) as usize;
                        let local_y = target_pos.y as usize;

                        // Safely modify using Copy-On-Write via Arc::make_mut
                        let chunk = Arc::make_mut(chunk_arc);
                        let current_block = chunk.contents[local_x][local_y][local_z];

                        let new_block = if is_place && current_block == block::BlockType::Air {
                            Some(block::BlockType::Stone)
                        } else if is_break {
                            Some(block::BlockType::Air)
                        } else {
                            None
                        };

                        if let Some(block) = new_block {
                            chunk.contents[local_x][local_y][local_z] = block;

                            let world_data = &self.chunk_manager.generated_data;
                            // Remesh the modified center chunk
                            self.chunk_manager.queue_mesh_job(world_data, chunk_loc);

                            // Remesh adjacent neighbor chunks if the block was on a boundary seam
                            if local_x == 0 {
                                self.chunk_manager
                                    .queue_mesh_job(world_data, [chunk_x - 1, chunk_z]);
                            }
                            if local_x == CHUNK_WIDTH - 1 {
                                self.chunk_manager
                                    .queue_mesh_job(world_data, [chunk_x + 1, chunk_z]);
                            }
                            if local_z == 0 {
                                self.chunk_manager
                                    .queue_mesh_job(world_data, [chunk_x, chunk_z - 1]);
                            }
                            if local_z == CHUNK_DEPTH - 1 {
                                self.chunk_manager
                                    .queue_mesh_job(world_data, [chunk_x, chunk_z + 1]);
                            }

                            // Diagonal corner seams
                            if local_x == 0 && local_z == 0 {
                                self.chunk_manager
                                    .queue_mesh_job(world_data, [chunk_x - 1, chunk_z - 1]);
                            }
                            if local_x == CHUNK_WIDTH - 1 && local_z == CHUNK_DEPTH - 1 {
                                self.chunk_manager
                                    .queue_mesh_job(world_data, [chunk_x + 1, chunk_z + 1]);
                            }
                            if local_x == 0 && local_z == CHUNK_DEPTH - 1 {
                                self.chunk_manager
                                    .queue_mesh_job(world_data, [chunk_x - 1, chunk_z + 1]);
                            }
                            if local_x == CHUNK_WIDTH - 1 && local_z == 0 {
                                self.chunk_manager
                                    .queue_mesh_job(world_data, [chunk_x + 1, chunk_z - 1]);
                            }
                        }
                    }
                }
            }
        }
        self.chunk_manager.insert_chunk(render_context);
    }

    fn save_all_chunks(&self) {
        let generated_chunkdata = &self.chunk_manager.generated_data;
        for (chunk_location, data) in generated_chunkdata {
            let location = format!(
                "{}.bin",
                chunk_location
                    .iter()
                    .map(i32::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            );
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
