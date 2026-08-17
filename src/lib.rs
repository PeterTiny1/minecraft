// Crate modules
mod block;
mod camera;
mod chunk;
mod mesh_gen;
mod player;
mod ray;
mod renderer;
mod texture;
mod ui;
mod world_gen;

// Public API re-exports
pub use block::BlockType;
pub use chunk::ChunkData;
use pollster::block_on;
pub use renderer::RenderContext;

// Imports
use std::{env, fs::File, path::Path, sync::Arc, time::Instant};

use vek::Vec3;
use winit::{
    application::ApplicationHandler,
    error::EventLoopError,
    event::{DeviceEvent, KeyEvent, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    keyboard::{Key, NamedKey},
    window::Window,
};

use chunk::{
    CHUNK_DEPTH, CHUNK_DEPTH_I32, CHUNK_HEIGHT, CHUNK_WIDTH, CHUNK_WIDTH_I32, ChunkManager,
};
use player::Player;

// --- CONSTANTS ---
pub const RENDER_DISTANCE: f32 = 768.0;
pub const SEED: u32 = 0;

pub const DIRECTION_OFFSETS: [Vec3<i32>; 6] = [
    Vec3 { x: -1, y: 0, z: 0 },
    Vec3 { x: 1, y: 0, z: 0 },
    Vec3 { x: 0, y: -1, z: 0 },
    Vec3 { x: 0, y: 1, z: 0 },
    Vec3 { x: 0, y: 0, z: -1 },
    Vec3 { x: 0, y: 0, z: 1 },
];
pub const PLAYER_START_POS: Vec3<f32> = Vec3::new(0.0, 100.0, 0.0);

// --- LOCAL STRUCTS ---

#[derive(Default)]
struct InputState {
    left_pressed: bool,
    right_pressed: bool,
}

// Pure game state that exists before the window opens
#[derive(Default)]
pub struct App {
    state: Option<RunningState>,
    save_on_exit: bool,
}

impl App {
    const fn new(save_on_exit: bool) -> Self {
        Self {
            state: None,
            save_on_exit,
        }
    }
}

// Systems that ONLY exist once the GPU & Window are active
pub struct RunningState {
    window: Arc<Window>,
    render_context: renderer::RenderContext, // No lifetime needed!
    camera: camera::Camera,
    camera_controller: camera::PlayerController,
    ui: ui::State,

    // World / Game Data
    chunk_manager: ChunkManager,
    player: Player,
    input: InputState,

    // Timers
    last_update_time: Instant,
    last_break_time: Instant,
}

impl RunningState {
    #[must_use]
    pub fn new(window: Arc<Window>, mut render_context: renderer::RenderContext) -> Self {
        let size = window.inner_size();
        let player = Player::new(PLAYER_START_POS);
        let chunk_manager = ChunkManager::default();
        let camera_controller = camera::PlayerController::new(10.0, 0.05);
        let camera_data = camera::CameraData::new(
            PLAYER_START_POS.into_tuple(), // Start camera at player's head
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
        render_context.write_uniforms();

        Self {
            window,
            ui: ui::init_state(&render_context, size),
            render_context,
            camera,

            chunk_manager,
            player,
            camera_controller,

            input: InputState::default(),
            last_update_time: Instant::now(),
            last_break_time: Instant::now(),
        }
    }

    /// The main game logic update tick.
    /// This is called by `RedrawRequested` *after* all systems
    /// are confirmed to be initialized.
    fn update(&mut self, dt: std::time::Duration) {
        // --- 1. Physics & Camera (Requires Read Lock) ---
        {
            let world_data = &self.chunk_manager.generated_data;
            self.camera_controller
                .update_camera(&mut self.camera.data, dt);
            self.player.update_physics(
                dt.as_secs_f32(),
                world_data,
                &self.camera_controller,
                &self.camera.data,
            );
        } // Read lock drops here automatically!

        self.camera.data.position = self.player.get_camera_position();
        self.render_context.uniforms.update_view_proj(&self.camera);
        self.render_context.write_uniforms();

        // A. Chunk Loading
        if let Some(chunk_loc) =
            chunk::nearest_visible_unloaded(&self.chunk_manager.generated_data, &self.camera)
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
        self.chunk_manager.insert_chunk(&self.render_context);
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
impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.state.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(
                        Window::default_attributes()
                            .with_title("Blockcraft")
                            .with_fullscreen(Some(winit::window::Fullscreen::Borderless(None))),
                    )
                    .unwrap(),
            );
            if window
                .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                .or_else(|_| window.set_cursor_grab(winit::window::CursorGrabMode::Locked))
                .is_err()
            {
                eprintln!("Warning: Failed to grab cursor");
            }
            window.set_cursor_visible(false);

            let size = window.inner_size();
            let render_context = block_on(renderer::RenderContext::new(window.clone(), size));

            self.state = Some(RunningState::new(window, render_context));
        }
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
                if self.save_on_exit
                    && let Some(state) = &self.state
                {
                    // Call the new, correct method
                    state.save_all_chunks();
                }
                event_loop.exit();
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let Some(state) = &mut self.state {
                    state
                        .camera_controller
                        .process_keyboard(event.physical_key, event.state.is_pressed());
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
                };
                if let Some(state) = &mut self.state {
                    state.camera_controller.process_scroll(scroll);
                }
            }

            WindowEvent::MouseInput {
                state: button_state,
                button,
                ..
            } => {
                if let Some(state) = &mut self.state {
                    match button {
                        winit::event::MouseButton::Left => {
                            state.input.left_pressed = button_state.is_pressed()
                        }
                        winit::event::MouseButton::Right => {
                            state.input.right_pressed = button_state.is_pressed()
                        }
                        _ => {}
                    }
                }
            }

            // --- These events DO need the app to be fully initialized ---
            WindowEvent::Resized(new_size) => {
                // Only resize if all systems are ready
                if let Some(state) = &mut self.state
                    && new_size.height > 0
                    && new_size.width > 0
                {
                    state.render_context.resize(new_size);
                    state
                        .camera
                        .projection
                        .resize(new_size.width, new_size.height);
                    // ui.resize(new_size, &render_context.queue); // TODO: Implement ui.resize
                }
            }

            WindowEvent::RedrawRequested => {
                if let Some(state) = &mut self.state {
                    let now = std::time::Instant::now();
                    let dt = now - state.last_update_time;
                    state.last_update_time = now;

                    state.update(dt);

                    // Now, we just pass the borrowed values to render
                    match state.render_context.render(
                        &state.chunk_manager,
                        &state.camera,
                        &state.ui,
                    ) {
                        Ok(()) => {}
                        Err(wgpu::SurfaceError::Lost) => {
                            state.render_context.resize(state.render_context.size)
                        }
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
            if let Some(state) = &mut self.state {
                state.camera_controller.process_mouse(delta.0, delta.1);
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
    fn exiting(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        // Forces wgpu buffers and context to drop while Wayland display is still active
        self.state = None;
    }
}

pub fn run() -> Result<(), EventLoopError> {
    let _ = env_logger::try_init();

    let mut save = false;

    for arg in env::args().skip(1) {
        match arg.as_str() {
            "-save" | "-s" => save = true,
            _ => log::warn!("Unrecognized argument '{arg}'"),
        }
    }

    let event_loop = EventLoop::new()?;
    let mut app = App::new(save);
    
    event_loop.run_app(&mut app)
}
