// Crate modules
mod block;
mod camera;
mod chunk;
mod input;
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
pub use renderer::RenderContext;

// Imports
use std::{env, path::Path, sync::Arc, time::Instant};

use vek::Vec3;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, KeyEvent, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    keyboard::{Key, NamedKey},
    window::Window,
};

use chunk::ChunkManager;
use player::Player;

use crate::{input::InputState, renderer::RenderOutcome};

#[derive(Debug)]
pub enum EngineError {
    EventLoop(winit::error::EventLoopError),
    Io(std::io::Error),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EventLoop(e) => write!(f, "Event loop error: {e}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for EngineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::EventLoop(err) => Some(err),
            Self::Io(err) => Some(err),
        }
    }
}

impl From<winit::error::EventLoopError> for EngineError {
    fn from(err: winit::error::EventLoopError) -> Self {
        Self::EventLoop(err)
    }
}

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

pub struct RunningState {
    window: Arc<Window>,
    render_context: renderer::RenderContext,
    camera: camera::Camera,
    ui: ui::State,

    chunk_manager: ChunkManager,
    player: Player,
    input: InputState,

    last_update_time: Instant,
}

impl RunningState {
    #[must_use]
    pub fn new(window: Arc<Window>, mut render_context: renderer::RenderContext) -> Self {
        let size = window.inner_size();
        let player = Player::new(PLAYER_START_POS, 10.0, 0.002);
        let chunk_manager = ChunkManager::default();
        let camera_data = camera::CameraData::new(
            PLAYER_START_POS.into_tuple(),
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

            input: InputState::default(),
            last_update_time: Instant::now(),
        }
    }

    /// The main game logic update tick.
/// The main game logic update tick.
    #[tracing::instrument(skip(self))]
    fn update(&mut self, dt: std::time::Duration) {
        let dt_secs = dt.as_secs_f32().min(0.1);

        // 1. Update player physics & camera rotation using InputState
        self.player.update_physics(
            dt_secs,
            &self.chunk_manager.generated_data,
            &self.input,
            &mut self.camera.data,
        );

        // 2. Sync camera position to player eye position & update rendering matrices
        self.camera.data.position = self.player.get_camera_position();
        self.render_context.uniforms.update_view_proj(&self.camera);
        self.render_context.write_uniforms();

        // 3. Block interaction (Break/Place)
        self.player.update_blocks(&self.input, &mut self.chunk_manager);

        // 4. Chunk Loading & Meshing
        if let Some(chunk_loc) =
            chunk::nearest_visible_unloaded(&self.chunk_manager.generated_data, &self.camera)
        {
            let path_str = format!("{},{}.bin", chunk_loc[0], chunk_loc[1]);
            tracing::trace!(chunk_loc = ?chunk_loc, "Queueing visible chunk");

            let _center_arc = self
                .chunk_manager
                .load_or_generate_chunk_arc(Path::new(&path_str), chunk_loc);

            let world_data = &self.chunk_manager.generated_data;
            let [chunk_x, chunk_z] = chunk_loc;

            // Queue mesh jobs for center chunk and surrounding neighbors
            self.chunk_manager.queue_mesh_job(world_data, chunk_loc);
            self.chunk_manager.queue_mesh_job(world_data, [chunk_x - 1, chunk_z]);
            self.chunk_manager.queue_mesh_job(world_data, [chunk_x + 1, chunk_z]);
            self.chunk_manager.queue_mesh_job(world_data, [chunk_x, chunk_z - 1]);
            self.chunk_manager.queue_mesh_job(world_data, [chunk_x, chunk_z + 1]);

            self.chunk_manager.queue_mesh_job(world_data, [chunk_x - 1, chunk_z - 1]);
            self.chunk_manager.queue_mesh_job(world_data, [chunk_x + 1, chunk_z + 1]);
            self.chunk_manager.queue_mesh_job(world_data, [chunk_x - 1, chunk_z + 1]);
            self.chunk_manager.queue_mesh_job(world_data, [chunk_x + 1, chunk_z - 1]);
        }

        self.chunk_manager.insert_chunk(&self.render_context);

        // 5. Reset frame-based input deltas (mouse dx/dy, scroll wheel)
        self.input.end_frame();
    }

    #[tracing::instrument(skip(self))]
    fn save_all_chunks(&self) {
        let save_dir = Path::new("saves");
        if let Err(e) = std::fs::create_dir_all(save_dir) {
            tracing::error!(error = %e, "Failed to create saves directory");
            return;
        }

        let generated_chunkdata = &self.chunk_manager.generated_data;
        let mut saved_count = 0;

        for (chunk_location, data) in generated_chunkdata {
            let file_path =
                save_dir.join(format!("{},{}.bin", chunk_location[0], chunk_location[1]));

            // Handle serialization errors without crashing
            let bytes = match rkyv::to_bytes::<rkyv::rancor::Error>(data) {
                Ok(bytes) => bytes,
                Err(e) => {
                    tracing::error!(
                        chunk_location = ?chunk_location,
                        error = %e,
                        "Failed to serialize chunk"
                    );
                    continue;
                }
            };

            // std::fs::write creates/truncates and writes in one step
            if let Err(e) = std::fs::write(&file_path, &bytes) {
                tracing::error!(
                    chunk_location = ?chunk_location,
                    path = %file_path.display(),
                    error = %e,
                    "Failed to write chunk file"
                );
            } else {
                saved_count += 1;
            }
        }

        tracing::info!(
            saved_count,
            total = generated_chunkdata.len(),
            "Finished saving chunks"
        );
    }
}

impl ApplicationHandler for App {
    #[tracing::instrument(skip(self, event_loop))]
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.state.is_none() {
            let attrs = Window::default_attributes()
                .with_title("Blockcraft")
                .with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));

            let window = match event_loop.create_window(attrs) {
                Ok(w) => Arc::new(w),
                Err(e) => {
                    tracing::error!(error = ?e, "Failed to create window");
                    event_loop.exit();
                    return;
                }
            };

            if window
                .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                .or_else(|_| window.set_cursor_grab(winit::window::CursorGrabMode::Locked))
                .is_err()
            {
                tracing::warn!("Failed to grab cursor");
            }
            window.set_cursor_visible(false);

            let size = window.inner_size();
            let render_context =
                match pollster::block_on(renderer::RenderContext::new(window.clone(), size)) {
                    Ok(ctx) => ctx,
                    Err(e) => {
                        tracing::error!(error = %e, "Failed to initialize render context");
                        event_loop.exit();
                        return;
                    }
                };
            self.state = Some(RunningState::new(window, render_context));
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
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
                    state.save_all_chunks();
                }
                event_loop.exit();
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let Some(state) = &mut self.state {
                    state
                        .input
                        .process_keyboard(event.physical_key, event.state.is_pressed());
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
                };
                if let Some(state) = &mut self.state {
                    state.input.scroll_delta += scroll;
                }
            }

            WindowEvent::MouseInput {
                state: button_state,
                button,
                ..
            } => {
                if let Some(state) = &mut self.state {
                    state
                        .input
                        .process_mouse_button(button, button_state.is_pressed());
                }
            }

            WindowEvent::Resized(new_size) => {
                if let Some(state) = &mut self.state
                    && new_size.height > 0
                    && new_size.width > 0
                {
                    state.render_context.resize(new_size);
                    state
                        .camera
                        .projection
                        .resize(new_size.width, new_size.height);
                }
            }

            WindowEvent::RedrawRequested => {
                if let Some(state) = &mut self.state {
                    let now = std::time::Instant::now();
                    let dt = now - state.last_update_time;
                    state.last_update_time = now;

                    state.update(dt);

                    if state
                        .render_context
                        .render(&state.chunk_manager, &state.camera, &state.ui)
                        == RenderOutcome::NeedsResize
                    {
                        state.render_context.resize(state.render_context.size);
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
        if let DeviceEvent::MouseMotion { delta } = event
            && let Some(state) = &mut self.state
        {
            #[allow(clippy::cast_possible_truncation)]
            state
                .input
                .accumulate_mouse_motion(delta.0 as f32, delta.1 as f32);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        self.state = None;
    }
}

pub fn run() -> Result<(), EngineError> {
    let mut save = false;

    for arg in env::args().skip(1) {
        match arg.as_str() {
            "-save" | "-s" => save = true,
            _ => tracing::warn!(arg = %arg, "Unrecognized command-line argument"),
        }
    }

    let event_loop = EventLoop::new()?;
    let mut app = App::new(save);

    event_loop.run_app(&mut app)?;
    Ok(())
}
