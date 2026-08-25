use winit::event::MouseButton;
use winit::keyboard::{KeyCode, PhysicalKey};

#[derive(Debug, Default)]
pub struct InputState {
    // Movement Intent (Keys Held)
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,

    // Mouse Actions
    pub left_click_held: bool,
    pub right_click_held: bool,
    pub left_click_just_pressed: bool,
    pub right_click_just_pressed: bool,

    // Mouse Motion Deltas (reset every frame)
    pub mouse_dx: f32,
    pub mouse_dy: f32,
    pub scroll_delta: f32,
}

impl InputState {
    pub fn accumulate_mouse_motion(&mut self, dx: f32, dy: f32) {
        self.mouse_dx += dx;
        self.mouse_dy += dy;
    }

    /// Call at the end of every frame to reset one-shot deltas/events
    pub fn end_frame(&mut self) {
        self.mouse_dx = 0.0;
        self.mouse_dy = 0.0;
        self.scroll_delta = 0.0;
        self.left_click_just_pressed = false;
        self.right_click_just_pressed = false;
    }

    pub fn process_keyboard(&mut self, key: PhysicalKey, pressed: bool) -> bool {
        let PhysicalKey::Code(code) = key else {
            return false;
        };
        match code {
            KeyCode::KeyW | KeyCode::ArrowUp => self.forward = pressed,
            KeyCode::KeyS | KeyCode::ArrowDown => self.backward = pressed,
            KeyCode::KeyA | KeyCode::ArrowLeft => self.left = pressed,
            KeyCode::KeyD | KeyCode::ArrowRight => self.right = pressed,
            KeyCode::Space => self.up = pressed,
            KeyCode::ShiftLeft => self.down = pressed,
            _ => return false,
        }
        true
    }

    pub fn process_mouse_button(&mut self, button: MouseButton, pressed: bool) {
        match button {
            MouseButton::Left => {
                if pressed && !self.left_click_held {
                    self.left_click_just_pressed = true;
                }
                self.left_click_held = pressed;
            }
            MouseButton::Right => {
                if pressed && !self.right_click_held {
                    self.right_click_just_pressed = true;
                }
                self.right_click_held = pressed;
            }
            _ => {}
        }
    }
}
