use std::sync::mpsc::{Receiver, SyncSender};
use std::thread::{self, JoinHandle};

pub struct GenericWorker<I, O> {
    pub sender: SyncSender<I>,
    pub receiver: Receiver<O>,
    _handle: JoinHandle<()>,
}

impl<I: Send + 'static, O: Send + 'static> GenericWorker<I, O> {
    pub fn spawn<F>(capacity: usize, mut work_fn: F) -> Self
    where
        F: FnMut(I) -> Option<O> + Send + 'static,
    {
        let (in_tx, in_rx) = std::sync::mpsc::sync_channel::<I>(capacity);
        let (out_tx, out_rx) = std::sync::mpsc::sync_channel::<O>(capacity);

        let handle = thread::spawn(move || {
            while let Ok(input) = in_rx.recv() {
                if let Some(output) = work_fn(input) {
                    if out_tx.send(output).is_err() {
                        break; // Receiver disconnected, shut down thread
                    }
                }
            }
        });

        Self {
            sender: in_tx,
            receiver: out_rx,
            _handle: handle,
        }
    }
}
