use crate::world::ChunkData;
use std::path::Path;
use std::sync::Arc;

/// Saves a collection of chunks to the target directory.
pub fn save_chunks<'a, I>(chunks: I, save_dir: &Path)
where
    I: IntoIterator<Item = (&'a [i32; 2], &'a Arc<ChunkData>)>,
{
    if let Err(e) = std::fs::create_dir_all(save_dir) {
        tracing::error!(error = %e, "Failed to create saves directory");
        return;
    }

    let mut saved_count = 0;
    let mut total = 0;

    for (chunk_location, data) in chunks {
        total += 1;
        let file_path = save_dir.join(format!("{},{}.bin", chunk_location[0], chunk_location[1]));

        // Dereference Arc to get &ChunkData for rkyv serialization
        let bytes = match rkyv::to_bytes::<rkyv::rancor::Error>(data.as_ref()) {
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

    tracing::info!(saved_count, total, "Finished saving chunks");
}

pub fn save_single_chunk(chunk_location: [i32; 2], data: &ChunkData, save_dir: &Path) {
    if let Err(e) = std::fs::create_dir_all(save_dir) {
        tracing::error!(error = %e, "Failed to create saves directory");
        return;
    }

    let file_path = save_dir.join(format!("{},{}.bin", chunk_location[0], chunk_location[1]));

    let bytes = match rkyv::to_bytes::<rkyv::rancor::Error>(data) {
        Ok(bytes) => bytes,
        Err(e) => {
            tracing::error!(
                chunk_location = ?chunk_location,
                error = %e,
                "Failed to serialize chunk"
            );
            return;
        }
    };

    if let Err(e) = std::fs::write(&file_path, &bytes) {
        tracing::error!(
            chunk_location = ?chunk_location,
            path = %file_path.display(),
            error = %e,
            "Failed to write chunk file"
        );
    }
}
