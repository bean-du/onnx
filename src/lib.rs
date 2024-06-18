pub mod yolo;
pub mod server;
pub mod processor;
pub mod tasks;
pub mod utils;
pub mod convert;
pub mod stream;
pub mod client;
pub mod report;
pub mod device;

use once_cell::sync::Lazy;
use tokio::sync::Mutex;
use anyhow::{ Result, Error };
//
pub const REPORT_INFERENCE_TOPIC: &str = "IntelliBox.Inference.Report";

// task bucket and topic
pub const TASK_BUCKET: &'static str = "IntelliBoxTaskBucket";
pub const TASK_TOPIC_PREFIX: &'static str = "IntelliBox.Task.Dispatch.";



static BASE_FILE_PATH: Lazy<Mutex<Option<String>>> = Lazy::new(|| Mutex::new(None));

pub async fn get_base_file_dir() -> Result<String> {
    let mut base_file_path = BASE_FILE_PATH.lock().await;
    if base_file_path.is_none() {
        let base_path = option_env!("BASE_FILE_DIR").unwrap_or("/tmp/IntelliBox/Data").to_string();
        *base_file_path = Some(base_path);
    }
    Ok(base_file_path.as_ref().unwrap().clone())
}