use std::collections::HashMap;
use std::sync::{Arc};
use tokio::sync::Mutex;
use opencv::core::Mat;
use tokio::sync::mpsc::Sender;
use tokio::sync::{broadcast, mpsc, RwLock};
use tracing::error;
use crate::processor::video::{VideoProcessor};
use crate::tasks::Task;


pub struct ProcessorsManager {
    // key is the task id
    pub video_processor: Arc<RwLock<HashMap<String, VideoProcessor>>>,
}

impl ProcessorsManager {
    pub fn new() -> Self {
        Self {
            video_processor: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    // Add a new task to the processor
    // enable_window: whether to show the window
    // enable_gpu: whether to use GPU
    pub async fn add_task(&mut self, task: Task, window_tx: Option<Sender<Arc<Mutex<Mat>>>>, enable_gpu: bool) -> anyhow::Result<()> {
        let (notify_tx, notify_rx) = broadcast::channel::<String>(1);
        let video_processor = VideoProcessor::new(window_tx, enable_gpu, task.clone(), notify_tx)?;

        // Add task to  HashMap
        {
            let mut vp_map = self.video_processor.write().await;
            vp_map.insert(task.id.clone(), video_processor);
        }

        // Start a new thread to run task VideoProcessor
        let video_processor = self.video_processor.clone();
        tokio::spawn(async move {
            if let Some(vp) = video_processor.write().await.get_mut(&task.id) {
                if let Err(e) = vp.run(task, notify_rx).await {
                    error!("Error running video processor: {}", e);
                }
            }
        });

        Ok(())
    }

    // Remove a task from the processor
    // task_id: the id of the task to be removed
    pub async fn remove_task(&mut self, task_id: String) -> anyhow::Result<()> {
        // Remove task from HashMap
        let vp_option = {
            let mut vp_map = self.video_processor.write().await;
            vp_map.remove(&task_id)
        };

        // Send a signal to stop the task
        if let Some(vp) = vp_option {
            vp.notify.send("stop".to_string())?;
            // Delay for a while to ensure the loop has time to exit
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }

        Ok(())
    }
}