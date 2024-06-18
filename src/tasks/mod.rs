use serde::{Serialize, Deserialize};
use std::sync::{Arc};
use tokio::sync::Mutex;
use async_nats;
use anyhow::{Error, Result};
use async_nats::connection::ShouldFlush::No;
use tracing::{info, debug};
use async_nats::jetstream;
use clap::builder::Str;
use futures::{StreamExt, TryStreamExt};
use opencv::core::Mat;
use tokio::sync::mpsc::Sender;
use crate::client::nats::get_nats_client;
use crate::processor::manager::ProcessorsManager;
use crate::TASK_BUCKET;
use crate::TASK_TOPIC_PREFIX;


// TaskType is the type of the task
enum TaskType {
    Detect,
    Track,
    Count,
    Recognize,
    Segment,
    Classify,
    Pose,
    Other,
}

impl From<&str> for TaskType {
    fn from(s: &str) -> Self {
        match s {
            "Detect" => TaskType::Detect,
            "Track" => TaskType::Track,
            "Count" => TaskType::Count,
            "Recognize" => TaskType::Recognize,
            "Segment" => TaskType::Segment,
            "Classify" => TaskType::Classify,
            "Pose" => TaskType::Pose,
            _ => TaskType::Other,
        }
    }
}

impl From<TaskType> for String {
    fn from(t: TaskType) -> Self {
        match t {
            TaskType::Detect => "Detect".to_string(),
            TaskType::Track => "Track".to_string(),
            TaskType::Count => "Count".to_string(),
            TaskType::Recognize => "Recognize".to_string(),
            TaskType::Segment => "Segment".to_string(),
            TaskType::Classify => "Classify".to_string(),
            TaskType::Pose => "Pose".to_string(),
            TaskType::Other => "Other".to_string(),
        }
    }
}

type Confidence = f32;


#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq, Eq, PartialOrd)]
pub struct Task {
    pub id: String,
    // pub status: String,
    pub model_type: String,
    pub task_type: String,
    pub stream_addr: String,
    pub model_addr: String,
    pub model_info: ModelInfo,
    pub output_type: String,
    pub output_addr: String,
    pub confidence: i32,
    pub detection_interval: u64,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq, Eq, PartialOrd)]
pub struct ModelInfo {
    pub inputs: String,
    pub outputs: String,
}

pub async fn init_task(pm: Arc<Mutex<ProcessorsManager>>,window_tx: Option<Sender<Arc<Mutex<Mat>>>>) -> Result<()> {
    let client = get_nats_client().await?;
    let js = jetstream::new(client);
    let kv = js.create_key_value(jetstream::kv::Config {
        bucket: TASK_BUCKET.to_string(),
        ..Default::default()
    }).await?;

    // Watch the task topic
    let key = format!("{}>", TASK_TOPIC_PREFIX.to_string());

    let mut pm = pm.lock().await;

    let mut histories = kv.history(key).await?;

    while let Some(entry) = histories.next().await {
        println!("entry: {:?}", entry);
    }


    Ok(())
}

// 1. listen NATS's topic(use jetStream) and get task from NATS. save the task to the memory or database
pub async fn listen_topic(pm: Arc<Mutex<ProcessorsManager>>, window_tx: Option<Sender<Arc<Mutex<Mat>>>>) -> Result<(), Error> {
    debug!("Init Nats Connection");
    let client = get_nats_client().await?;
    let js = jetstream::new(client);
    let kv = js.create_key_value(jetstream::kv::Config {
        bucket: TASK_BUCKET.to_string(),
        ..Default::default()
    }).await?;

    // Watch the task topic
    let watch_key = format!("{}>", TASK_TOPIC_PREFIX.to_string());
    info!("Watch Task Topic: {}", watch_key);

    let mut watcher = kv.watch(watch_key).await?;

    while let Ok(entry) = watcher.next().await.ok_or(anyhow::anyhow!("watcher error"))? {
        let key = entry.key;
        let revision = entry.revision;
        let operation = entry.operation;

        let mut pm = pm.lock().await;
        match operation {
            jetstream::kv::Operation::Put => {
                let task: Task = serde_json::from_slice(&entry.value)?;
                info!("Receive Task key: {:?}, revision: {:?}, (op: {:?}) \n Task: {:?}", key, revision, operation, task);
                pm.create_task(task, window_tx.clone()).await?
            }
            jetstream::kv::Operation::Delete => {
                info!("Received Delete task: {:?}", key);
                let val = key.replace(TASK_TOPIC_PREFIX, "");
                pm.remove_task(val).await?;
            }
            jetstream::kv::Operation::Purge => {
                info!("Received Purge task: {:?}", key);
                let val = key.replace(TASK_TOPIC_PREFIX, "");
                pm.remove_task(val).await?;
            }
        }

        info!("key: {:?}, revision: {:?}, (op: {:?})", key, revision, operation);
    }

    Ok(())
}

pub fn mock_task() -> Task {
    Task {
        id: format!("{}", "1xxx".to_string()),
        model_type: "yolo".to_string(),
        task_type: "Detect".to_string(),
        // addr: "rtsp://192.168.1.168:8554/zlm/001".to_string(),
        // addr: "rtsp://192.168.2.202:8554/zlm/001".to_string(),
        stream_addr: "./test/qizai.mp4".to_string(),
        model_info: ModelInfo::default(),
        output_addr: "./runs".to_string(),
        detection_interval: 1,
        ..Default::default()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::env;
    use std::io::Bytes;
    use std::str::from_utf8;
    use async_nats::jetstream;
    use futures::{StreamExt, TryStreamExt};
    use tokio::sync::Mutex;
    use crate::utils::logger;
    use tokio::time::Instant;
    use crate::processor::manager::ProcessorsManager;

    #[tokio::test]
    async fn test_listen_topic(pm: Arc<Mutex<ProcessorsManager>>) {
        let _guard = logger::init("logs".into()).unwrap();

        tokio::spawn(async {
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            let task = mock_task();
            let task_str = serde_json::to_string(&task).unwrap();
            // let client = async_nats::connect(DEFAULT_NATS_ADDR).await.unwrap();
            let client = get_nats_client().await.unwrap();
            let js = jetstream::new(client);
            let kv = js.create_key_value(jetstream::kv::Config {
                bucket: TASK_BUCKET.to_string(),
                ..Default::default()
            }).await.unwrap();
            kv.put(TASK_TOPIC_PREFIX.to_string(), task_str.clone().into()).await.unwrap();
            info!("Put task: {:?}", task_str);

            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            kv.delete(TASK_TOPIC_PREFIX.to_string()).await.unwrap();
            info!("Delete task")
        });

        listen_topic().await.unwrap()
    }
}