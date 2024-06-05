
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use nats;
use anyhow::{Error, Result};
use tracing::{info, debug};
use crate::{publish};
use crate::BUS;

const DEFAULT_NATS_ADDR: &'static str = "nats://127.0.0.1:4222";


#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, PartialOrd)]
pub struct Task {
    pub id: String,
    pub status: String,
    pub ttype: String,
    pub addr: String,
    pub model_info: String,
    pub detection_interval: i32,
    pub result_topic: String,
}

// 1. listen NATS's topic(use jetStream) and get task from NATS. save the task to the memory or database
pub fn listen_topic() -> Result<(), Error> {
    debug!("Listening to NATS topic... ");
    let nc = nats::connect(DEFAULT_NATS_ADDR)?;
    let sub = nc.subscribe("tasks")?;

    // loop to get the message from the topic
    for msg in sub.messages() {
        let task: Task = serde_json::from_slice(&msg.data)?;

        // send the task to event_bus
        publish!("task", task.clone());

        info!("Received task: {:?}", task);
    }
    Ok(())
}

pub fn mock_task() -> Task {
    Task {
        id: "1".to_string(),
        status: "running".to_string(),
        ttype: "yolo".to_string(),
        addr: "rtsp://192.168.1.168:8554/zlm/001".to_string(),
        // addr: "rtsp://192.168.2.202:8554/zlm/001".to_string(),
        // addr: "./test/qizai.mp4".to_string(),
        model_info: "yolov8".to_string(),
        detection_interval: 1,
        result_topic: "result".to_string(),
    }
}