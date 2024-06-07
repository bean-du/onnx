use std::fmt::{Display, Formatter};
use std::sync::{Arc};
use async_nats::Client;
use anyhow::{Result, Error};
use log::info;
use opencv::core::Mat;
use serde::{Deserialize, Serialize};
use tokio::{
    select,
    sync::{broadcast::Receiver, Mutex},
    time::{Duration, sleep},
};
use tracing::error;


const REPORT_TOPIC: &str = "IntelliBox.Report";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportData {
    task_id: String,
    img_path: String,
    confidence: f32,
}

impl ReportData {
    pub fn new(task_id: String, confidence: f32) -> Self {
        Self { task_id, confidence, img_path: "".to_string() }
    }
}


#[derive(Debug)]
pub struct Report {
    client: Client,
    interval: u64,
    tmp: Arc<Mutex<Option<ReportData>>>,
}

impl Report {
    pub fn new(client: Client, interval: u64) -> Self {
        Self {
            client,
            interval,
            tmp: Arc::new(Mutex::new(None)),
        }
    }

    pub async fn run(&mut self, mut closer: Receiver<String>) {
        let interval = Arc::new(Mutex::new(self.interval));
        let nc = Arc::new(Mutex::new(self.client.clone()));
        let data = Arc::clone(&self.tmp);

        tokio::spawn(async move {
            info!("Result Report Start");
            loop {
                let i = interval.lock().await;
                let timer = sleep(Duration::from_secs(*i));
                select! {
                    _ = closer.recv() => {
                        break;
                    }
                    _ = timer => {
                        let c = nc.lock().await;

                        if let Some(data) = &*data.lock().await {
                            let rd = match serde_json::to_string(&data) {
                                Ok(r) => r,
                                Err(e) => {
                                    error!("Serialize ReportData Failed: {:?}", e);
                                    continue;
                                }
                            };

                            match c.publish(REPORT_TOPIC, rd.into()).await {
                                Ok(_) => {
                                    info!("Report Inference Result to Nats Success: {:?}", data)
                                }
                                Err(e) => {
                                    error!("Report Inference Result to Nats Failed: {:?}", e)
                                }
                            }
                        }

                    }
                }
            }
        });
    }

    pub async fn report(&mut self, img: Arc<Mutex<Mat>>, data: ReportData) {
        let tmp = self.tmp.lock().await.take();
        info!("Report Inference Result: {:?}", data);
        match tmp {
            Some(t) => {
                if t.confidence < data.confidence {
                    *self.tmp.lock().await = Some(data);
                } else {
                    *self.tmp.lock().await = Some(t);
                }
            }
            None => {
                *self.tmp.lock().await = Some(data);
            }
        }
    }
}
