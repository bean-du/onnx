use std::collections::HashMap;
use std::fmt::{Display, Formatter, Write};
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
use crate::utils::file;
use chrono;
use chrono::Utc;
use crate::REPORT_INFERENCE_TOPIC;


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Data {
    #[serde(rename = "Header")]
    header: HashMap<String, String>,
    #[serde(rename = "Body")]
    body: String,
}

impl Data {
    pub fn new(header: HashMap<String, String>, body: String) -> Self {
        Self { header, body }
    }
}

impl Display for Data {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = serde_json::to_string(&self).map_err(|e| std::fmt::Error)?;
        f.write_str(&s)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportData {
    task_id: String,
    path: String,
    confidence: f32,
}

impl ReportData {
    pub fn new(task_id: String, confidence: f32) -> Self {
        Self { task_id, confidence, path: "".to_string() }
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
                let timer = sleep(Duration::from_millis(*i));
                select! {
                    _ = closer.recv() => {
                        break;
                    }
                    _ = timer => {
                        let c = nc.lock().await;

                        if let Some(data) = &*data.lock().await {
                            let res =  serde_json::to_string(&data);
                            let rd =  match res {
                                Ok(r) => r,
                                Err(e) => {
                                    error!("Serialize ReportData Failed: {:?}", e);
                                    continue;
                                }
                            };
                            let h = HashMap::new();


                            let rd = base64::encode(rd);
                            let data = Data::new(h, rd);

                            let data = data.to_string();
                            println!("Report Inference Result: {:?}", data);

                            match c.publish(REPORT_INFERENCE_TOPIC.to_string(), data.clone().into()).await {
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

    pub async fn report(&mut self, img: Arc<Mutex<Mat>>, mut data: ReportData) {
        let tmp = self.tmp.lock().await.take();
        info!("Report Inference Result: {:?}", data);
        match tmp {
            Some(t) => {
                if t.confidence < data.confidence {
                    let file_name = format!("{}_{}", data.task_id, Utc::now().timestamp());
                    let file_path = file::save_opencv_img(img, file_name).await.unwrap();
                    data.path = file_path;

                    *self.tmp.lock().await = Some(data);
                } else {
                    // do nothing
                }
            }
            None => {
                *self.tmp.lock().await = Some(data);
            }
        }
    }
}
