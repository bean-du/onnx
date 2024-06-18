use std::path::PathBuf;
use std::sync::{Arc};
use tokio::sync::Mutex;
use anyhow::{anyhow, Error};
use image::{DynamicImage, GenericImageView};
use ndarray::{Array, IxDyn};
use opencv::core::{have_opencl, Mat, MatTrait, MatTraitConst, split};
use opencv::hub_prelude::{VideoCaptureTrait, VideoCaptureTraitConst};
use opencv::types::VectorOfMat;
use opencv::videoio;
use opencv::{imgproc, prelude::*};
use opencv::videoio::VideoCapture;
use tokio::select;
use tokio::sync::{broadcast, mpsc};
use tokio::sync::mpsc::Sender;
use tracing::{error, info, debug};
use crate::tasks::Task;
use crate::yolo::cli::default_args;
use crate::yolo::model::YOLOv8;
use crate::yolo::{check_font, SKELETON};
use rayon::prelude::*;
use tokio::time::sleep;
use crate::client::nats::get_nats_client;
use crate::report::{Report, ReportData};
use crate::yolo::ort::BoundingBox;


// VideoProcessor is responsible for processing video streams
// It can show the video stream in a window, or use GPU to process the video stream

// And it supports video file and video streaming
#[derive(Debug)]
pub struct VideoProcessor {
    pub window_tx: Option<Sender<Arc<Mutex<Mat>>>>,
    // notify is used to stop the video processing thread
    pub notify: broadcast::Sender<String>,
    pub model: YOLOv8,
    pub video_writer: Option<Arc<Mutex<videoio::VideoWriter>>>,
    pub path: PathBuf,
    pub task: Task,
}

impl VideoProcessor {
    pub fn new(window_tx: Option<Sender<Arc<Mutex<Mat>>>>, task: Task, notify: broadcast::Sender<String>) -> anyhow::Result<Self> {
        let model = YOLOv8::new(default_args())?;
        let path = PathBuf::from(task.output_addr.clone());
        info!("Init VideoProcessor for task: {:?} successfully", task.id);

        Ok(Self {
            window_tx,
            notify,
            model,
            path,
            task,
            video_writer: None,
        })
    }

    pub fn update_task(&mut self, task: Task) {
        self.task = task;
    }


    pub async fn run(&mut self, task: Task, mut closer: broadcast::Receiver<String>) -> anyhow::Result<()> {
        let build_info = opencv::core::get_build_information().unwrap();
        info!("opencl enabled: {}", have_opencl().unwrap());

        // info!("OpenCV build info: {}", build_info);
        // Open the video stream
        let mut video = if task.stream_addr.is_empty() {
            videoio::VideoCapture::new(0, videoio::CAP_ANY)?
        } else {
            videoio::VideoCapture::from_file(&task.stream_addr, videoio::CAP_ANY)?
        };

        // Check if the video stream is opened
        let opened = videoio::VideoCapture::is_opened(&video)?;
        if !opened {
            error!("Unable to open video streaming!");
            return Err(anyhow!("Unable to open video streaming!"));
        }

        // Get the video FPS
        let fps = video.get(videoio::CAP_PROP_FPS)?;
        let frame_size = opencv::core::Size {
            width: video.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32,
            height: video.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32,
        };

        // Create a video writer
        let fourcc = videoio::VideoWriter::fourcc('M', 'J', 'P', 'G')?;
        let output_file = format!("{}_output.avi", task.id);

        let video_writer = Some(Arc::new(Mutex::new(videoio::VideoWriter::new(&output_file, fourcc, fps, frame_size, false)?)));
        self.video_writer = video_writer.clone();
        let mut c1 = closer.resubscribe();
        let mut c2 = closer.resubscribe();

        let tx = self.window_tx.clone();
        let video = Arc::new(Mutex::new(video));

        let (tx_frames, mut rx_frames) = mpsc::channel(1);

        let interval = Arc::new(Mutex::new(self.task.detection_interval));
        let interval_clone = Arc::clone(&interval);

        let mut report = Report::new(get_nats_client().await?, task.detection_interval);
        report.run(closer.resubscribe()).await;
        let report_clone = Arc::new(Mutex::new(report));

        // 创建一个新线程来读取视频帧并发送到 channel
        tokio::spawn(async move {
            loop {
                let mut frame = Mat::default();
                video.lock().await.read(&mut frame).unwrap();
                select! {
                    _ = c1.recv() => {
                        info!("Receive Stop signal. Stop video capture thread");
                        break;
                    }
                    res = tx_frames.send(Arc::new(Mutex::new(frame))) => {
                        match res {
                            Ok(_) => {
                                debug!("Send frame to channel success time: {:?} ", chrono::Local::now().time())
                            }
                            Err(e) => {
                                error!("Send frame to channel error: {}", e);
                                break;
                            }
                        }
                    }
                }
            }
        });

        loop {
            select! {
                _ = c2.recv() => {
                    info!("Receive Stop signal. Stop video process thread");
                    break;
                }
                Some(frame) = rx_frames.recv() => {
                    debug!("Frame buffer lens: {:?}", rx_frames.len());
                    let mut mat = frame.lock().await.clone();
                    let process_result = self.model.process_frame(Arc::new(mat.clone()));
                    match process_result {
                        Ok(res) => {
                            let f = self.model.plot(res.clone(), &mut mat)?;
                            let img = Arc::new(Mutex::new(f));

                            if let  Some(tx) = &tx {
                                // send the video stream to the window channel
                                tx.send(Arc::clone(&img)).await?;
                            }

                            // info!("Inference result: {:?}", res);
                            if res.len() == 0 {
                                continue;
                            }
                            let confidence = res[0].2;
                            // report the inference result
                            let mut r = report_clone.lock().await;
                            let mut rd = ReportData::new(task.id.clone(), confidence);
                            // info!("Report Inference Result: {:?}", rd);
                            r.report(Arc::clone(&img), rd).await;
                        }
                        Err(e) => {
                            error!("process_frame Error: {}", e);
                            return Err(e);
                        }
                    }
                    continue;
                }
            }
        }
        Ok(())
    }
}