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
use tracing::{error, info};
use crate::tasks::Task;
use crate::yolo::cli::default_args;
use crate::yolo::model::YOLOv8;
use crate::yolo::{check_font, SKELETON};
use rayon::prelude::*;


// VideoProcessor is responsible for processing video streams
// It can show the video stream in a window, or use GPU to process the video stream
// And it supports video file and video streaming
pub(crate) struct VideoProcessor {
    pub window_tx: Option<Sender<Arc<Mutex<Mat>>>>,
    pub enable_gpu: bool,
    pub notify: broadcast::Sender<String>,
    pub model: YOLOv8,
    pub video_writer: Option<Arc<Mutex<videoio::VideoWriter>>>,
}

impl VideoProcessor {
    pub fn new(window_tx: Option<Sender<Arc<Mutex<Mat>>>>, enable_gpu: bool, task: Task, notify: broadcast::Sender<String>) -> anyhow::Result<Self> {
        let model = YOLOv8::new(default_args())?;
        Ok(Self {
            window_tx,
            enable_gpu,
            notify,
            model,
            video_writer: None,
        })
    }

    pub async fn run(&mut self, task: Task, mut closer: broadcast::Receiver<String>) -> anyhow::Result<()> {
        let build_info = opencv::core::get_build_information().unwrap();
        info!("opencl enabled: {}", have_opencl().unwrap());

        // info!("OpenCV build info: {}", build_info);
        // Open the video stream
        let mut video = if task.addr.is_empty() {
            videoio::VideoCapture::new(0, videoio::CAP_ANY)?
        } else {
            videoio::VideoCapture::from_file(&task.addr, videoio::CAP_ANY)?
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
        let is_color = true;
        let video_writer = Some(Arc::new(Mutex::new(videoio::VideoWriter::new(&output_file, fourcc, fps, frame_size, is_color)?)));
        self.video_writer = video_writer.clone();

        let tx = self.window_tx.clone();

        let video = Arc::new(Mutex::new(video));

        let (tx_frames, mut rx_frames) = mpsc::channel(1);


        let mut c1 = closer.resubscribe();
        let mut c2 = closer.resubscribe();
        // 创建一个新线程来读取视频帧并发送到 channel
        let video_capture_thread = tokio::spawn(async move {
            loop {
                let mut frame = Mat::default();
                video.lock().await.read(&mut frame).unwrap();
                select! {
                    _ = c1.recv() => {
                        break;
                    }
                    res = tx_frames.send(Arc::new(Mutex::new(frame))) => {
                        match res {
                            Ok(_) => {
                                info!("Send frame to channel success time: {:?} ", chrono::Local::now().time())
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
                    break;
                }
                Some(frame) = rx_frames.recv() => {
                    info!("Frame buffer lens: {:?}", rx_frames.len());
                    let result = self.process_frame(frame.clone());
                    match result.await {
                        Ok(frame) => {
                            if let  Some(tx) = &tx {
                                // send the video stream to the window channel
                                let f = Arc::new(Mutex::new(frame.clone()));
                                tx.send(f).await?;

                            }
                        }
                        Err(e) => {
                            error!("Error: {}", e);
                            return Err(e);
                        }
                    }
                    continue;
                }
            }
        }
        Ok(())
    }

    pub async fn process_frame(&mut self, frame: Arc<Mutex<Mat>>) -> anyhow::Result<Mat> {
        let mut mat = frame.lock().await.clone();
        let result = self.model.run_video_frame(Arc::new(mat.clone()))?;
        return Ok(result.clone());
    }
}