use std::sync::{Arc, Mutex};
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

        info!("OpenCV build info: {}", build_info);
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

        // 创建一个 channel，缓冲区大小为 3 秒的帧数
        let (tx_frames, mut rx_frames) = mpsc::channel(3 * fps as usize);


        let mut c1 = closer.resubscribe();
        let mut c2 = closer.resubscribe();
        // 创建一个新线程来读取视频帧并发送到 channel
        let video_capture_thread = tokio::spawn(async move {
            loop {
                let mut frame = Mat::default();
                video.lock().unwrap().read(&mut frame).unwrap();
                select! {
                _ = c1.recv() => {
                    break;
                }
                res = tx_frames.send(Arc::new(Mutex::new(frame))) => {
                    match res {
                        Ok(_) => {}
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
                // 在这里调用我们的新的 process_frame 函数
                let result = self.process_frame(frame.clone());
                match result.await {
                    Ok(frame) => {
                        if let  Some(tx) = &tx {
                            // show the video stream in a window
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
        let mut mat = frame.lock().unwrap().clone();
        let result = self.model.run_video_frame(Arc::new(mat.clone()))?;
        return Ok(result.clone());
    }


    pub fn transform_frame_to_img(&self, frame: &Mat) -> Result<DynamicImage, Error> {
        let mut channel: opencv::core::Vector<opencv::core::Mat> = opencv::core::Vector::new();
        split(&frame, &mut channel)?;

        let data_b = channel.get(0)?;
        let data_g = channel.get(1)?;
        let data_r = channel.get(2)?;

        let mut data = Vec::with_capacity((frame.cols() * frame.rows() * 3) as usize);
        for y in 0..frame.rows() {
            for x in 0..frame.cols() {
                let b = *data_b.at_2d::<u8>(y, x)?;
                let g = *data_g.at_2d::<u8>(y, x)?;
                let r = *data_r.at_2d::<u8>(y, x)?;
                data.push(r);
                data.push(g);
                data.push(b);
            }
        }

        let img_buffer = image::ImageBuffer::from_raw(frame.cols() as u32, frame.rows() as u32, data)
            .ok_or_else(|| anyhow::anyhow!("can not create image from ndarray"))?;


        Ok(DynamicImage::ImageRgb8(img_buffer))
    }

    pub fn transform_img_to_frame(&self, img: &DynamicImage) -> Result<Mat, Error> {
        let rgb_img = img.to_rgb8();
        let (w, h) = rgb_img.dimensions();

        let mut r_c = vec![0; (w * h) as usize];
        let mut g_c = vec![0; (w * h) as usize];
        let mut b_c = vec![0; (w * h) as usize];

        for (x, y, pixel) in rgb_img.enumerate_pixels() {
            let index = (y * w + x) as usize;
            r_c[index] = pixel[0];
            g_c[index] = pixel[1];
            b_c[index] = pixel[2];
        }

        let r_mat = Mat::from_slice_2d(&r_c.chunks(w as usize).collect::<Vec<_>>())?;
        let g_mat = Mat::from_slice_2d(&g_c.chunks(w as usize).collect::<Vec<_>>())?;
        let b_mat = Mat::from_slice_2d(&b_c.chunks(w as usize).collect::<Vec<_>>())?;


        let mut v_frame = VectorOfMat::new();
        v_frame.push(b_mat);
        v_frame.push(g_mat);
        v_frame.push(r_mat);

        let mut frame = Mat::default();
        opencv::core::merge(&v_frame, &mut frame)?;
        Ok(frame)
    }
}