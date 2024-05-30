use pineal::server;
use pineal::processor;
use pineal::tasks;
use pineal::yolo;
use tokio;
use pineal::subscribe;
use std::sync::{Arc, Mutex};
use std::any::Any;
use std::ops::Deref;
use anyhow::{anyhow, Result};
use opencv::{highgui, videoio, prelude::*};
use opencv::core::Mat;
use pineal::utils::logger;

use tracing::{info, error};
use tracing::instrument;
use pineal::tasks::mock_get_task;
use tokio::signal;
use tokio::sync::mpsc;
use pineal::BUS;
use pineal::processor::manager::ProcessorsManager;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _guard = logger::init("../logs".into())?;
    info!("Starting the pineal");
    // // 0. listen NATS's topic(jetStream) and get task from NATS. save the task to the memory or database
    // tasks::listen_topic()?;
    //
    // subscribe!("task", |data: Arc<dyn Any + Send + Sync> | {
    //     if let Some(task) = data.downcast_ref::<tasks::Task>() {
    //
    //          // 1. opencv get resources from camera or video file
    //         let video = processor::get_video()?;
    //
    //         // 2. decode the video frame by frame(use hardware acceleration)
    //         let frames = video_processing::decode_video(&video)?;
    //
    //         // 3. load the yolo model
    //         let model = yolo_model::load_model()?;
    //
    //         for frame in frames {
    //             // 4. detect the object in the frame
    //             let detections = yolo_model::detect_objects(&model, &frame)?;
    //
    //             // 5. draw the bounding box
    //             let frame_with_box = video_processing::draw_bounding_box(&frame, &detections)?;
    //
    //             // 6. show the frame with bounding box
    //             video_processing::show_frame(&frame_with_box)?;
    //
    //             // 7. save the frame with bounding box
    //             video_processing::save_frame(&frame_with_box)?;
    //
    //             // 8. save the video with bounding box
    //             let video_stream = video_processing::save_video(&frame_with_box)?;
    //
    //             // 9. return the result to NATS
    //             nats_comm::send_result(&detections)?;
    //
    //             // Add the video stream to the streaming server
    //             streaming_server::add_video_stream(&task.id, video_stream)?;
    //         }
    //     }
    // });

    let mut p = ProcessorsManager::new();

    let task = tasks::mock_get_task();

    let window = "Video AI";
    highgui::named_window(window, highgui::WINDOW_AUTOSIZE)?;

    let (tx, mut rx) = mpsc::channel::<Arc<Mutex<Mat>>>(100);


    p.add_task(task, Some(tx), false).await?;
    loop {
        tokio::select! {
            _ = signal::ctrl_c() => {
                info!("Received Ctrl-C signal");
                break;
            }
            
            frame = rx.recv() => {
                if let Some(f) = frame {
                    show_frame(f)?;
                }
            }
        }
    }

    info!("pineal application shutdown successfully");

    Ok(())
}

fn show_frame(frame: Arc<Mutex<Mat>>) -> Result<()> {
    let window_name = "Video AI";
    // highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE)?;

    let key = highgui::wait_key(1)?;
    if key > 0 && key != 255 {
        error!("Key is pressed");
        return Err(anyhow!("Key is pressed"));
    }
    let f = frame.lock().unwrap();
    highgui::imshow(window_name, f.deref())?;
    Ok(())
}