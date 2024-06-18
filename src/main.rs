use {
    pineal::{
        server, processor, tasks, yolo,
        utils::logger,
        client::nats::get_nats_client,
        tasks::{listen_topic, mock_task, Task},
        processor::manager::ProcessorsManager,
    },
    std::{sync::{Arc}, any::Any, future::Future, ops::Deref},
    tokio::{sync::Mutex, sync::mpsc, signal},
    opencv::{highgui, videoio, prelude::*, core::Mat},
    anyhow::{anyhow, Result},
    async_nats::jetstream,
    tracing::{info, error, instrument},
    flamegraph,
};
use pineal::tasks::init_task;


#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _guard = logger::init("../logs".into())?;
    info!("Starting the pineal");

    // 0. listen NATS's topic(jetStream) and get task from NATS. save the task to the memory or database
    // 1. opencv get resources from camera or video file
    // 2. decode the video frame by frame(use hardware acceleration)
    // 3. load the yolo model
    // 4. detect the object in the frame
    // 5. draw the bounding box
    // 6. show the frame with bounding box
    // 7. save the frame with bounding box
    // 8. save the video with bounding box
    // 9. return the result to NATS
    // 10.Push the video stream to the streaming server


    // test add task
    // // todo: remove this code if deploy as production
    // tokio::spawn(async {
    //     tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    //     let task = mock_task();
    //     let task_str = serde_json::to_string(&task).unwrap();
    //     // let client = async_nats::connect(DEFAULT_NATS_ADDR).await.unwrap();
    //     let client = get_nats_client().await.unwrap();
    //     let js = jetstream::new(client);
    //     let kv = js.create_key_value(jetstream::kv::Config {
    //         bucket: pineal::TASK_BUCKET.to_string(),
    //         ..Default::default()
    //     }).await.unwrap();
    //
    //     let key = format!("{}.{}", pineal::TASK_TOPIC_PREFIX, task.id);
    //     kv.put(key.clone(), task_str.clone().into()).await.unwrap();
    //     info!("Put task: {:?}", task_str);
    //
    //     tokio::time::sleep(tokio::time::Duration::from_secs(180)).await;
    //     kv.delete(key).await.unwrap();
    //     info!("Delete task")
    // });


    let (tx, mut rx) = mpsc::channel::<Arc<Mutex<Mat>>>(100);
    let pm = Arc::new(Mutex::new(ProcessorsManager::new()));


    // init_task(Arc::clone(&pm), Some(tx.clone())).await?;


    // open a new thread to listen the NATS topic
    tokio::spawn(async {
        match listen_topic(pm, Some(tx)).await {
            Ok(_) => info!("Listen topic successfully"),
            Err(e) => error!("Listen topic failed: {:?}", e),
        }
    });

    loop {
        tokio::select! {
            _ = signal::ctrl_c() => {
                info!("Received Ctrl-C signal");
                break;
            }
            
            frame = rx.recv() => {
                if let Some(f) = frame {
                    show_frame(f).await?;

                }
            }
        }
    }

    info!("pineal application shutdown successfully");

    Ok(())
}


// todo: implement the multiple-video stream window to show
async fn show_frame(frame: Arc<Mutex<Mat>>) -> Result<()> {
    let window_name = "Video AI";
    highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE)?;

    let key = highgui::wait_key(1)?;
    if key > 0 && key != 255 {
        error!("Key is pressed");
        return Err(anyhow!("Key is pressed"));
    }

    let f = frame.lock().await;
    highgui::imshow(window_name, f.deref())?;
    Ok(())
}