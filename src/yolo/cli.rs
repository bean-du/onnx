use clap::Parser;

use crate::yolo::YOLOTask;

#[derive(Parser, Clone, Default)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// ONNX model path
    #[arg(long, required = true)]
    pub model: String,

    /// input path
    #[arg(long, required = true)]
    pub source: String,

    /// device id
    #[arg(long, default_value_t = 0)]
    pub device_id: u32,

    /// using TensorRT EP
    #[arg(long)]
    pub trt: bool,

    /// using CUDA EP
    #[arg(long)]
    pub cuda: bool,

    /// using open_vino EP
    #[arg(long)]
    pub open_vino: bool,

    /// using core_ml EP
    #[arg(long)]
    pub core_ml: bool,

    /// input batch size
    #[arg(long, default_value_t = 1)]
    pub batch: u32,

    /// trt input min_batch size
    #[arg(long, default_value_t = 1)]
    pub batch_min: u32,

    /// trt input max_batch size
    #[arg(long, default_value_t = 32)]
    pub batch_max: u32,

    /// using TensorRT --fp16
    #[arg(long)]
    pub fp16: bool,

    /// specify YOLO task
    #[arg(long, value_enum)]
    pub task: Option<YOLOTask>,

    /// num_classes
    #[arg(long)]
    pub nc: Option<u32>,

    /// num_keypoints
    #[arg(long)]
    pub nk: Option<u32>,

    /// num_masks
    #[arg(long)]
    pub nm: Option<u32>,

    /// input image width
    #[arg(long)]
    pub width: Option<u32>,

    /// input image height
    #[arg(long)]
    pub height: Option<u32>,

    /// confidence threshold
    #[arg(long, required = false, default_value_t = 0.3)]
    pub conf: f32,

    /// iou threshold in NMS
    #[arg(long, required = false, default_value_t = 0.45)]
    pub iou: f32,

    /// confidence threshold of keypoint
    #[arg(long, required = false, default_value_t = 0.55)]
    pub kconf: f32,

    /// plot inference result and save
    #[arg(long)]
    pub plot: bool,

    /// check time consumed in each stage
    #[arg(long)]
    pub profile: bool,
}

pub fn default_args() -> Args {
    Args {
        model: "./yolov8n.onnx".to_string(),
        device_id: 0,
        batch: 1,
        task: Some(YOLOTask::Detect),
        cuda: false,
        open_vino: false,
        core_ml: true,
        profile: true,
        ..Args::default()
    }
}