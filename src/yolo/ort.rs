use anyhow::Result;
use clap::ValueEnum;
use ndarray::{Array, Array2, Array4, ArrayBase, Axis, Dim, Ix4, IxDyn, IxDynImpl, OwnedRepr, s};
use opencv::core::{CV_32FC1, Mat, MatTrait, MatTraitConst, MatTraitConstManual, split};
use opencv::{
    imgproc::{
        self,
        COLOR_BGR2RGB,
    }
};
use ort_2::{
    self,
    ExecutionProviderDispatch,
    TensorRTExecutionProvider,
    OpenVINOExecutionProvider,
    CoreMLExecutionProvider,
    CPUExecutionProvider,
    CUDAExecutionProvider,
    ExecutionProvider,
    ValueType,
    Session,
    Tensor,
    Value,
    inputs,
};
use regex::Regex;
use tracing::info;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum YOLOTask {
    Classify,
    Detect,
    Pose,
    Segment,
}


#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum OrtEP {
    Cpu,
    Cuda(i32),
    Trt(i32),
    OpenVino,
    CoreML,
}


#[derive(Debug)]
pub struct OrtInputs {
    // ONNX model inputs attrs
    pub shapes: Vec<Vec<i32>>,
    pub date_types: Vec<ValueType>,
    pub names: Vec<String>,
    pub sizes: Vec<Vec<u32>>,
}


impl OrtInputs {
    pub fn new(session: &Session) -> Self {
        let mut shapes = Vec::new();
        let mut date_types = Vec::new();
        let mut names = Vec::new();
        for i in session.inputs.iter() {
            if let Some(tensor_dimensions) = i.input_type.tensor_dimensions() {
                let shape: Vec<i32> = tensor_dimensions.iter().map(|&x| x as i32).collect();
                shapes.push(shape);
            } else {
                shapes.push(vec![-1i32]);
            }

            date_types.push(i.input_type.clone());
            names.push(i.name.clone());
        }


        Self {
            shapes,
            date_types,
            names,
            sizes: Vec::new(),
        }
    }
}

pub struct OrtConfig {
    pub model_file: String,
    pub task: Option<YOLOTask>,
    pub ep: OrtEP,
    pub image_size: (Option<u32>, Option<u32>),
}


#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

#[rustfmt::skip]
const YOLOV8_CLASS_LABELS: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];


#[derive(Debug)]
pub struct OrtBackend {
    session: Session,
    task: YOLOTask,
    ep: ExecutionProviderDispatch,
    inputs: OrtInputs,
}


impl OrtBackend {
    pub fn build(config: OrtConfig) -> Result<Self> {
        let ep = ep(&config)?;

        ort_2::init().
            with_execution_providers([ep.clone()])
            .commit()?;

        info!("Init Env With Execution Provider: {:?} Success", ep.as_str());

        let session = Session::builder()?
            .commit_from_file(config.model_file.clone())?;

        info!("Load Model File: {:?} Success", config.model_file);

        let mut inputs = OrtInputs::new(&session);

        // input size: height and width
        let height = if inputs.shapes[0][2] == -1 {
            match config.image_size.0 {
                Some(height) => height,
                None => panic!("Failed to get model height. Make it explicit with `--height`"),
            }
        } else {
            inputs.shapes[0][2] as u32
        };
        let width = if inputs.shapes[0][3] == -1 {
            match config.image_size.1 {
                Some(width) => width,
                None => panic!("Failed to get model width. Make it explicit with `--width`"),
            }
        } else {
            inputs.shapes[0][3] as u32
        };
        inputs.sizes.push(vec![height, width]);


        // task: using given one or guessing
        let task = match config.task {
            Some(task) => task,
            None => match session.metadata() {
                Err(_) => panic!("No metadata found. Try making it explicit by `--task`"),
                Ok(metadata) => match metadata.custom("task") {
                    Err(_) => panic!("Can not get custom value. Try making it explicit by `--task`"),
                    Ok(value) => match value {
                        None => panic!("No corresponding value of `task` found in metadata. Make it explicit by `--task`"),
                        Some(task) => match task.as_str() {
                            "classify" => YOLOTask::Classify,
                            "detect" => YOLOTask::Detect,
                            "pose" => YOLOTask::Pose,
                            "segment" => YOLOTask::Segment,
                            x => todo!("{:?} is not supported for now!", x),
                        },
                    },
                },
            },
        };


        info!("Init Onnx Runtime Backend Success");
        Ok(Self {
            session,
            inputs,
            task,
            ep,
        })
    }


    pub fn run(&self, input: Tensor<f32>) -> Result<Array<f32, IxDyn>> {
        let outputs = self.session.run(inputs![input]?)?;
        let output = outputs["output0"].try_extract_tensor::<f32>()?.t().into_owned();

        Ok(output)
    }

    pub fn pre_process(&self, img: &Mat) -> Result<ArrayBase<OwnedRepr<f32>, Ix4>> {
        let mut resized_frame = Mat::default();
        // resize image to 640 * 640
        imgproc::resize(
            &img,
            &mut resized_frame,
            opencv::core::Size::new(640 as i32, 640 as i32),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        let mut dst = Mat::default();
        imgproc::cvt_color(&resized_frame, &mut dst, COLOR_BGR2RGB, 0).unwrap();

        let mut converted_mat = Mat::default();
        dst.convert_to(&mut converted_mat, CV_32FC1, 1.0, 0.0)?;

        let mut channels: opencv::core::Vector<Mat> = opencv::core::Vector::new();
        split(&converted_mat, &mut channels)?;

        let b = channels.get(0)?.data_typed::<f32>()?.to_vec();
        let g = channels.get(1)?.data_typed::<f32>()?.to_vec();
        let r = channels.get(2)?.data_typed::<f32>()?.to_vec();


        let mut model_input = Array4::<f32>::zeros((1, 3, 640, 640));
        model_input.slice_mut(s![0, 0, .., ..]).assign(&Array2::from_shape_vec((640, 640), b)?);
        model_input.slice_mut(s![0, 1, .., ..]).assign(&Array2::from_shape_vec((640, 640), g)?);
        model_input.slice_mut(s![0, 2, .., ..]).assign(&Array2::from_shape_vec((640, 640), r)?);

        model_input /= 255.0;

        Ok(model_input)
    }

    fn intersection(&self, box1: &BoundingBox, box2: &BoundingBox) -> f32 {
        (box1.x2.min(box2.x2) - box1.x1.max(box2.x1)) * (box1.y2.min(box2.y2) - box1.y1.max(box2.y1))
    }

    fn union(&self, box1: &BoundingBox, box2: &BoundingBox) -> f32 {
        ((box1.x2 - box1.x1) * (box1.y2 - box1.y1)) + ((box2.x2 - box2.x1) * (box2.y2 - box2.y1)) - self.intersection(box1, box2)
    }

    pub fn plot(&self, results: Vec<(BoundingBox, &str, f32)>, img: &mut Mat) -> Result<Mat> {
        let mut img = img.clone();
        // Draw bounding boxes
        let red_color = opencv::core::Scalar::new(0., 0., 255.0, 0.);
        for bb in results {
            let x = if bb.0.x1 < 0. { 0 as f32 } else { bb.0.x1  };
            let y = if bb.0.y1 < 30. { 30  as f32} else { bb.0.y1 };

            let w = bb.0.x2 - x;
            let h = bb.0.y2 - y;
            imgproc::rectangle(
                &mut img,
                opencv::core::Rect::new(
                    x as i32, // x center
                    y as i32, // y center
                    w as i32,
                    h as i32,
                ),
                red_color,
                2,
                opencv::imgproc::LINE_8,
                0,
            )?;
            // put label and prob info to image
            let text = format!("{}: {:.2}", bb.1, bb.2);
            let point = opencv::core::Point::new(bb.0.x1 as i32, bb.0.y1 as i32);
            imgproc::put_text(
                &mut img,
                &text,
                point,
                imgproc::INTER_LINEAR,
                2.0,
                opencv::core::Scalar::new(0., 255., 0., 0.),
                2,
                imgproc::LINE_8,
                false,
            )?;
        }
        Ok(img)
    }

    pub fn postprocess<'a>(
        &self,
        output: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
        original_img_width: f32,
        original_img_height: f32,
        img_wh: f32,
        accepted_prob: f32, // 0 - 1
    ) -> Vec<(BoundingBox, &'a str, f32)> {
        let fx = original_img_width / img_wh;
        let fy = original_img_height / img_wh;

        let mut boxes = Vec::new();
        let output = output.slice(s![.., .., 0]);
        for row in output.axis_iter(Axis(0)) {
            let row: Vec<f32> = row.iter().copied().collect();

            let (class_id, prob) = row
                .iter()
                .skip(4)
                .enumerate()
                .map(|(index, value)| (index, *value))
                .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
                .unwrap();

            if prob < accepted_prob {
                continue;
            }

            let label = YOLOV8_CLASS_LABELS[class_id];
            let xc = row[0] * fx;
            let yc = row[1] * fy;
            let w = row[2] * fx;
            let h = row[3] * fy;

            boxes.push((
                BoundingBox {
                    x1: xc - w / 2.,
                    y1: yc - h / 2.,
                    x2: xc + w / 2.,
                    y2: yc + h / 2.,
                },
                label,
                prob,
            ));
        }

        let mut bb_results = Vec::new();
        boxes.sort_by(|box1, box2| box2.2.total_cmp(&box1.2));
        while !boxes.is_empty() {
            let f = boxes[0];
            bb_results.push(f);
            boxes.retain(|box1| self.intersection(&f.0, &box1.0) / self.union(&f.0, &box1.0) < 0.7);
        }

        bb_results
    }

    pub fn names(&self) -> Option<Vec<String>> {
        // class names, metadata parsing
        // String format: `{0: 'person', 1: 'bicycle', 2: 'sports ball', ..., 27: "yellow_lady's_slipper"}`
        match self.fetch_from_metadata("names") {
            Some(names) => {
                let re = Regex::new(r#"(['"])([-()\w '"]+)(['"])"#).unwrap();
                let mut names_ = vec![];
                for (_, [_, name, _]) in re.captures_iter(&names).map(|x| x.extract()) {
                    names_.push(name.to_string());
                }
                Some(names_)
            }
            None => None,
        }
    }

    pub fn fetch_from_metadata(&self, key: &str) -> Option<String> {
        // fetch value from onnx model file by key
        match self.session.metadata() {
            Err(_) => None,
            Ok(metadata) => metadata.custom(key).unwrap_or_else(|_| None),
        }
    }

    pub fn output_shapes(&self) -> Vec<Vec<i32>> {
        let mut shapes = Vec::new();
        for o in &self.session.outputs {
            if let Some(dims) = o.output_type.tensor_dimensions() {
                let shape: Vec<i32> = dims.iter().map(|&x| x as i32).collect();
                shapes.push(shape);
            } else {
                shapes.push(vec![-1]);
            }
        }
        shapes
    }

    pub fn num_class(&self) -> Option<u32> {
        match self.names() {
            // by names
            Some(names) => Some(names.len() as u32),
            None => match self.task {
                // by task calculation
                YOLOTask::Classify => Some(self.output_shapes()[0][1] as u32),
                YOLOTask::Detect => {
                    if self.output_shapes()[0][1] == -1 {
                        None
                    } else {
                        // cxywhclss
                        Some(self.output_shapes()[0][1] as u32 - 4)
                    }
                }
                YOLOTask::Pose => {
                    match self.num_keypoint() {
                        None => None,
                        Some(nk) => {
                            if self.output_shapes()[0][1] == -1 {
                                None
                            } else {
                                // cxywhclss3*kpt
                                Some(self.output_shapes()[0][1] as u32 - 4 - 3 * nk)
                            }
                        }
                    }
                }
                YOLOTask::Segment => {
                    if self.output_shapes()[0][1] == -1 {
                        None
                    } else {
                        // cxywhclssnm
                        Some((self.output_shapes()[0][1] - self.output_shapes()[1][1]) as u32 - 4)
                    }
                }
            },
        }
    }
    pub fn num_keypoint(&self) -> Option<u32> {
        // num_keypoints, metadata parsing: String `nk` in onnx model: `[17, 3]`
        match self.fetch_from_metadata("kpt_shape") {
            None => None,
            Some(kpt_string) => {
                let re = Regex::new(r"([0-9]+), ([0-9]+)").unwrap();
                let caps = re.captures(&kpt_string).unwrap();
                Some(caps.get(1).unwrap().as_str().parse::<u32>().unwrap())
            }
        }
    }

    pub fn num_masks(&self) -> Option<u32> {
        // num_masks
        match self.task {
            YOLOTask::Segment => Some(self.output_shapes()[1][1] as u32),
            _ => None,
        }
    }

    pub fn num_anchors(&self) -> Option<u32> {
        // num_anchors
        match self.task {
            YOLOTask::Segment | YOLOTask::Detect | YOLOTask::Pose => {
                if self.output_shapes()[0][2] == -1 {
                    None
                } else {
                    Some(self.output_shapes()[0][2] as u32)
                }
            }
            _ => None,
        }
    }

    pub fn height(&self) -> u32 {
        self.inputs.sizes[0][0]
    }

    pub fn width(&self) -> u32 {
        self.inputs.sizes[0][1]
    }

    pub fn task(&self) -> YOLOTask {
        self.task.clone()
    }

    pub fn author(&self) -> Option<String> {
        self.fetch_from_metadata("author")
    }

    pub fn version(&self) -> Option<String> {
        self.fetch_from_metadata("version")
    }
}


pub fn ep(config: &OrtConfig) -> Result<ExecutionProviderDispatch> {
    let ep = match config.ep {
        OrtEP::Cpu => {
            ExecutionProviderDispatch::CPU(CPUExecutionProvider::default())
        }
        OrtEP::Cuda(device_id) => {
            let ep = ExecutionProviderDispatch::CUDA(
                CUDAExecutionProvider::default()
                    .with_device_id(device_id)
            );
            if ep.is_available()? {
                ep
            } else {
                info!("CUDA Execution Provider is not available, Use Cpu Provider Instead");
                ExecutionProviderDispatch::CPU(CPUExecutionProvider::default())
            }
        }
        OrtEP::Trt(device_id) => {
            let ep = ExecutionProviderDispatch::TensorRT(
                TensorRTExecutionProvider::default()
                    .with_device_id(device_id)
            );

            if ep.is_available()? {
                ep
            } else {
                info!("TensorRT Execution Provider is not available, Use Cpu Provider Instead");
                ExecutionProviderDispatch::CPU(CPUExecutionProvider::default())
            }
        }
        OrtEP::OpenVino => {
            let ep = ExecutionProviderDispatch::OpenVINO(
                OpenVINOExecutionProvider::default()
                    .with_opencl_throttling()
            );
            if ep.is_available()? {
                ep
            } else {
                info!("OpenVino Execution Provider is not available, Use Cpu Provider Instead");
                ExecutionProviderDispatch::CPU(CPUExecutionProvider::default())
            }
        }
        OrtEP::CoreML => {
            let ep = ExecutionProviderDispatch::CoreML(
                CoreMLExecutionProvider::default().
                    with_ane_only().
                    with_subgraphs()
            );
            if ep.is_available()? {
                ep
            } else {
                info!("CoreML Execution Provider is not available, Use Cpu Provider Instead");
                ExecutionProviderDispatch::CPU(CPUExecutionProvider::default())
            }
        }
    };

    Ok(ep)
}

pub mod tests {
    use super::*;
    use opencv::core::{MatTraitConst};
    use opencv::imgcodecs::imread;
    use std::path::Path;
    use crate::utils::logger::init;

    #[test]
    fn test_ort_backend() -> Result<()> {
        let _guard = init("./logs".to_string());

        let config = OrtConfig {
            model_file: "./yolov8n.onnx".to_string(),
            task: Some(YOLOTask::Detect),
            ep: OrtEP::CoreML,
            image_size: (Some(640), Some(640)),
        };


        let backend = OrtBackend::build(config).unwrap();

        let mut img = imread("./test/2.jpg", 1).unwrap();

        let (img_width, img_height) = (img.cols(), img.rows());

        let t_pre = std::time::Instant::now();
        let model_input = backend.pre_process(&img)?;

        println!("[Image Preprocess]: {:?}", t_pre.elapsed());


        let t_infer = std::time::Instant::now();

        let model_input = Tensor::from_array(model_input)?;
        let output = backend.run(model_input).unwrap();

        println!("[Inference]: {:?}", t_infer.elapsed());

        let t_post = std::time::Instant::now();
        let results = backend.postprocess(output, img_width as f32, img_height as f32, 640., 0.5);
        println!("[Post Process]: {:?}", t_post.elapsed());

        let t_plot = std::time::Instant::now();
        // Draw bounding boxes
        let img = backend.plot(results, &mut img)?;
        println!("[Image plot Preprocess]: {:?}", t_plot.elapsed());

        // 将图片保存到 runs 文件夹
        let output_path = Path::new("./runs").join("output.jpg");
        opencv::imgcodecs::imwrite(output_path.to_str().unwrap(), &img, &opencv::core::Vector::new())?;
        Ok(())
    }
}
