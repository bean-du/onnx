#![allow(clippy::type_complexity)]

use std::ops::Deref;
use std::sync::{Arc, Mutex};
use anyhow::Error;
use ndarray::{Array2, Array4, ArrayBase, ArrayD, Ix, Ix4, OwnedRepr};
use opencv::core::{Mat, MatTrait, MatTraitConst, MatTraitConstManual, split, Vec3b, self, NORM_MINMAX, CV_32FC1, CV_32FC3};
use opencv::imgproc;
use opencv::imgproc::COLOR_BGR2RGB;
use ort::{ort, Value};
use ort::tensor::TensorData;
use rusttype::Font;
use tracing::{debug, info};
use {
    anyhow::Result,
    std::path::PathBuf,
    ndarray::{s, Array, Axis, IxDyn},
    rand::{thread_rng, Rng},
    image::{DynamicImage, GenericImageView, ImageBuffer},
    crate::yolo::{
        check_font, gen_time_string, non_max_suppression,
        Args, Bbox, Embedding, OrtBackend,
        OrtConfig, OrtEP, Point2, YOLOResult, YOLOTask, SKELETON,
    },
};
use rayon::prelude::*;
use ort_2;
use ort_2::ValueType::Tensor;

#[derive(Debug)]
pub struct YOLOv8 {
    // YOLOv8 model for all yolo-tasks
    engine: OrtBackend,
    nc: u32,
    nk: u32,
    nm: u32,
    height: u32,
    width: u32,
    task: YOLOTask,
    conf: f32,
    kconf: f32,
    iou: f32,
    names: Vec<String>,
    color_palette: Vec<(u8, u8, u8)>,
    profile: bool,
    plot: bool,
}

impl YOLOv8 {
    pub fn new(config: Args) -> Result<Self> {
        // execution provider
        let ep = if config.trt {
            OrtEP::Trt(config.device_id as i32)
        } else if config.cuda {
            OrtEP::Cuda(config.device_id as i32)
        } else if config.open_vino {
            OrtEP::OpenVino
        } else if config.core_ml {
            OrtEP::CoreML
        } else {
            OrtEP::Cpu
        };


        // build ort engine
        let ort_args = OrtConfig {
            ep,
            model_file: config.model,
            task: config.task,
            image_size: (config.height, config.width),
        };
        let engine = OrtBackend::build(ort_args)?;

        //  get batch, height, width, tasks, nc, nk, nm
        let (height, width, task) = (
            engine.height(),
            engine.width(),
            engine.task(),
        );
        let nc = engine.num_class().or(config.nc).unwrap_or_else(|| {
            panic!("Failed to get num_classes, make it explicit with `--nc`");
        });
        let (nk, nm) = match task {
            YOLOTask::Pose => {
                let nk = engine.num_keypoint().or(config.nk).unwrap_or_else(|| {
                    panic!("Failed to get num_keypoints, make it explicit with `--nk`");
                });
                (nk, 0)
            }
            YOLOTask::Segment => {
                let nm = engine.num_masks().or(config.nm).unwrap_or_else(|| {
                    panic!("Failed to get num_masks, make it explicit with `--nm`");
                });
                (0, nm)
            }
            _ => (0, 0),
        };

        // class names
        let names = engine.names().unwrap_or(vec!["Unknown".to_string()]);

        // color palette
        let mut rng = thread_rng();
        let color_palette: Vec<_> = names
            .iter()
            .map(|_| {
                (
                    rng.gen_range(0..=255),
                    rng.gen_range(0..=255),
                    rng.gen_range(0..=255),
                )
            })
            .collect();

        Ok(Self {
            engine,
            names,
            conf: config.conf,
            kconf: config.kconf,
            iou: config.iou,
            color_palette,
            profile: config.profile,
            plot: config.plot,
            nc,
            nk,
            nm,
            height,
            width,
            task,
        })
    }

    pub fn scale_wh(&self, w0: f32, h0: f32, w1: f32, h1: f32) -> (f32, f32, f32) {
        let r = (w1 / w0).min(h1 / h0);
        (r, (w0 * r).round(), (h0 * r).round())
    }


    pub fn pre_process(&self, img: &Mat) -> Result<Array<f32, IxDyn>> {
        let interpolation = match self.task() {
            YOLOTask::Classify => imgproc::INTER_LINEAR,
            YOLOTask::Segment | YOLOTask::Pose => imgproc::INTER_CUBIC,
            _ => imgproc::INTER_LINEAR,
        };
        let mut resized_frame = Mat::default();
        // resize img to 640 * 640
        imgproc::resize(
            &img,
            &mut resized_frame,
            opencv::core::Size::new(640 as i32, 640 as i32),
            0.0,
            0.0,
            interpolation,
        )?;

        let mut dst = Mat::default();
        imgproc::cvt_color(&img, &mut dst, COLOR_BGR2RGB, 0)?;

        let mut converted_mat = Mat::default();
        dst.convert_to(&mut converted_mat, CV_32FC1, 1.0, 0.0)?;

        let mut channels: opencv::core::Vector<Mat> = opencv::core::Vector::new();
        split(&converted_mat, &mut channels)?;


        let b = channels.get(0)?.data_typed::<f32>()?.to_vec();
        let g = channels.get(1)?.data_typed::<f32>()?.to_vec();
        let r = channels.get(2)?.data_typed::<f32>()?.to_vec();


        let w = self.width() as Ix;
        let h = self.height() as Ix;
        let mut model_input = ArrayD::<f32>::zeros(IxDyn(&[1, 3, 640, 640]));
        model_input.slice_mut(s![0, 0, .., ..]).assign(&Array2::from_shape_vec((640, 640), b)?);
        model_input.slice_mut(s![0, 1, .., ..]).assign(&Array2::from_shape_vec((640, 640), g)?);
        model_input.slice_mut(s![0, 2, .., ..]).assign(&Array2::from_shape_vec((640, 640), r)?);

        model_input /= 255.0;

        Ok(model_input)
    }

    pub fn run_video_frame(&mut self, img: Arc<Mat>) -> Result<Mat> {
        let mut mat = img.deref().clone();

        let w = mat.cols();
        let h = mat.rows();

        let t_pre = std::time::Instant::now();
        let data = self.pre_process(&mat)?;
        let model_input = ort_2::Tensor::from_array(data)?;
        if self.profile {
            println!("[Frame Preprocess]: {:?}", t_pre.elapsed());
        }

        // 执行推理
        // 调用 YOLOv8::run 方法，传入 ndarray::Array 对象，返回一个 Result 类型，包含一个 YOLOResult 类型的向量
        let ys = match self.engine.run(model_input) {
            Ok(ys) => ys,
            Err(e) => {
                info!("Execute predict Error: {:?}", e);
                return Err(e);
            }
        };


        // 后处理检测结果
        let t_post = std::time::Instant::now();
        let results = self.engine.postprocess(ys, w as f32, h as f32, 640., 0.5);
        if self.profile {
            println!("[Inference Postprocess]: {:?}", t_post.elapsed());
        }

        let t_plot = std::time::Instant::now();
        let img = self.engine.plot(results, &mut mat)?;
        println!("[Image plot Preprocess]: {:?}", t_plot.elapsed());

        // println!("[Per image at shape: (1, 3, {:?}, {})]", self.height(), self.width());
        // info!(" <[Result]> {:?}", ys);
        Ok(img)
    }


    pub fn save(&self, img: DynamicImage, path: String) -> Result<()> {
        // 如果不存在runs目录，则创建runs目录
        let mut runs = PathBuf::from(path);
        if !runs.exists() {
            std::fs::create_dir_all(&runs).unwrap();
        }
        // 生成保存的文件名
        runs.push(gen_time_string("-"));
        let saveout = format!("{}.jpg", runs.to_str().unwrap());

        // 保存图像
        match img.save(saveout) {
            Ok(_) => Ok(()),
            Err(e) => Err(anyhow::anyhow!("Error: {:?}", e)),
        }
    }


    pub fn postprocess(
        &self,
        xs: &Array<f32, IxDyn>,
        xs0: Arc<Mat>,
    ) -> Result<Vec<YOLOResult>> {
        if let YOLOTask::Classify = self.task() {
            let mut ys = Vec::new();
            let preds = xs;
            for batch in preds.axis_iter(Axis(0)) {
                ys.push(YOLOResult::new(
                    Some(Embedding::new(batch.into_owned())),
                    None,
                    None,
                    None,
                ));
            }
            Ok(ys)
        } else {
            const CXYWH_OFFSET: usize = 4; // cxcywh
            const KPT_STEP: usize = 3; // xyconf
            let preds = xs;
            let protos: Option<&Array<f32, IxDyn>> = None;
            let mut ys = Vec::new();
            for (idx, anchor) in preds.axis_iter(Axis(0)).enumerate() {
                // [bs, 4 + nc + nm, anchors]
                // input image
                let width_original = xs0.cols() as f32;
                let height_original = xs0.rows() as f32;
                let ratio = (self.width() as f32 / width_original)
                    .min(self.height() as f32 / height_original);

                // save each result
                let mut data: Vec<(Bbox, Option<Vec<Point2>>, Option<Vec<f32>>)> = Vec::new();
                for pred in anchor.axis_iter(Axis(1)) {
                    // split preds for different tasks
                    let bbox = pred.slice(s![0..CXYWH_OFFSET]);
                    let clss = pred.slice(s![CXYWH_OFFSET..CXYWH_OFFSET + self.nc() as usize]);
                    let kpts = {
                        if let YOLOTask::Pose = self.task() {
                            Some(pred.slice(s![pred.len() - KPT_STEP * self.nk() as usize..]))
                        } else {
                            None
                        }
                    };
                    let coefs = {
                        if let YOLOTask::Segment = self.task() {
                            Some(pred.slice(s![pred.len() - self.nm() as usize..]).to_vec())
                        } else {
                            None
                        }
                    };

                    // confidence and id
                    let (id, &confidence) = clss
                        .into_iter()
                        .enumerate()
                        .reduce(|max, x| if x.1 > max.1 { x } else { max })
                        .unwrap(); // definitely will not panic!

                    // confidence filter
                    if confidence < self.conf {
                        continue;
                    }

                    // debug!("Bbox info: cx: {}, cy: {}, w: {}, h: {}", bbox[0], bbox[1], bbox[2], bbox[3]);
                    // bbox re-scale
                    let cx = bbox[0] / ratio;
                    let cy = bbox[1] / ratio;
                    let w = bbox[2] / ratio;
                    let h = bbox[3] / ratio;
                    let x = cx - w / 2.;
                    let y = cy - h / 2.;

                    let y_bbox = Bbox::new(
                        x.max(0.0f32).min(width_original),
                        y.max(0.0f32).min(height_original),
                        w,
                        h,
                        id,
                        confidence,
                    );

                    // kpts
                    let y_kpts = {
                        if let Some(kpts) = kpts {
                            let mut kpts_ = Vec::new();
                            // rescale
                            for i in 0..self.nk() as usize {
                                let kx = kpts[KPT_STEP * i] / ratio;
                                let ky = kpts[KPT_STEP * i + 1] / ratio;
                                let kconf = kpts[KPT_STEP * i + 2];
                                if kconf < self.kconf {
                                    kpts_.push(Point2::default());
                                } else {
                                    kpts_.push(Point2::new_with_conf(
                                        kx.max(0.0f32).min(width_original),
                                        ky.max(0.0f32).min(height_original),
                                        kconf,
                                    ));
                                }
                            }
                            Some(kpts_)
                        } else {
                            None
                        }
                    };

                    // data merged
                    data.push((y_bbox, y_kpts, coefs));
                }

                // nms
                non_max_suppression(&mut data, self.iou);

                // decode
                let mut y_bboxes: Vec<Bbox> = Vec::new();
                let mut y_kpts: Vec<Vec<Point2>> = Vec::new();
                let mut y_masks: Vec<Vec<u8>> = Vec::new();
                for elem in data.into_iter() {
                    if let Some(kpts) = elem.1 {
                        y_kpts.push(kpts)
                    }

                    // decode masks
                    if let Some(coefs) = elem.2 {
                        let proto = protos.unwrap().slice(s![idx, .., .., ..]);
                        let (nm, nh, nw) = proto.dim();

                        // coefs * proto -> mask
                        let coefs = Array::from_shape_vec((1, nm), coefs)?; // (n, nm)
                        let proto = proto.to_owned().into_shape((nm, nh * nw))?; // (nm, nh*nw)
                        let mask = coefs.dot(&proto).into_shape((nh, nw, 1))?; // (nh, nw, n)

                        // build image from ndarray
                        let mask_im: ImageBuffer<image::Luma<_>, Vec<f32>> =
                            match ImageBuffer::from_raw(nw as u32, nh as u32, mask.into_raw_vec()) {
                                Some(image) => image,
                                None => panic!("can not create image from ndarray"),
                            };
                        let mut mask_im = image::DynamicImage::from(mask_im); // -> dyn

                        // rescale masks
                        let (_, w_mask, h_mask) =
                            self.scale_wh(width_original, height_original, nw as f32, nh as f32);
                        let mask_cropped = mask_im.crop(0, 0, w_mask as u32, h_mask as u32);
                        let mask_original = mask_cropped.resize_exact(
                            // resize_to_fill
                            width_original as u32,
                            height_original as u32,
                            match self.task() {
                                YOLOTask::Segment => image::imageops::FilterType::CatmullRom,
                                _ => image::imageops::FilterType::Triangle,
                            },
                        );

                        // crop-mask with bbox
                        let mut mask_original_cropped = mask_original.into_luma8();
                        for y in 0..height_original as usize {
                            for x in 0..width_original as usize {
                                if x < elem.0.xmin() as usize
                                    || x > elem.0.xmax() as usize
                                    || y < elem.0.ymin() as usize
                                    || y > elem.0.ymax() as usize
                                {
                                    mask_original_cropped.put_pixel(
                                        x as u32,
                                        y as u32,
                                        image::Luma([0u8]),
                                    );
                                }
                            }
                        }
                        y_masks.push(mask_original_cropped.into_raw());
                    }
                    y_bboxes.push(elem.0);
                }

                // save each result
                let y = YOLOResult {
                    probs: None,
                    bboxes: if !y_bboxes.is_empty() {
                        Some(y_bboxes)
                    } else {
                        None
                    },
                    keypoints: if !y_kpts.is_empty() {
                        Some(y_kpts)
                    } else {
                        None
                    },
                    masks: if !y_masks.is_empty() {
                        Some(y_masks)
                    } else {
                        None
                    },
                };
                ys.push(y);
            }

            Ok(ys)
        }
    }


    // pub fn summary(&self) {
    //     println!(
    //         "\nSummary:\n\
    //         > Task: {:?}{}\n\
    //         > Batch: {} ({}), Height: {} ({}), Width: {} ({})\n\
    //         > nc: {} nk: {}, nm: {}, conf: {}, kconf: {}, iou: {}\n\
    //         ",
    //         self.task(),
    //         match self.engine.author().zip(self.engine.version()) {
    //             Some((author, ver)) => format!(" ({} {})", author, ver),
    //             None => String::from(""),
    //         },
    //         // self.engine.ep(),
    //         // if let OrtEP::Cpu = self.engine.ep() {
    //         //     ""
    //         // } else {
    //         //     "(May still fall back to CPU)"
    //         // },
    //         // self.engine.dtype(),
    //
    //         self.height(),
    //         if self.engine.is_height_dynamic() {
    //             "Dynamic"
    //         } else {
    //             "Const"
    //         },
    //         self.width(),
    //         if self.engine.is_width_dynamic() {
    //             "Dynamic"
    //         } else {
    //             "Const"
    //         },
    //         self.nc(),
    //         self.nk(),
    //         self.nm(),
    //         self.conf,
    //         self.kconf,
    //         self.iou,
    //     );
    // }

    pub fn engine(&self) -> &OrtBackend {
        &self.engine
    }

    pub fn conf(&self) -> f32 {
        self.conf
    }

    pub fn set_conf(&mut self, val: f32) {
        self.conf = val;
    }

    pub fn conf_mut(&mut self) -> &mut f32 {
        &mut self.conf
    }

    pub fn kconf(&self) -> f32 {
        self.kconf
    }

    pub fn iou(&self) -> f32 {
        self.iou
    }

    pub fn task(&self) -> &YOLOTask {
        &self.task
    }


    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn nc(&self) -> u32 {
        self.nc
    }

    pub fn nk(&self) -> u32 {
        self.nk
    }

    pub fn nm(&self) -> u32 {
        self.nm
    }

    pub fn names(&self) -> &Vec<String> {
        &self.names
    }
}
