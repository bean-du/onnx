#![allow(clippy::type_complexity)]

use std::ops::Deref;
use std::sync::{Arc, Mutex};
use anyhow::Error;
use ndarray::Ix;
use opencv::core::{Mat, MatTrait, MatTraitConst, MatTraitConstManual, split, Vec3b};
use opencv::imgproc;
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
        Args, Batch, Bbox, Embedding, OrtBackend,
        OrtConfig, OrtEP, Point2, YOLOResult, YOLOTask, SKELETON,
    },
};
use rayon::prelude::*;

#[derive(Debug)]
pub struct YOLOv8 {
    // YOLOv8 model for all yolo-tasks
    engine: OrtBackend,
    nc: u32,
    nk: u32,
    nm: u32,
    height: u32,
    width: u32,
    batch: u32,
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
            OrtEP::Trt(config.device_id)
        } else if config.cuda {
            OrtEP::Cuda(config.device_id)
        } else if config.open_vino {
            OrtEP::OpenVino
        } else if config.core_ml {
            OrtEP::CoreML
        } else {
            OrtEP::Cpu
        };

        // batch
        let batch = Batch {
            opt: config.batch,
            min: config.batch_min,
            max: config.batch_max,
        };

        // build ort engine
        let ort_args = OrtConfig {
            ep,
            batch,
            f: config.model,
            task: config.task,
            trt_fp16: config.fp16,
            image_size: (config.height, config.width),
        };
        let engine = OrtBackend::build(ort_args)?;

        //  get batch, height, width, tasks, nc, nk, nm
        let (batch, height, width, task) = (
            engine.batch(),
            engine.height(),
            engine.width(),
            engine.task(),
        );
        let nc = engine.nc().or(config.nc).unwrap_or_else(|| {
            panic!("Failed to get num_classes, make it explicit with `--nc`");
        });
        let (nk, nm) = match task {
            YOLOTask::Pose => {
                let nk = engine.nk().or(config.nk).unwrap_or_else(|| {
                    panic!("Failed to get num_keypoints, make it explicit with `--nk`");
                });
                (nk, 0)
            }
            YOLOTask::Segment => {
                let nm = engine.nm().or(config.nm).unwrap_or_else(|| {
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
            batch,
            task,
        })
    }

    pub fn scale_wh(&self, w0: f32, h0: f32, w1: f32, h1: f32) -> (f32, f32, f32) {
        let r = (w1 / w0).min(h1 / h0);
        (r, (w0 * r).round(), (h0 * r).round())
    }

    // 定义预处理函数，输入是一个包含 DynamicImage 的向量，输出是一个 Result 类型，包含一个 f32 类型的 ndarray::Array
    pub fn preprocess(&mut self, xs: &Vec<DynamicImage>) -> Result<Array<f32, IxDyn>> {
        // 创建一个全为 1 的 ndarray::Array，尺寸为 (图像数量, 3, 高度, 宽度)
        let mut ys = Array::ones((xs.len(), 3, self.height() as usize, self.width() as usize)).into_dyn();
        // 将 ys 中的所有元素填充为 144.0 / 255.0
        ys.fill(144.0 / 255.0);

        // 遍历输入的图像
        for (idx, x) in xs.iter().enumerate() {
            // 根据任务类型调整图像尺寸
            let img = match self.task() {
                // 如果任务类型是 Classify，则将图像尺寸调整为模型的输入尺寸
                YOLOTask::Classify => x.resize_exact(
                    self.width(),
                    self.height(),
                    image::imageops::FilterType::Triangle,
                ),
                // 如果任务类型是其他类型，则根据原始图像和模型输入尺寸的比例调整图像尺寸
                _ => {
                    let (w0, h0) = x.dimensions();
                    let w0 = w0 as f32;
                    let h0 = h0 as f32;
                    let (_, w_new, h_new) =
                        self.scale_wh(w0, h0, self.width() as f32, self.height() as f32); // f32 round
                    x.resize_exact(
                        w_new as u32,
                        h_new as u32,
                        if let YOLOTask::Segment = self.task() {
                            image::imageops::FilterType::CatmullRom
                        } else {
                            image::imageops::FilterType::Triangle
                        },
                    )
                }
            };

            // 遍历调整尺寸后的图像的每个像素
            for (x, y, rgb) in img.pixels() {
                let x = x as usize;
                let y = y as usize;
                let [r, g, b, _] = rgb.0;
                // 将像素的 RGB 值归一化后存入 ys
                ys[[idx, 0, y, x]] = (r as f32) / 255.0;
                ys[[idx, 1, y, x]] = (g as f32) / 255.0;
                ys[[idx, 2, y, x]] = (b as f32) / 255.0;
            }
        }
        // 返回 ys
        Ok(ys)
    }


    pub fn run_video_frame(&mut self, img: Arc<Mat>) -> Result<Vec<YOLOResult>> {
        let mut frame = Mat::default();
        imgproc::cvt_color(img.deref(), &mut frame, imgproc::COLOR_BGR2RGB, 0)?;

        let t_pre = std::time::Instant::now();
        // 调整图像尺寸以匹配 YOLO 模型的输入尺寸
        let mut resized_frame = Mat::default();
        let interpolation = match self.task() {
            // 如果任务类型是 Classify，则使用线性插值
            YOLOTask::Classify => imgproc::INTER_LINEAR,
            YOLOTask::Segment | YOLOTask::Pose => imgproc::INTER_CUBIC,
            // 如果任务类型是其他类型，则使用立方插值
            _ => imgproc::INTER_LINEAR,
        };

        imgproc::resize(
            &frame,
            &mut resized_frame,
            opencv::core::Size::new(self.width() as i32, self.height() as i32),
            0.0,
            0.0,
            interpolation,
        )?;

        debug!("resized_image size: {:?}, {:?}", resized_frame.rows(), resized_frame.cols());
        // 归一化处理
        let mut ys = Array::ones((1, 3, self.height() as usize, self.width() as usize)).into_dyn();
        // 将 ys 中的所有元素填充为 144.0 / 255.0
        ys.fill(144.0 / 255.0);


        let rows = resized_frame.rows() as usize;
        let cols = resized_frame.cols() as usize;

        let mut channel: opencv::core::Vector<opencv::core::Mat> = opencv::core::Vector::new();
        split(&resized_frame, &mut channel)?;
        let mut ys_b = Array::from_shape_vec(
            (rows, cols),
            channel.get(0)?.data_typed::<u8>()?.to_vec(),
        )?;

        let mut ys_g = Array::from_shape_vec(
            (rows, cols),
            channel.get(1)?.data_typed::<u8>()?.to_vec(),
        )?;

        let mut ys_r = Array::from_shape_vec(
            (rows, cols),
            channel.get(2)?.data_typed::<u8>()?.to_vec(),
        )?;

        let mut ys_vec = vec![ys_r.view_mut(), ys_g.view_mut(), ys_b.view_mut()];
        // 遍历调整尺寸后的图像的每个像素,归一化处理
        ys_vec.par_iter_mut().for_each(|ys| {
            ys.mapv(|x| x as f32 / 255.0);
        });

        if self.profile {
            println!("[Model Preprocess]: {:?}", t_pre.elapsed());
        }

        // 执行推理
        // 调用 YOLOv8::run 方法，传入 ndarray::Array 对象，返回一个 Result 类型，包含一个 YOLOResult 类型的向量
        let ys = match self.engine.run(ys, self.profile) {
            Ok(ys) => ys,
            Err(e) => {
                info!("Execute predict Error: {:?}", e);
                return Err(e);
            }
        };


        // 后处理检测结果
        let t_post = std::time::Instant::now();
        let ys = match self.postprocess(&ys[0], img) {
            Ok(ys) => ys,
            Err(e) => {
                info!("Postprocess Error: {:?}", e);
                return Err(e);
            }
        };
        if self.profile {
            println!("[Model Postprocess]: {:?}", t_post.elapsed());
        }
        println!("[Per image at shape: (1, 3, {:?}, {})]", self.height(), self.width());
        // info!(" <[Result]> {:?}", ys);
        Ok(ys)
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
            let xs0 = xs0.deref();
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

    pub fn plot<'a>(&'a self, font: &Font, res: &YOLOResult, img: &'a mut Mat, skeletons: Option<&[(usize, usize)]>) -> Result<&Mat> {
        // 如果YOLO结果中存在概率，则绘制分类结果
        if let Some(probs) = res.probs() {
            // 遍历概率的前5个元素
            for (i, k) in probs.topk(5).iter().enumerate() {
                // 格式化类别名称和概率
                let legend = format!("{} {:.2}%", self.names[k.0], k.1);
                // 设置图例的大小
                let scale = 32;
                let legend_size = img.cols().max(img.cols()) / scale;
                // 设置图例的位置
                let x = img.cols() / 20 + 10; // add offset
                let y = img.rows() as u32 / 20 + i as u32 * legend_size as u32 + 10; // add offset
                // 在图像上绘制图例
                imgproc::put_text(
                    img,
                    &legend,
                    opencv::core::Point::new(x, y as i32),
                    imgproc::INTER_LINEAR,
                    1.0,
                    opencv::core::Scalar::new(0., 255., 0., 0.),
                    2,
                    imgproc::LINE_8,
                    false,
                )?;
            }
        }

        // 如果YOLO结果中存在边界框，则绘制边界框和关键点
        if let Some(bboxes) = res.bboxes() {
            // 遍历所有的边界框
            for (_idx, bbox) in bboxes.iter().enumerate() {
                // 在图像上绘制边界框
                imgproc::rectangle(
                    img,
                    opencv::core::Rect::new(bbox.xmin() as i32, bbox.ymin() as i32, bbox.width() as i32, bbox.height() as i32),
                    opencv::core::Scalar::new(self.color_palette[bbox.id()].0 as f64, self.color_palette[bbox.id()].1 as f64, self.color_palette[bbox.id()].2 as f64, 0.),
                    2,
                    opencv::imgproc::LINE_8,
                    0,
                )?;
                // 格式化类别名称和置信度
                let legend = format!("{} {:.2}%", self.names[bbox.id()], bbox.confidence());
                // 设置图例的大小
                let scale = 40;
                let legend_size = img.rows().max(img.rows()) / scale;
                // 在图像上绘制图例
                let y = (bbox.ymin() - legend_size as f32) as i32;
                let y = if y < 0 { 0 } else { y }; // ensure y is within image
                imgproc::put_text(
                    img,
                    &legend,
                    opencv::core::Point::new(bbox.xmin() as i32, y),
                    opencv::imgproc::INTER_LINEAR,
                    1.0,
                    opencv::core::Scalar::new(self.color_palette[bbox.id()].0 as f64, self.color_palette[bbox.id()].1 as f64, self.color_palette[bbox.id()].2 as f64, 0.),
                    2,
                    opencv::imgproc::LINE_8,
                    false,
                )?;
            }
        }

        // 如果YOLO结果中存在关键点，则绘制关键点
        if let Some(keypoints) = res.keypoints() {
            // 遍历所有的关键点
            for kpts in keypoints.iter() {
                for kpt in kpts.iter() {
                    // 如果关键点的置信度小于设定的阈值，则跳过此关键点
                    if kpt.confidence() < self.kconf {
                        continue;
                    }

                    // 在图像上绘制关键点
                    imgproc::circle(
                        img,
                        opencv::core::Point::new(kpt.x() as i32, kpt.y() as i32),
                        2,
                        opencv::core::Scalar::new(0., 255., 0., 0.),
                        2,
                        opencv::imgproc::LINE_8,
                        0,
                    )?;
                }

                // 如果存在骨架连接关系，则绘制骨架
                if let Some(skeletons) = skeletons {
                    for &(idx1, idx2) in skeletons.iter() {
                        let kpt1 = &kpts[idx1];
                        let kpt2 = &kpts[idx2];
                        // 如果关键点的置信度小于设定的阈值，则跳过此关键点
                        if kpt1.confidence() < self.kconf || kpt2.confidence() < self.kconf {
                            continue;
                        }
                        // 在图像上绘制骨架
                        imgproc::line(
                            img,
                            opencv::core::Point::new(kpt1.x() as i32, kpt1.y() as i32),
                            opencv::core::Point::new(kpt2.x() as i32, kpt2.y() as i32),
                            opencv::core::Scalar::new(233., 14., 57., 0.),
                            2,
                            opencv::imgproc::LINE_8,
                            0,
                        )?;
                    }
                }
            }
        }
        Ok(img)
    }


    pub fn summary(&self) {
        println!(
            "\nSummary:\n\
            > Task: {:?}{}\n\
            > EP: {:?} {}\n\
            > Dtype: {:?}\n\
            > Batch: {} ({}), Height: {} ({}), Width: {} ({})\n\
            > nc: {} nk: {}, nm: {}, conf: {}, kconf: {}, iou: {}\n\
            ",
            self.task(),
            match self.engine.author().zip(self.engine.version()) {
                Some((author, ver)) => format!(" ({} {})", author, ver),
                None => String::from(""),
            },
            self.engine.ep(),
            if let OrtEP::Cpu = self.engine.ep() {
                ""
            } else {
                "(May still fall back to CPU)"
            },
            self.engine.dtype(),
            self.batch(),
            if self.engine.is_batch_dynamic() {
                "Dynamic"
            } else {
                "Const"
            },
            self.height(),
            if self.engine.is_height_dynamic() {
                "Dynamic"
            } else {
                "Const"
            },
            self.width(),
            if self.engine.is_width_dynamic() {
                "Dynamic"
            } else {
                "Const"
            },
            self.nc(),
            self.nk(),
            self.nm(),
            self.conf,
            self.kconf,
            self.iou,
        );
    }

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

    pub fn batch(&self) -> u32 {
        self.batch
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
