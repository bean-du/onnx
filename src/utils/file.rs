use std::path::PathBuf;
use std::sync::Arc;
use opencv::core::Mat;
use tokio::sync::Mutex;
use crate::get_base_file_dir;


pub async fn save_opencv_img(img: Arc<Mutex<Mat>>, img_name: String) -> anyhow::Result<String> {
    let img = img.lock().await;
    let img = img.clone();

    let mut name = img_name.clone();
    let name = if name.contains("/") {
        name.replace("/", "")
    } else if name.contains("jpg") {
        name
    } else {
        name.push_str(".jpg");
        name
    };
    let mut path = PathBuf::from(&get_base_file_dir().await?);
    path.push("images");
    path.push(name);

    opencv::imgcodecs::imwrite(&path.to_string_lossy(), &img, &opencv::core::Vector::new())?;
    Ok(path.to_string_lossy().to_string())
}
