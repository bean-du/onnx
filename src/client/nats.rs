use async_nats;
use anyhow::{Error, Result};
use tracing::{info, debug};
use async_nats::jetstream;
use futures::{StreamExt, TryStreamExt};
use opencv::core::add;
use tokio::sync::Mutex;
use once_cell::sync::Lazy;

type NatsClient = Lazy<async_nats::Client>;

const DEFAULT_NATS_ADDR: &'static str = "nats://localhost:4222";

static NATS_CLIENT: Lazy<Mutex<Option<async_nats::Client>>> = Lazy::new(|| Mutex::new(None));

pub async fn get_nats_client() -> Result<async_nats::Client, Error> {
    let mut nats_client = NATS_CLIENT.lock().await;
    if nats_client.is_none() {
        let addr = option_env!("NATS_ADDR").unwrap_or(DEFAULT_NATS_ADDR);
        *nats_client = Some(async_nats::connect(addr).await?);
        info!("Connected to Nats server[{:?}] successfully!", addr);
    }
    Ok(nats_client.as_ref().unwrap().clone())
}

#[cfg(test)]
mod test {
    use crate::client::nats::get_nats_client;
    use crate::utils::logger;

    #[tokio::test]
    async fn test_listen_topic() {
        let _guard = logger::init("logs".into()).unwrap();

        // Initialize the NATS client before starting the test
        let c = get_nats_client().await.unwrap();

        c.publish("test", "test".into()).await.unwrap()
    }
}
