// 1. 提供注册事件的方法 subscribe
// 2. 提供发送事件的方法  publish
// 3. 提供移除事件的方法 unsubscribe
// 4. 提供清空事件的方法 clear
// 事件的定义如下:
// Event 使用 string 作为事件的名称
// Listener 使用 Box<dyn Fn()> 作为事件的监听器
// 事件的名称和事件的监听器是一对多的关系
// 事件支持传输数据，数据可以是任何类型


use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::any::Any;
use lazy_static::lazy_static;

lazy_static!{
    pub static ref BUS: EventBus = EventBus::new();
}

/// Example:
/// fn main() {
///
///     subscribe!(bus, "event1".to_string(), |data: Arc<dyn Any + Send + Sync>| {
///         if let Some(data) = data.downcast_ref::<String>() {
///            println!("Received event1 with data: {}", data);
///        }
///     });
///
///     publish!(bus, "event1", "Hello, world!".to_string());
/// }
pub struct EventBus {
    listeners: Arc<Mutex<HashMap<String, Vec<Box<dyn Fn(Arc<dyn Any + Send + Sync>) + Send + Sync>>>>>,
}

impl EventBus {
    pub fn new() -> Self {
        Self {
            listeners: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn subscribe<F>(&self, event: String, listener: F)
    where
        F: Fn(Arc<dyn Any + Send + Sync>) + Send + Sync + 'static,
    {
        let mut listeners = self.listeners.lock().unwrap();
        listeners.entry(event).or_insert_with(Vec::new).push(Box::new(listener));
    }

    pub fn publish(&self, event: &str, data: Arc<dyn Any + Send + Sync>) {
        let listeners = self.listeners.lock().unwrap();
        if let Some(listeners) = listeners.get(event) {
            for listener in listeners {
                listener(data.clone());
            }
        }
    }

    pub fn unsubscribe(&self, event: &str) {
        let mut listeners = self.listeners.lock().unwrap();
        listeners.remove(event);
    }

    pub fn clear(&self) {
        let mut listeners = self.listeners.lock().unwrap();
        listeners.clear();
    }
}

#[macro_export]
macro_rules! subscribe {
    ($event:expr, $listener:expr) => {
        BUS.subscribe($event, $listener);
    };
}

#[macro_export]
macro_rules! publish {
    ($event:expr, $data:expr) => {
        BUS.publish($event, Arc::new($data));
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::any::Any;

    #[test]
    fn test_event_bus() {
        subscribe!("event1".to_string(), |data: Arc<dyn Any + Send + Sync>| {
            if let Some(data) = data.downcast_ref::<String>() {
                assert_eq!(data, "Hello, world!");
            } else {
                panic!("Unexpected data type");
            }
        });

        publish!("event1", "Hello, world!".to_string());
    }
}