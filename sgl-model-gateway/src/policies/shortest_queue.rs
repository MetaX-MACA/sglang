use core::panic;
use std::{
    collections::HashMap,
    sync::{Arc, LazyLock, Mutex},
    time::Instant,
};

use async_trait::async_trait;

use super::{LoadBalancingPolicy, SelectWorkerInfo};
use crate::core::Worker;

fn get_env_bool(name: &str, default: bool) -> bool {
    let value = std::env::var(name);
    if value.is_err() {
        return default;
    }
    match value.unwrap().to_ascii_lowercase().as_str() {
        "0" | "n" | "no" | "f" | "false" => false,
        "1" | "y" | "yes" | "t" | "true" => true,
        _ => panic!("Invalid value for environment variable {name}"),
    }
}

// Read once on first selection; set this before starting either Rust or Python.
static DEBUG_SHORTEST_QUEUE: LazyLock<bool> =
    LazyLock::new(|| get_env_bool("SGLANG_DEBUG_SHORTEST_QUEUE", false));

#[derive(Debug, Default)]
pub struct ShortestQueuePolicy {
    last_times: Mutex<HashMap<String, Instant>>,
    debug: bool,
}

#[async_trait]
impl LoadBalancingPolicy for ShortestQueuePolicy {
    fn name(&self) -> &'static str {
        "shortest_queue"
    }

    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        _info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let mut loads = self.debug.then(|| Vec::with_capacity(workers.len()));
        if workers.is_empty() {
            if let Some(loads) = loads {
                tracing::info!(idx = ?Option::<usize>::None, ?loads, "shortest_queue none");
            }
            return None;
        }
        let mut select_idx = 0usize;
        let mut min_load = usize::MAX;
        let mut min_last_time = Instant::now();
        let mut last_times = self.last_times.lock().unwrap();
        for (i, worker) in workers.iter().enumerate() {
            let load = worker.load();
            if let Some(loads) = &mut loads {
                loads.push(load);
            }
            if load > min_load {
                continue;
            }
            // 同一个服务的不同 dp 的 url 不同
            let worker_url = worker.url();
            let last_time = if let Some(last_time) = last_times.get_mut(worker_url) {
                *last_time
            } else {
                let last_time = Instant::now();
                last_times.insert(worker_url.to_string(), last_time);
                last_time
            };
            if load < min_load {
                select_idx = i;
                min_load = load;
                min_last_time = last_time;
                continue;
            }
            if last_time < min_last_time {
                select_idx = i;
                min_last_time = last_time;
            }
        }
        *last_times.get_mut(workers[select_idx].url()).unwrap() = Instant::now();
        drop(last_times);
        if let Some(loads) = loads {
            tracing::info!(
                idx = select_idx,
                ?loads,
                "shortest_queue {}",
                workers[select_idx].worker_type().as_metric_label()
            );
        }
        Some(select_idx)
    }

    fn reset(&self) {
        self.last_times.lock().unwrap().clear();
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl ShortestQueuePolicy {
    pub fn new() -> Self {
        Self {
            debug: *DEBUG_SHORTEST_QUEUE,
            ..Default::default()
        }
    }
}
