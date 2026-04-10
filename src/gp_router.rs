// SPDX-License-Identifier: AGPL-3.0-or-later
use chrono::Timelike;
use gp_routing::features::BACKEND_SLOTS;
use gp_routing::{thompson_sample, ucb_score, FeatureVector, GpRoutingConfig, GpSurrogate, ObservationBuffer};
use parking_lot::Mutex;
use rand::Rng;
use std::cmp::Ordering as CmpOrdering;
use std::collections::HashMap;
use tracing::{debug, info, warn};

use crate::config::{Config, GpAcquisitionStrategy, GpRoutingRuntimeConfig};
use crate::router::AnthropicRequest;

#[derive(Debug)]
pub struct GpRequestRouter {
    runtime_config: GpRoutingRuntimeConfig,
    model_config: GpRoutingConfig,
    surrogate: GpSurrogate,
    buffer: Mutex<ObservationBuffer>,
    backend_indices: HashMap<String, usize>,
}

#[derive(Clone, Debug)]
pub struct GpRoutingPlan {
    pub ordered: Vec<(String, String)>,
    request_features: RequestFeatureContext,
}

#[derive(Clone, Debug)]
struct RequestFeatureContext {
    prompt_chars: usize,
    request_class: u8,
    has_system_prompt: bool,
    message_count: usize,
    tool_count: usize,
    active_streams: usize,
    max_streams: usize,
    hour_of_day: u32,
}

impl RequestFeatureContext {
    fn from_request(request: &AnthropicRequest, active_streams: usize, max_streams: usize) -> Self {
        Self {
            prompt_chars: prompt_chars(request),
            request_class: classify_request(request),
            has_system_prompt: request.system.is_some(),
            message_count: request.messages.len(),
            tool_count: request.tools.as_ref().map_or(0, Vec::len),
            active_streams,
            max_streams,
            hour_of_day: chrono::Local::now().hour(),
        }
    }
}

impl GpRequestRouter {
    pub fn new(runtime_config: GpRoutingRuntimeConfig, canonical_tiers: &[String]) -> Self {
        let backend_indices = canonical_tiers
            .iter()
            .take(BACKEND_SLOTS)
            .enumerate()
            .map(|(idx, tier)| (tier.clone(), idx))
            .collect::<HashMap<_, _>>();

        let model_config = GpRoutingConfig::builder()
            .buffer_capacity(runtime_config.buffer_capacity)
            .min_observations(runtime_config.min_observations)
            .refit_interval(runtime_config.refit_interval)
            .ucb_kappa(runtime_config.ucb_kappa)
            .nugget(runtime_config.nugget)
            .kpls_dim(runtime_config.kpls_dim)
            .prior_mean(runtime_config.prior_mean)
            .prior_variance(runtime_config.prior_variance)
            .n_backends(backend_indices.len().max(1))
            .build();

        if canonical_tiers.len() > BACKEND_SLOTS {
            info!(
                configured_tiers = canonical_tiers.len(),
                gp_slots = BACKEND_SLOTS,
                "gp-routing will learn only the first configured backend slots"
            );
        }

        Self {
            runtime_config,
            surrogate: GpSurrogate::new(model_config.clone()),
            buffer: Mutex::new(ObservationBuffer::new(model_config.buffer_capacity)),
            model_config,
            backend_indices,
        }
    }

    pub fn plan_rerank(
        &self,
        ordered: &[(String, String)],
        request: &AnthropicRequest,
        config: &Config,
        active_streams: usize,
        max_streams: usize,
        pinned_prefix_len: usize,
    ) -> GpRoutingPlan {
        let request_features = RequestFeatureContext::from_request(request, active_streams, max_streams);
        if !self.surrogate.is_fitted() {
            return GpRoutingPlan {
                ordered: ordered.to_vec(),
                request_features,
            };
        }

        let prefix_len = pinned_prefix_len.min(ordered.len());
        let candidate_positions = ordered
            .iter()
            .enumerate()
            .skip(prefix_len)
            .filter(|(_, (tier, _))| self.backend_indices.contains_key(tier))
            .map(|(idx, _)| idx)
            .take(self.runtime_config.max_candidates.min(BACKEND_SLOTS))
            .collect::<Vec<_>>();

        if candidate_positions.len() <= 1 {
            return GpRoutingPlan {
                ordered: ordered.to_vec(),
                request_features,
            };
        }

        let epsilon_pick = match self.runtime_config.acquisition {
            GpAcquisitionStrategy::EpsilonGreedy => {
                let mut rng = rand::thread_rng();
                if rng.gen::<f32>() < self.runtime_config.epsilon.clamp(0.0, 1.0) {
                    Some(rng.gen_range(0..candidate_positions.len()))
                } else {
                    None
                }
            }
            _ => None,
        };

        let mut scored = candidate_positions
            .iter()
            .enumerate()
            .filter_map(|(candidate_idx, pos)| {
                let entry = ordered.get(*pos)?.clone();
                let features = self.build_features(&request_features, &entry.0, 0, config)?;
                let (mean, variance) = self.surrogate.predict(&features);
                let score = self.score_candidate(mean, variance, candidate_idx, epsilon_pick);
                Some((candidate_idx, entry, score, mean, variance))
            })
            .collect::<Vec<_>>();

        scored.sort_by(|left, right| {
            right
                .2
                .partial_cmp(&left.2)
                .unwrap_or(CmpOrdering::Equal)
                .then_with(|| left.0.cmp(&right.0))
        });

        let mut reranked = ordered.to_vec();
        for (pos, (_, entry, _, _, _)) in candidate_positions.iter().zip(scored.iter()) {
            reranked[*pos] = entry.clone();
        }

        if reranked != ordered {
            let scored_log = scored
                .iter()
                .map(|(_, (_, tier_name), score, mean, variance)| {
                    format!(
                        "{}(score={:.3}, mean={:.3}, var={:.3})",
                        tier_name, score, mean, variance
                    )
                })
                .collect::<Vec<_>>();
            debug!(
                prefix_len,
                scored = ?scored_log,
                before = ?ordered.iter().map(|(_, name)| name.clone()).collect::<Vec<_>>(),
                after = ?reranked.iter().map(|(_, name)| name.clone()).collect::<Vec<_>>(),
                "gp-routing reranked tiers"
            );
        }

        GpRoutingPlan {
            ordered: reranked,
            request_features,
        }
    }

    pub fn record_attempt(
        &self,
        plan: &GpRoutingPlan,
        tier: &str,
        attempt: usize,
        duration_secs: Option<f64>,
        config: &Config,
    ) {
        let Some(features) = self.build_features(&plan.request_features, tier, attempt, config) else {
            return;
        };

        let outcome = outcome_score(duration_secs);
        let mut buffer = self.buffer.lock();
        buffer.push(features, outcome);

        let should_fit = buffer.len() >= self.model_config.min_observations
            && (!self.surrogate.is_fitted()
                || (self.model_config.refit_interval > 0
                    && buffer.requests_since_last_fit() >= self.model_config.refit_interval));

        if should_fit {
            match self.surrogate.fit(&buffer) {
                Ok(()) => {
                    buffer.mark_fitted();
                    info!(
                        observations = buffer.len(),
                        fit_count = self.surrogate.fit_count(),
                        duration_ms = self.surrogate.last_fit_duration_ms(),
                        "gp-routing surrogate fit complete"
                    );
                }
                Err(err) => {
                    warn!(error = %err, "gp-routing surrogate fit failed");
                }
            }
        }
    }

    fn build_features(
        &self,
        request_features: &RequestFeatureContext,
        tier: &str,
        attempt: usize,
        config: &Config,
    ) -> Option<FeatureVector> {
        let backend_index = *self.backend_indices.get(tier)?;
        let total_capacity = normalize_capacity(request_features.max_streams, request_features.active_streams);
        let clamped_active = request_features.active_streams.min(total_capacity);
        let idle_capacity = total_capacity.saturating_sub(clamped_active);
        let cost_tier = is_paid_tier(config, tier);

        Some(
            FeatureVector::builder()
                .prompt_length(request_features.prompt_chars)
                // Reuse the categorical priority slots as request-shape buckets:
                // plain / streaming / tool-heavy / mixed.
                .priority(request_features.request_class)
                // Reuse the verify bit as a "has system prompt" indicator.
                .has_verify(request_features.has_system_prompt)
                .dependency_count(request_features.message_count.saturating_sub(1))
                .retry_count(attempt)
                .backend_index(backend_index, self.model_config.n_backends)
                .idle_ratio(idle_capacity, total_capacity)
                .in_flight_ratio(clamped_active, total_capacity)
                .hour_of_day(request_features.hour_of_day)
                .loop_count(request_features.tool_count)
                .cost_tier(cost_tier)
                .build(),
        )
    }

    fn score_candidate(
        &self,
        mean: f32,
        variance: f32,
        candidate_idx: usize,
        epsilon_pick: Option<usize>,
    ) -> f32 {
        match self.runtime_config.acquisition {
            GpAcquisitionStrategy::Ucb => ucb_score(mean, variance, self.runtime_config.ucb_kappa),
            GpAcquisitionStrategy::Thompson => {
                let mut rng = rand::thread_rng();
                thompson_sample(mean, variance, &mut rng)
            }
            GpAcquisitionStrategy::Greedy => mean,
            GpAcquisitionStrategy::EpsilonGreedy => {
                if epsilon_pick == Some(candidate_idx) {
                    mean + 2.0
                } else {
                    mean
                }
            }
        }
    }
}

fn prompt_chars(request: &AnthropicRequest) -> usize {
    let message_bytes = request
        .messages
        .iter()
        .map(|message| message.content.to_string().len())
        .sum::<usize>();
    let system_bytes = request
        .system
        .as_ref()
        .map_or(0, |system| system.to_string().len());
    message_bytes + system_bytes
}

fn classify_request(request: &AnthropicRequest) -> u8 {
    let streaming = request.stream.unwrap_or(false);
    let tool_heavy = request.tools.as_ref().is_some_and(|tools| !tools.is_empty())
        || request.messages.iter().any(|message| message.role == "tool");

    match (streaming, tool_heavy) {
        (false, false) => 0,
        (true, false) => 1,
        (false, true) => 2,
        (true, true) => 3,
    }
}

fn normalize_capacity(max_streams: usize, active_streams: usize) -> usize {
    if max_streams == 0 {
        active_streams.saturating_add(1).max(1)
    } else {
        max_streams.max(1)
    }
}

fn is_paid_tier(config: &Config, tier: &str) -> bool {
    config.resolve_provider(tier).is_none_or(|provider| {
        let name = provider.name.to_ascii_lowercase();
        let url = provider.api_base_url.to_ascii_lowercase();
        !(name.contains("ollama")
            || name.contains("local")
            || url.contains("localhost")
            || url.contains("127.0.0.1"))
    })
}

fn outcome_score(duration_secs: Option<f64>) -> f32 {
    match duration_secs {
        Some(duration) if duration.is_finite() && duration >= 0.0 => {
            (1.0 / (1.0 + duration as f32)).clamp(0.0, 1.0)
        }
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;

    fn minimal_config() -> Config {
        let raw = json!({
            "Providers": [{
                "name": "mock",
                "api_base_url": "http://localhost:9999/v1/chat/completions",
                "api_key": "x",
                "models": ["m"]
            }],
            "Router": {
                "default": "mock,m",
                "tiers": ["mock,m"]
            }
        });
        let temp = tempfile::NamedTempFile::new().expect("temp config file");
        fs::write(temp.path(), serde_json::to_vec(&raw).expect("serialize config"))
            .expect("write config file");
        Config::from_file(temp.path().to_str().expect("config path"))
            .expect("load Config from file")
    }

    fn sample_request() -> AnthropicRequest {
        AnthropicRequest {
            model: "mock,m".to_string(),
            messages: vec![crate::router::Message {
                role: "user".to_string(),
                content: serde_json::Value::String("hello world".to_string()),
                tool_call_id: None,
            }],
            system: None,
            max_tokens: Some(64),
            temperature: Some(0.2),
            stream: Some(false),
            tools: None,
            openai_passthrough_body: None,
        }
    }

    #[test]
    fn unfitted_router_keeps_original_order() {
        let router = GpRequestRouter::new(
            GpRoutingRuntimeConfig {
                enabled: true,
                ..GpRoutingRuntimeConfig::default()
            },
            &["mock,m".to_string()],
        );
        let config = minimal_config();
        let ordered = vec![("mock,m".to_string(), "mock".to_string())];
        let plan = router.plan_rerank(&ordered, &sample_request(), &config, 0, 16, 0);
        assert_eq!(plan.ordered, ordered);
    }

    #[test]
    fn failure_observations_map_to_zero_score() {
        assert_eq!(outcome_score(None), 0.0);
        assert_eq!(outcome_score(Some(-1.0)), 0.0);
        assert!(outcome_score(Some(0.5)) > outcome_score(Some(3.0)));
    }
}