//! Google OAuth2 token management for GCP Code Assist API.
//!
//! Handles token caching and automatic refresh using credentials from
//! `~/.gemini/oauth_creds.json` (written by the Gemini CLI on first auth).
//!
//! The OAuth client ID and secret are from Google's public Gemini CLI OAuth app
//! (distributed in the `@anthropic/gemini-cli` npm package). They identify the
//! application, not the user — the refresh token is the real credential.

use anyhow::{Context, Result};
use serde::Deserialize;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Google OAuth2 token endpoint.
const TOKEN_ENDPOINT: &str = "https://oauth2.googleapis.com/token";

/// Gemini CLI's public OAuth client ID (embedded in the npm package).
pub const GEMINI_CLI_CLIENT_ID: &str =
    "REDACTED_GOOGLE_OAUTH_CLIENT_ID";

/// Gemini CLI's public OAuth client secret (embedded in the npm package).
pub const GEMINI_CLI_CLIENT_SECRET: &str = "REDACTED_GOOGLE_OAUTH_CLIENT_SECRET";

/// Refresh token 5 minutes before expiry.
const EXPIRY_BUFFER_SECS: i64 = 300;

/// Cached OAuth2 access token with expiry.
#[derive(Debug, Clone)]
struct CachedToken {
    access_token: String,
    expires_at: chrono::DateTime<chrono::Utc>,
}

/// On-disk format for `~/.gemini/oauth_creds.json`.
#[derive(Debug, Deserialize)]
struct OAuthCredsFile {
    refresh_token: String,
    #[serde(default)]
    access_token: Option<String>,
    /// Milliseconds since epoch.
    #[serde(default)]
    expiry_date: Option<i64>,
}

/// Token refresh response from Google's OAuth2 endpoint.
#[derive(Debug, Deserialize)]
struct TokenRefreshResponse {
    access_token: String,
    expires_in: i64,
}

/// Thread-safe OAuth2 token cache with automatic refresh.
pub struct GoogleOAuthCache {
    client_id: String,
    client_secret: String,
    refresh_token: String,
    cached: RwLock<Option<CachedToken>>,
    http_client: reqwest::Client,
}

impl GoogleOAuthCache {
    /// Load credentials from `~/.gemini/oauth_creds.json` and create a cache.
    ///
    /// Uses the Gemini CLI's public OAuth client ID/secret by default.
    pub fn from_gemini_creds() -> Result<Self> {
        Self::from_gemini_creds_with_client(GEMINI_CLI_CLIENT_ID, GEMINI_CLI_CLIENT_SECRET)
    }

    /// Load credentials with custom client ID/secret.
    pub fn from_gemini_creds_with_client(client_id: &str, client_secret: &str) -> Result<Self> {
        let creds_path = dirs::home_dir()
            .context("No home directory")?
            .join(".gemini")
            .join("oauth_creds.json");

        let contents = std::fs::read_to_string(&creds_path)
            .with_context(|| format!("Failed to read {}", creds_path.display()))?;

        let creds_file: OAuthCredsFile = serde_json::from_str(&contents)
            .with_context(|| format!("Failed to parse {}", creds_path.display()))?;

        // Pre-populate cache if the on-disk token is still valid.
        let cached = match (creds_file.access_token, creds_file.expiry_date) {
            (Some(ref token), Some(expiry_ms)) => {
                let expires_at = chrono::DateTime::from_timestamp_millis(expiry_ms)
                    .unwrap_or_else(chrono::Utc::now);
                let cutoff = chrono::Utc::now() + chrono::Duration::seconds(EXPIRY_BUFFER_SECS);
                if expires_at > cutoff {
                    debug!(
                        "Pre-populated Google OAuth cache from disk (expires {})",
                        expires_at
                    );
                    Some(CachedToken {
                        access_token: token.clone(),
                        expires_at,
                    })
                } else {
                    None
                }
            }
            _ => None,
        };

        Ok(Self {
            client_id: client_id.to_string(),
            client_secret: client_secret.to_string(),
            refresh_token: creds_file.refresh_token,
            cached: RwLock::new(cached),
            http_client: reqwest::Client::new(),
        })
    }

    /// Get a valid access token, refreshing if necessary.
    pub async fn get_access_token(&self) -> Result<String> {
        // Fast path: check if cached token is still valid.
        {
            let cached = self.cached.read().await;
            if let Some(ref token) = *cached {
                let cutoff = chrono::Utc::now() + chrono::Duration::seconds(EXPIRY_BUFFER_SECS);
                if token.expires_at > cutoff {
                    return Ok(token.access_token.clone());
                }
            }
        }

        // Slow path: refresh token.
        self.refresh().await
    }

    /// Refresh the access token using the refresh token.
    async fn refresh(&self) -> Result<String> {
        info!("Refreshing Google OAuth2 access token");

        let resp = self
            .http_client
            .post(TOKEN_ENDPOINT)
            .form(&[
                ("client_id", self.client_id.as_str()),
                ("client_secret", self.client_secret.as_str()),
                ("refresh_token", self.refresh_token.as_str()),
                ("grant_type", "refresh_token"),
            ])
            .send()
            .await
            .context("Failed to send token refresh request")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Google OAuth token refresh failed ({}): {}", status, body);
        }

        let token_resp: TokenRefreshResponse = resp
            .json()
            .await
            .context("Failed to parse token refresh response")?;

        let expires_at = chrono::Utc::now() + chrono::Duration::seconds(token_resp.expires_in);
        let access_token = token_resp.access_token.clone();

        // Update cache.
        let mut cached = self.cached.write().await;
        *cached = Some(CachedToken {
            access_token: access_token.clone(),
            expires_at,
        });

        info!(
            "Google OAuth2 token refreshed, expires in {}s",
            token_resp.expires_in
        );
        Ok(access_token)
    }
}
