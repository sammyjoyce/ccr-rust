use anyhow::Result;

pub struct McpArgs {
    pub level: String,
    pub wrap: Vec<String>,
    pub include: Option<String>,
    pub exclude: Option<String>,
}

pub async fn run(args: McpArgs) -> Result<()> {
    tracing::info!(
        level = %args.level,
        wrap = ?args.wrap,
        include = ?args.include,
        exclude = ?args.exclude,
        "Starting MCP server"
    );
    todo!("MCP server implementation")
}
