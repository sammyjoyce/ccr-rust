// SPDX-License-Identifier: AGPL-3.0-or-later
use std::collections::BTreeMap;

use lazy_static::lazy_static;
use regex::Regex;

lazy_static! {
    /// Matches grep-style output prefix: `filepath:lineno:`
    static ref GREP_LINE_RE: Regex = Regex::new(r"^(.+):(\d+):").unwrap();
}

const THRESHOLD: usize = 50;
const LINES_PER_FILE: usize = 3;

/// Compress grep-style output when there are more than 50 matches.
///
/// Groups matches by file, shows first 3 per file + "... and N more".
/// Returns `None` if ≤50 matches (no compression needed).
pub fn try_compress(text: &str) -> Option<String> {
    // Collect grep-style lines grouped by file, preserving file order.
    let mut files: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    let mut file_order: Vec<&str> = Vec::new();
    let mut match_count: usize = 0;

    for line in text.lines() {
        if let Some(caps) = GREP_LINE_RE.captures(line) {
            let file = caps.get(1).unwrap().as_str();
            match_count += 1;
            let entry = files.entry(file).or_default();
            if entry.is_empty() {
                file_order.push(file);
            }
            entry.push(line);
        }
    }

    if match_count <= THRESHOLD {
        return None;
    }

    let mut out = String::with_capacity(text.len() / 2);
    out.push_str(&format!(
        "{match_count} matches across {} files:\n\n",
        file_order.len()
    ));

    for file in &file_order {
        let lines = &files[file];
        out.push_str(&format!("{}:\n", file));
        for line in lines.iter().take(LINES_PER_FILE) {
            out.push_str("  ");
            out.push_str(line);
            out.push('\n');
        }
        if lines.len() > LINES_PER_FILE {
            out.push_str(&format!(
                "  ... and {} more\n",
                lines.len() - LINES_PER_FILE
            ));
        }
        out.push('\n');
    }

    Some(out.trim_end().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_below_threshold_returns_none() {
        let input = (1..=50)
            .map(|i| format!("file.rs:{}:content {}", i, i))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(try_compress(&input).is_none());
    }

    #[test]
    fn test_above_threshold_compresses() {
        let input = (1..=60)
            .map(|i| format!("file.rs:{}:content {}", i, i))
            .collect::<Vec<_>>()
            .join("\n");
        let result = try_compress(&input).unwrap();
        assert!(result.contains("60 matches across 1 files:"));
        assert!(result.contains("file.rs:1:content 1"));
        assert!(result.contains("file.rs:3:content 3"));
        assert!(!result.contains("file.rs:4:content 4"));
        assert!(result.contains("... and 57 more"));
    }

    #[test]
    fn test_multiple_files_grouped() {
        let mut lines = Vec::new();
        for i in 1..=30 {
            lines.push(format!("src/foo.rs:{}:match {}", i, i));
        }
        for i in 1..=25 {
            lines.push(format!("src/bar.rs:{}:match {}", i, i));
        }
        let input = lines.join("\n");
        let result = try_compress(&input).unwrap();
        assert!(result.contains("55 matches across 2 files:"));
        assert!(result.contains("src/foo.rs:"));
        assert!(result.contains("src/bar.rs:"));
        assert!(result.contains("... and 27 more")); // 30 - 3
        assert!(result.contains("... and 22 more")); // 25 - 3
    }

    #[test]
    fn test_non_grep_lines_ignored() {
        let mut lines = Vec::new();
        lines.push("-- some header --".to_string());
        for i in 1..=51 {
            lines.push(format!("file.rs:{}:content", i));
        }
        lines.push("-- footer --".to_string());
        let input = lines.join("\n");
        let result = try_compress(&input).unwrap();
        assert!(result.contains("51 matches"));
    }

    #[test]
    fn test_detects_prefix_pattern_without_content() {
        let input = (1..=51)
            .map(|i| format!("file.rs:{}:", i))
            .collect::<Vec<_>>()
            .join("\n");
        let result = try_compress(&input).unwrap();
        assert!(result.contains("51 matches across 1 files:"));
        assert!(result.contains("... and 48 more"));
    }

    #[test]
    fn test_file_with_exactly_3_matches_no_more_line() {
        let mut lines = Vec::new();
        // 3 matches in a.rs, 48 in b.rs = 51 total
        for i in 1..=3 {
            lines.push(format!("a.rs:{}:x", i));
        }
        for i in 1..=48 {
            lines.push(format!("b.rs:{}:x", i));
        }
        let input = lines.join("\n");
        let result = try_compress(&input).unwrap();
        assert!(!result.contains("a.rs:\n  a.rs:1:x\n  a.rs:2:x\n  a.rs:3:x\n  ... and"));
        assert!(result.contains("b.rs:\n  b.rs:1:x\n  b.rs:2:x\n  b.rs:3:x\n  ... and 45 more"));
    }
}
