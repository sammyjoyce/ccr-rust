use std::collections::{BTreeMap, HashMap, HashSet};

use lazy_static::lazy_static;
use regex::Regex;

lazy_static! {
    // Detection patterns (required order: status -> diff -> log -> branch).
    static ref STATUS_DETECT_RE: Regex = Regex::new(r"(?m)^On branch ").unwrap();
    static ref DIFF_DETECT_RE: Regex = Regex::new(r"(?m)^diff --git ").unwrap();
    static ref LOG_DETECT_RE: Regex = Regex::new(r"(?m)^commit [0-9a-f]{40}$").unwrap();
    static ref BRANCH_DETECT_RE: Regex = Regex::new(r"(?m)^[* ] +\S+").unwrap();

    // Status parsing.
    static ref ON_BRANCH_RE: Regex = Regex::new(r"^On branch (.+)$").unwrap();
    static ref STATUS_SECTION_RE: Regex = Regex::new(
        r"^(Changes not staged for commit|Changes to be committed|Untracked files|Unmerged paths):$"
    )
    .unwrap();
    static ref STATUS_ENTRY_RE: Regex = Regex::new(
        r"^\s*(modified|deleted|new file|renamed|copied|both modified|both added|both deleted):\s+(.+)$"
    )
    .unwrap();
    static ref STATUS_PORCELAIN_RE: Regex = Regex::new(r"^([MADRCU?!][ MADRCU?!]|[ ][MADRCU?!])\s+(.+)$").unwrap();

    // Diff parsing.
    static ref DIFF_FILE_RE: Regex = Regex::new(r"^diff --git a/(.+?) b/(.+)$").unwrap();
    static ref DIFF_HUNK_RE: Regex = Regex::new(r"^@@ .+ @@").unwrap();

    // Log parsing.
    static ref LOG_COMMIT_RE: Regex = Regex::new(r"^commit ([0-9a-f]{40})$").unwrap();
    static ref LOG_DATE_ISO_RE: Regex = Regex::new(r"^\s*Date:\s+([0-9]{4}-[0-9]{2}-[0-9]{2})").unwrap();
    static ref LOG_DATE_DEFAULT_RE: Regex = Regex::new(
        r"^\s*Date:\s+\w+\s+([A-Za-z]{3})\s+(\d{1,2})\s+\d{2}:\d{2}:\d{2}\s+(\d{4})"
    )
    .unwrap();

    // Branch parsing.
    static ref BRANCH_LINE_RE: Regex = Regex::new(r"^\s*([* ])\s+(.+?)\s*$").unwrap();
}

const DIFF_SUMMARY_THRESHOLD_LINES: usize = 100;
const STATUS_ORDER: [&str; 7] = ["M", "A", "D", "R", "C", "U", "??"];

pub fn try_compress(text: &str) -> Option<String> {
    if STATUS_DETECT_RE.is_match(text) {
        return compress_status(text);
    }

    if DIFF_DETECT_RE.is_match(text) {
        return compress_diff(text);
    }

    if LOG_DETECT_RE.is_match(text) {
        return compress_log(text);
    }

    if BRANCH_DETECT_RE.is_match(text) {
        return compress_branch(text);
    }

    None
}

fn compress_status(text: &str) -> Option<String> {
    let mut branch = String::new();
    let mut grouped: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut grouped_seen: HashMap<String, HashSet<String>> = HashMap::new();
    let mut in_untracked = false;

    for line in text.lines() {
        if let Some(caps) = ON_BRANCH_RE.captures(line) {
            branch = caps.get(1).unwrap().as_str().trim().to_string();
            continue;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if STATUS_SECTION_RE.is_match(trimmed) {
            in_untracked = trimmed == "Untracked files:";
            continue;
        }

        if trimmed.starts_with('(')
            || trimmed.starts_with("Your branch")
            || trimmed.starts_with("nothing to commit")
            || trimmed.starts_with("no changes added")
            || trimmed.starts_with("no changes yet")
            || trimmed.starts_with("use ")
        {
            continue;
        }

        if let Some(caps) = STATUS_ENTRY_RE.captures(line) {
            in_untracked = false;
            let kind = caps.get(1).unwrap().as_str();
            let file = caps.get(2).unwrap().as_str().trim().to_string();
            let code = match kind {
                "modified" => "M",
                "deleted" => "D",
                "new file" => "A",
                "renamed" => "R",
                "copied" => "C",
                "both modified" | "both added" | "both deleted" => "U",
                _ => continue,
            };
            push_grouped_file(&mut grouped, &mut grouped_seen, code, file);
            continue;
        }

        if in_untracked {
            let candidate = trimmed.trim_start_matches(':').trim();
            if !candidate.is_empty() {
                push_grouped_file(&mut grouped, &mut grouped_seen, "??", candidate.to_string());
            }
            continue;
        }

        if let Some(caps) = STATUS_PORCELAIN_RE.captures(line) {
            let raw_code = caps.get(1).unwrap().as_str().trim();
            let file = caps.get(2).unwrap().as_str().trim().to_string();
            let code = normalize_porcelain_status(raw_code);
            push_grouped_file(&mut grouped, &mut grouped_seen, code, file);
            continue;
        }
    }

    if branch.is_empty() && grouped.is_empty() {
        return None;
    }

    let mut out: Vec<String> = Vec::new();
    if !branch.is_empty() {
        out.push(format!("branch {branch}"));
    }

    for code in STATUS_ORDER {
        if let Some(files) = grouped.get(code) {
            if !files.is_empty() {
                out.push(format!("{code} {}", files.join(", ")));
            }
        }
    }

    for (code, files) in &grouped {
        if STATUS_ORDER.contains(&code.as_str()) || files.is_empty() {
            continue;
        }
        out.push(format!("{code} {}", files.join(", ")));
    }

    if out.is_empty() {
        None
    } else {
        Some(out.join("\n"))
    }
}

fn push_grouped_file(
    grouped: &mut BTreeMap<String, Vec<String>>,
    grouped_seen: &mut HashMap<String, HashSet<String>>,
    code: &str,
    file: String,
) {
    let seen = grouped_seen.entry(code.to_string()).or_default();
    if seen.insert(file.clone()) {
        grouped.entry(code.to_string()).or_default().push(file);
    }
}

fn normalize_porcelain_status(raw_code: &str) -> &str {
    if raw_code.contains("??") {
        return "??";
    }
    if raw_code.contains('M') {
        return "M";
    }
    if raw_code.contains('A') {
        return "A";
    }
    if raw_code.contains('D') {
        return "D";
    }
    if raw_code.contains('R') {
        return "R";
    }
    if raw_code.contains('C') {
        return "C";
    }
    if raw_code.contains('U') {
        return "U";
    }
    raw_code
}

#[derive(Debug, Default, Clone)]
struct DiffStats {
    hunks: usize,
    adds: usize,
    dels: usize,
}

fn compress_diff(text: &str) -> Option<String> {
    let mut current_file = String::new();
    let mut last_emitted_file = String::new();
    let mut in_hunk = false;
    let mut saw_diff = false;
    let mut out_lines: Vec<String> = Vec::new();

    let mut stats: HashMap<String, DiffStats> = HashMap::new();
    let mut file_order: Vec<String> = Vec::new();
    let input_line_count = text.lines().count();

    for line in text.lines() {
        if let Some(caps) = DIFF_FILE_RE.captures(line) {
            saw_diff = true;
            let file = caps.get(2).unwrap().as_str().to_string();
            current_file = file.clone();
            in_hunk = false;

            if !stats.contains_key(&file) {
                stats.insert(file.clone(), DiffStats::default());
                file_order.push(file);
            }
            continue;
        }

        if DIFF_HUNK_RE.is_match(line) {
            in_hunk = true;
            if !current_file.is_empty() && current_file != last_emitted_file {
                out_lines.push(current_file.clone());
                last_emitted_file = current_file.clone();
            }
            out_lines.push(line.to_string());
            if let Some(file_stats) = stats.get_mut(&current_file) {
                file_stats.hunks += 1;
            }
            continue;
        }

        if !in_hunk {
            continue;
        }

        if line.starts_with('+') && !line.starts_with("+++") {
            out_lines.push(line.to_string());
            if let Some(file_stats) = stats.get_mut(&current_file) {
                file_stats.adds += 1;
            }
            continue;
        }

        if line.starts_with('-') && !line.starts_with("---") {
            out_lines.push(line.to_string());
            if let Some(file_stats) = stats.get_mut(&current_file) {
                file_stats.dels += 1;
            }
        }
    }

    if !saw_diff {
        return None;
    }

    if input_line_count > DIFF_SUMMARY_THRESHOLD_LINES
        || out_lines.len() > DIFF_SUMMARY_THRESHOLD_LINES
    {
        return Some(summarize_diff(input_line_count, &file_order, &stats));
    }

    if out_lines.is_empty() {
        return Some(summarize_diff(input_line_count, &file_order, &stats));
    }

    Some(out_lines.join("\n"))
}

fn summarize_diff(
    total_input_lines: usize,
    file_order: &[String],
    stats: &HashMap<String, DiffStats>,
) -> String {
    let total_files = file_order.len();
    let total_hunks: usize = stats.values().map(|s| s.hunks).sum();
    let total_adds: usize = stats.values().map(|s| s.adds).sum();
    let total_dels: usize = stats.values().map(|s| s.dels).sum();

    let mut out = String::new();
    out.push_str(&format!(
        "diff summary: {total_files} files, {total_hunks} hunks, +{total_adds}/-{total_dels} ({total_input_lines} lines)\n"
    ));

    const MAX_FILES: usize = 40;
    for file in file_order.iter().take(MAX_FILES) {
        if let Some(s) = stats.get(file) {
            out.push_str(&format!(
                "{file}: {} hunks, +{}/-{}\n",
                s.hunks, s.adds, s.dels
            ));
        }
    }

    if total_files > MAX_FILES {
        out.push_str(&format!("... and {} more files", total_files - MAX_FILES));
    } else {
        out = out.trim_end().to_string();
    }

    out
}

fn compress_log(text: &str) -> Option<String> {
    let mut out: Vec<String> = Vec::new();
    let mut current_hash: Option<String> = None;
    let mut current_date = String::new();
    let mut current_subject: Option<String> = None;

    let flush_current =
        |out: &mut Vec<String>, hash: &Option<String>, date: &str, subject: &Option<String>| {
            if let Some(h) = hash {
                let d = if date.is_empty() {
                    "unknown-date"
                } else {
                    date
                };
                let s = subject.as_deref().unwrap_or("(no subject)");
                out.push(format!("{h} {d} {s}"));
            }
        };

    for line in text.lines() {
        if let Some(caps) = LOG_COMMIT_RE.captures(line) {
            flush_current(&mut out, &current_hash, &current_date, &current_subject);

            let hash = caps.get(1).unwrap().as_str();
            current_hash = Some(hash[..7].to_string());
            current_date.clear();
            current_subject = None;
            continue;
        }

        if current_hash.is_none() {
            continue;
        }

        if let Some(caps) = LOG_DATE_ISO_RE.captures(line) {
            current_date = caps.get(1).unwrap().as_str().to_string();
            continue;
        }

        if let Some(caps) = LOG_DATE_DEFAULT_RE.captures(line) {
            let month = caps.get(1).unwrap().as_str();
            let day = caps.get(2).unwrap().as_str().parse::<u32>().unwrap_or(1);
            let year = caps.get(3).unwrap().as_str();
            current_date = format!("{year}-{}-{day:02}", month_to_num(month));
            continue;
        }

        if current_subject.is_none() && line.starts_with("    ") {
            let subject = line.trim();
            if !subject.is_empty() {
                current_subject = Some(subject.to_string());
            }
        }
    }

    flush_current(&mut out, &current_hash, &current_date, &current_subject);

    if out.is_empty() {
        None
    } else {
        Some(out.join("\n"))
    }
}

fn month_to_num(month: &str) -> &'static str {
    match month {
        "Jan" => "01",
        "Feb" => "02",
        "Mar" => "03",
        "Apr" => "04",
        "May" => "05",
        "Jun" => "06",
        "Jul" => "07",
        "Aug" => "08",
        "Sep" => "09",
        "Oct" => "10",
        "Nov" => "11",
        "Dec" => "12",
        _ => "01",
    }
}

fn compress_branch(text: &str) -> Option<String> {
    let mut branches: Vec<String> = Vec::new();
    let mut non_empty = 0usize;

    for line in text.lines() {
        let trimmed = line.trim_end();
        if trimmed.trim().is_empty() {
            continue;
        }
        non_empty += 1;

        let caps = BRANCH_LINE_RE.captures(trimmed)?;
        let marker = caps.get(1).unwrap().as_str();
        let name = caps.get(2).unwrap().as_str().trim();
        if marker == "*" {
            branches.push(format!("*{name}"));
        } else {
            branches.push(name.to_string());
        }
    }

    if non_empty == 0 || branches.len() != non_empty {
        return None;
    }

    Some(branches.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compresses_git_status_groups_files() {
        let input = r#"On branch main
Your branch is up to date with 'origin/main'.

Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        modified:   src/lib.rs
        new file:   src/new.rs

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
        modified:   README.md
        deleted:    old.txt

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        scratch.log
"#;

        let result = try_compress(input).unwrap();
        assert!(result.contains("branch main"));
        assert!(result.contains("M src/lib.rs, README.md"));
        assert!(result.contains("A src/new.rs"));
        assert!(result.contains("D old.txt"));
        assert!(result.contains("?? scratch.log"));
        assert!(!result.contains("Changes not staged for commit"));
    }

    #[test]
    fn compresses_git_diff_to_hunks() {
        let input = r#"diff --git a/src/lib.rs b/src/lib.rs
index 1234..5678 100644
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -1,4 +1,4 @@
 use std::fmt;
-fn old_fn() {}
+fn new_fn() {}
 fn keep() {}
"#;

        let result = try_compress(input).unwrap();
        assert!(result.contains("src/lib.rs"));
        assert!(result.contains("@@ -1,4 +1,4 @@"));
        assert!(result.contains("-fn old_fn() {}"));
        assert!(result.contains("+fn new_fn() {}"));
        assert!(!result.contains("diff --git"));
        assert!(!result.contains("index 1234..5678"));
    }

    #[test]
    fn summarizes_large_git_diff() {
        let mut input = String::new();
        input.push_str("diff --git a/a.rs b/a.rs\n");
        input.push_str("@@ -1,1 +1,1 @@\n");
        for i in 0..120 {
            input.push_str(&format!("-old_{i}\n+new_{i}\n"));
        }

        let result = try_compress(&input).unwrap();
        assert!(result.starts_with("diff summary:"));
        assert!(result.contains("a.rs:"));
        assert!(result.contains("hunks"));
    }

    #[test]
    fn compresses_git_log_to_oneliners() {
        let input = r#"commit 1234567890abcdef1234567890abcdef12345678
Author: Dev <dev@example.com>
Date:   Wed Apr 1 12:34:56 2026 -0700

    First change

commit aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
Author: Dev <dev@example.com>
Date:   2026-03-31 09:00:00 -0700

    Second change
"#;

        let result = try_compress(input).unwrap();
        let lines: Vec<&str> = result.lines().collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "1234567 2026-04-01 First change");
        assert_eq!(lines[1], "aaaaaaa 2026-03-31 Second change");
    }

    #[test]
    fn compresses_git_branch_to_csv() {
        let input = r#"* main
  feature/refactor
  release/1.0
"#;

        let result = try_compress(input).unwrap();
        assert_eq!(result, "*main, feature/refactor, release/1.0");
    }

    #[test]
    fn unrelated_text_returns_none() {
        let input = "this is unrelated output";
        assert!(try_compress(input).is_none());
    }
}
