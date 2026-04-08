// SPDX-License-Identifier: AGPL-3.0-or-later
const COMPILING_PREFIX: &str = "Compiling ";
const TEST_RESULT_PREFIX: &str = "test result:";
const CMAKE_PREFIX: &str = "-- ";
const CMAKE_CHECKING_PREFIX: &str = "-- Checking";
const CMAKE_FOUND_PREFIX: &str = "-- Found";

/// Compress cargo/cmake-style build output.
///
/// Handles:
/// - cargo build/check: summarize `Compiling` lines, keep diagnostics
/// - cargo test: keep failure blocks + `test result:` summaries
/// - cmake configure: strip noisy `-- Checking` / `-- Found` lines
pub fn try_compress(text: &str) -> Option<String> {
    let has_compiling = text
        .lines()
        .any(|line| line.trim_start().starts_with(COMPILING_PREFIX));
    let has_test_result = text
        .lines()
        .any(|line| line.trim_start().starts_with(TEST_RESULT_PREFIX));
    let has_cmake = text
        .lines()
        .any(|line| line.trim_start().starts_with(CMAKE_PREFIX));

    let mut best: Option<String> = None;

    if has_test_result {
        if let Some(candidate) = compress_cargo_test(text) {
            best = pick_shorter(best, candidate);
        }
    }

    // `cargo test` output often contains compile lines; prefer test-specific
    // compression semantics when `test result:` is present.
    if has_compiling && !has_test_result {
        if let Some(candidate) = compress_cargo_build_check(text) {
            best = pick_shorter(best, candidate);
        }
    }

    if has_cmake {
        if let Some(candidate) = compress_cmake(text) {
            best = pick_shorter(best, candidate);
        }
    }

    best
}

fn pick_shorter(current: Option<String>, candidate: String) -> Option<String> {
    match current {
        None => Some(candidate),
        Some(existing) => {
            if candidate.len() < existing.len() {
                Some(candidate)
            } else {
                Some(existing)
            }
        }
    }
}

fn compress_cargo_build_check(text: &str) -> Option<String> {
    let mut compiling_count = 0usize;
    let mut warning_count = 0usize;
    let mut error_count = 0usize;
    let mut kept_lines: Vec<&str> = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with(COMPILING_PREFIX) {
            compiling_count += 1;
            continue;
        }

        if is_cargo_warning_line(trimmed) {
            warning_count += 1;
            kept_lines.push(line);
            continue;
        }

        if is_cargo_error_line(trimmed) {
            error_count += 1;
            kept_lines.push(line);
        }
    }

    if compiling_count == 0 {
        return None;
    }

    let mut out = format!(
        "[cargo] elided {compiling_count} Compiling lines; kept {warning_count} warning lines and {error_count} error lines"
    );
    if !kept_lines.is_empty() {
        out.push('\n');
        out.push_str(kept_lines.join("\n").trim_matches('\n'));
    }

    if out.len() < text.len() {
        Some(out)
    } else {
        None
    }
}

fn compress_cargo_test(text: &str) -> Option<String> {
    let mut kept_lines: Vec<&str> = Vec::new();
    let mut in_failures = false;
    let mut saw_summary = false;

    for line in text.lines() {
        let trimmed = line.trim_start();

        if trimmed.starts_with(TEST_RESULT_PREFIX) {
            saw_summary = true;
            in_failures = false;
            kept_lines.push(line);
            continue;
        }

        if trimmed == "failures:" {
            in_failures = true;
            kept_lines.push(line);
            continue;
        }

        if in_failures {
            kept_lines.push(line);
            continue;
        }
    }

    if !saw_summary {
        return None;
    }

    let out = kept_lines.join("\n").trim().to_string();
    if out.len() < text.len() {
        Some(out)
    } else {
        None
    }
}

fn is_cargo_warning_line(line: &str) -> bool {
    line.starts_with("warning:") || line.starts_with("warning[")
}

fn is_cargo_error_line(line: &str) -> bool {
    line.starts_with("error:") || line.starts_with("error[")
}

fn compress_cmake(text: &str) -> Option<String> {
    let mut removed = false;
    let mut kept_lines: Vec<&str> = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with(CMAKE_CHECKING_PREFIX) || trimmed.starts_with(CMAKE_FOUND_PREFIX) {
            removed = true;
            continue;
        }
        kept_lines.push(line);
    }

    if !removed {
        return None;
    }

    let out = kept_lines.join("\n");
    if out.len() < text.len() {
        Some(out)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compresses_cargo_build_compiling_lines() {
        let input = [
            "   Compiling foo v0.1.0 (/tmp/foo)",
            "   Compiling bar v0.1.0 (/tmp/bar)",
            "    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.62s",
            "warning: unused variable: `x`",
            "error[E0425]: cannot find value `y` in this scope",
        ]
        .join("\n");
        let output = try_compress(&input).unwrap();

        assert!(output.contains("[cargo] elided 2 Compiling lines"));
        assert!(output.contains("warning: unused variable: `x`"));
        assert!(output.contains("error[E0425]: cannot find value `y` in this scope"));
        assert!(!output.contains("Compiling foo"));
        assert!(!output.contains("Finished `dev` profile"));
    }

    #[test]
    fn compresses_cargo_test_to_failures_and_summary() {
        let input = [
            "running 2 tests",
            "test tests::ok ... ok",
            "test tests::bad ... FAILED",
            "",
            "failures:",
            "",
            "---- tests::bad stdout ----",
            "thread 'tests::bad' panicked at src/lib.rs:10:9:",
            "boom",
            "",
            "failures:",
            "    tests::bad",
            "",
            "test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s",
            "error: test failed, to rerun pass `--lib`",
        ]
        .join("\n");
        let output = try_compress(&input).unwrap();

        assert!(!output.contains("running 2 tests"));
        assert!(output.contains("failures:"));
        assert!(output.contains("---- tests::bad stdout ----"));
        assert!(output.contains("test result: FAILED."));
        assert!(!output.contains("error: test failed, to rerun pass `--lib`"));
    }

    #[test]
    fn compresses_cargo_test_success_to_summary_only() {
        let input = [
            "running 2 tests",
            "test tests::ok ... ok",
            "test tests::also_ok ... ok",
            "",
            "test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s",
        ]
        .join("\n");
        let output = try_compress(&input).unwrap();

        assert_eq!(
            output,
            "test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s"
        );
    }

    #[test]
    fn compresses_cmake_checking_and_found_lines() {
        let input = [
            "-- The C compiler identification is AppleClang 15.0.0.15000309",
            "-- Checking for module 'libfoo'",
            "--   Found libfoo, version 1.0.0",
            "-- Found Threads: TRUE",
            "CMake Error at CMakeLists.txt:42 (message):",
            "  missing dependency",
        ]
        .join("\n");
        let output = try_compress(&input).unwrap();

        assert!(output.contains("-- The C compiler identification is AppleClang 15.0.0.15000309"));
        assert!(output.contains("CMake Error at CMakeLists.txt:42 (message):"));
        assert!(!output.contains("-- Checking for module 'libfoo'"));
        assert!(!output.contains("-- Found Threads: TRUE"));
    }

    #[test]
    fn returns_none_when_no_known_patterns() {
        let input = "plain output line 1\nplain output line 2";
        assert!(try_compress(input).is_none());
    }
}
