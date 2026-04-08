// SPDX-License-Identifier: AGPL-3.0-or-later
use lazy_static::lazy_static;
use regex::Regex;

lazy_static! {
    static ref PYTEST_COUNT_RE: Regex =
        Regex::new(r"(?i)\b(\d+)\s+(passed|failed|error|errors)\b").unwrap();
    static ref PASS_FAIL_COUNT_RE: Regex = Regex::new(r"(?i)\b(\d+)\s+(passed|failed)\b").unwrap();
}

/// Compress verbose output from common test runners.
///
/// Detection:
/// - pytest: contains `====` and either `PASSED` or `FAILED`
/// - jest/vitest: contains `Test Suites:` and either `PASS ` or `FAIL `
///
/// Compression behavior:
/// - pytest: strip per-test `PASSED` lines, keep `FAILED` + `ERROR` lines, add counts summary
/// - jest/vitest: strip `✓` and `PASS ` lines, keep `✗` + `FAIL ` lines, add counts summary
pub fn try_compress(text: &str) -> Option<String> {
    if is_pytest_output(text) {
        return compress_pytest(text);
    }

    if is_jest_vitest_output(text) {
        return compress_jest_vitest(text);
    }

    None
}

fn is_pytest_output(text: &str) -> bool {
    text.contains("====") && (text.contains("PASSED") || text.contains("FAILED"))
}

fn is_jest_vitest_output(text: &str) -> bool {
    text.contains("Test Suites:") && (text.contains("PASS ") || text.contains("FAIL "))
}

fn compress_pytest(text: &str) -> Option<String> {
    let mut kept_lines: Vec<&str> = Vec::new();
    let mut stripped_passed_lines = 0usize;
    let mut failed_lines = 0usize;
    let mut error_lines = 0usize;

    for line in text.lines() {
        let trimmed = line.trim_start();

        if is_pytest_status_line(trimmed, "PASSED") {
            stripped_passed_lines += 1;
            continue;
        }

        if is_pytest_status_line(trimmed, "FAILED") || trimmed.starts_with("FAILED ") {
            failed_lines += 1;
        }

        if is_pytest_status_line(trimmed, "ERROR") || trimmed.starts_with("ERROR ") {
            error_lines += 1;
        }

        kept_lines.push(line);
    }

    if stripped_passed_lines == 0 {
        return None;
    }

    let (summary_passed, summary_failed, summary_errors) = parse_pytest_summary_counts(text);
    let passed = summary_passed.unwrap_or(stripped_passed_lines);
    let failed = summary_failed.unwrap_or(failed_lines);
    let errors = summary_errors.unwrap_or(error_lines);

    let mut out = String::with_capacity(text.len());
    out.push_str(&format!(
        "pytest summary: stripped {stripped_passed_lines} PASSED lines; passed: {passed}, failed: {failed}, errors: {errors}\n\n"
    ));
    out.push_str(kept_lines.join("\n").trim_matches('\n'));
    Some(out.trim_end().to_string())
}

fn compress_jest_vitest(text: &str) -> Option<String> {
    let mut kept_lines: Vec<&str> = Vec::new();
    let mut stripped_passed_tests = 0usize;
    let mut failed_tests = 0usize;
    let mut stripped_passed_suites = 0usize;
    let mut failed_suites = 0usize;

    let mut suite_passed: Option<usize> = None;
    let mut suite_failed: Option<usize> = None;
    let mut test_passed: Option<usize> = None;
    let mut test_failed: Option<usize> = None;

    for line in text.lines() {
        let trimmed = line.trim_start();

        if trimmed.starts_with("PASS ") {
            stripped_passed_suites += 1;
            continue;
        }

        if trimmed.starts_with("FAIL ") {
            failed_suites += 1;
            kept_lines.push(line);
            continue;
        }

        if trimmed.starts_with('✓') {
            stripped_passed_tests += 1;
            continue;
        }

        if trimmed.starts_with('✗') {
            failed_tests += 1;
        }

        if trimmed.starts_with("Test Suites:") {
            let (passed, failed) = parse_pass_fail_counts(trimmed);
            suite_passed = passed.or(suite_passed);
            suite_failed = failed.or(suite_failed);
        }

        if trimmed.starts_with("Tests:") {
            let (passed, failed) = parse_pass_fail_counts(trimmed);
            test_passed = passed.or(test_passed);
            test_failed = failed.or(test_failed);
        }

        kept_lines.push(line);
    }

    if stripped_passed_tests == 0 && stripped_passed_suites == 0 {
        return None;
    }

    let passed_suites = suite_passed.unwrap_or(stripped_passed_suites);
    let failed_suites = suite_failed.unwrap_or(failed_suites);
    let passed_tests = test_passed.unwrap_or(stripped_passed_tests);
    let failed_tests = test_failed.unwrap_or(failed_tests);

    let mut out = String::with_capacity(text.len());
    out.push_str(&format!(
        "jest/vitest summary: {passed_suites} passed suites (stripped), {failed_suites} failed suites, {passed_tests} passed tests (stripped), {failed_tests} failed tests\n\n"
    ));
    out.push_str(kept_lines.join("\n").trim_matches('\n'));
    Some(out.trim_end().to_string())
}

fn is_pytest_status_line(line: &str, status: &str) -> bool {
    let marker = format!(" {status}");
    if let Some(idx) = line.find(&marker) {
        return line[..idx].contains("::");
    }

    false
}

fn parse_pytest_summary_counts(text: &str) -> (Option<usize>, Option<usize>, Option<usize>) {
    let mut passed: Option<usize> = None;
    let mut failed: Option<usize> = None;
    let mut errors: Option<usize> = None;

    for line in text.lines().rev() {
        if !line.contains("====") {
            continue;
        }

        let mut matched = false;
        for captures in PYTEST_COUNT_RE.captures_iter(line) {
            matched = true;

            let count = match captures
                .get(1)
                .and_then(|m| m.as_str().parse::<usize>().ok())
            {
                Some(v) => v,
                None => continue,
            };

            let label = captures.get(2).map(|m| m.as_str()).unwrap_or_default();
            if label.eq_ignore_ascii_case("passed") {
                passed = Some(count);
            } else if label.eq_ignore_ascii_case("failed") {
                failed = Some(count);
            } else if label.eq_ignore_ascii_case("error") || label.eq_ignore_ascii_case("errors") {
                errors = Some(count);
            }
        }

        if matched {
            break;
        }
    }

    (passed, failed, errors)
}

fn parse_pass_fail_counts(line: &str) -> (Option<usize>, Option<usize>) {
    let mut passed: Option<usize> = None;
    let mut failed: Option<usize> = None;

    for captures in PASS_FAIL_COUNT_RE.captures_iter(line) {
        let count = match captures
            .get(1)
            .and_then(|m| m.as_str().parse::<usize>().ok())
        {
            Some(v) => v,
            None => continue,
        };

        let label = captures.get(2).map(|m| m.as_str()).unwrap_or_default();
        if label.eq_ignore_ascii_case("passed") {
            passed = Some(count);
        } else if label.eq_ignore_ascii_case("failed") {
            failed = Some(count);
        }
    }

    (passed, failed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compresses_pytest_strip_passed_keep_failures_and_errors() {
        let input = [
            "============================= test session starts =============================",
            "tests/test_math.py::test_add PASSED",
            "tests/test_math.py::test_div FAILED",
            "ERROR tests/test_math.py::test_fixture",
            "FAILED tests/test_math.py::test_div - AssertionError",
            "=================== 1 failed, 1 passed, 1 error in 0.12s ====================",
        ]
        .join("\n");

        let output = try_compress(&input).expect("expected pytest compression");
        assert!(!output.contains("test_add PASSED"));
        assert!(output.contains("test_div FAILED"));
        assert!(output.contains("ERROR tests/test_math.py::test_fixture"));
        assert!(output.contains("passed: 1, failed: 1, errors: 1"));
    }

    #[test]
    fn compresses_jest_vitest_strip_checkmarks_keep_fail_marks() {
        let input = [
            "PASS src/math.test.ts",
            " FAIL src/divide.test.ts",
            " ✓ adds numbers",
            " ✗ throws on divide by zero",
            "Test Suites: 1 failed, 1 passed, 2 total",
            "Tests:       1 failed, 1 passed, 2 total",
        ]
        .join("\n");

        let output = try_compress(&input).expect("expected jest/vitest compression");
        assert!(!output.contains("✓ adds numbers"));
        assert!(output.contains("✗ throws on divide by zero"));
        assert!(!output.contains("PASS src/math.test.ts"));
        assert!(output.contains("FAIL src/divide.test.ts"));
        assert!(output.contains("jest/vitest summary: 1 passed suites (stripped), 1 failed suites, 1 passed tests (stripped), 1 failed tests"));
    }

    #[test]
    fn pytest_detection_requires_equals_sections() {
        let input = "tests/test_math.py::test_add PASSED\ntests/test_math.py::test_div FAILED";
        assert!(try_compress(input).is_none());
    }

    #[test]
    fn jest_detection_requires_test_suites() {
        let input = "PASS src/math.test.ts\n ✓ adds numbers";
        assert!(try_compress(input).is_none());
    }
}
