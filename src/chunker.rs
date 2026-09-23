//! Splits file contents into embeddable chunks.
//!
//! Recognized code extensions get boundary-aware splitting via tree-sitter:
//! top-level nodes (functions, classes, impls, ...) are packed into windows
//! of up to [`ChunkingConfig::chunk_lines`] lines / [`ChunkingConfig::max_chars`]
//! characters, with [`ChunkingConfig::overlap_lines`] lines of trailing
//! context carried into the next window — mirroring the tuning the Python
//! original used (`chunk_lines=80`, `chunk_lines_overlap=15`,
//! `max_chars=1500`). Everything else falls back to the same
//! line/char-budget windowing applied directly to the raw text.
//!
//! Determinism matters here beyond output quality: the indexing pipeline's
//! idempotency (see `db.rs`) depends on chunking the same file content
//! producing the same chunks every time, so re-indexing an unchanged file
//! is a true no-op rather than a spurious upsert.

use tree_sitter::{Language, Parser};

/// Tuning for both the code-aware and plain-text splitters.
#[derive(Debug, Clone, Copy)]
pub struct ChunkingConfig {
    /// Maximum number of lines per chunk.
    pub chunk_lines: usize,
    /// Lines of trailing context repeated at the start of the next chunk.
    pub overlap_lines: usize,
    /// Maximum characters per chunk (checked alongside `chunk_lines`;
    /// whichever budget is hit first ends the chunk).
    pub max_chars: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            chunk_lines: 80,
            overlap_lines: 15,
            max_chars: 1500,
        }
    }
}

/// One chunk of a file, with 0-indexed, inclusive line bounds into the
/// original content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub content: String,
    pub start_line: u32,
    pub end_line: u32,
}

/// Splits `content` into chunks appropriate for `extension` (e.g. `"rs"`,
/// without the leading dot): tree-sitter-based for recognized code
/// extensions, falling back to plain line/char-budget windowing for
/// anything else (including code whose parse fails).
pub fn chunk_file(extension: &str, content: &str, config: &ChunkingConfig) -> Vec<Chunk> {
    language_for_extension(extension)
        .and_then(|language| chunk_code(content, language, config))
        .unwrap_or_else(|| chunk_text(content, config))
}

/// Extensions this daemon never indexes: images, audio/video, archives,
/// compiled/binary artifacts, fonts, and other non-text formats. Ported
/// from the Python original's `DEFAULT_BINARY_EXTENSIONS`.
const BINARY_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "bmp", "ico", "webp", "tiff", "exr", "hdr", "svg", "psd", "ai",
    "eps", "mp3", "wav", "mp4", "avi", "mov", "webm", "flac", "ogg", "m4a", "aac", "wma", "flv",
    "mkv", "wmv", "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "odt", "zip", "tar", "gz",
    "7z", "rar", "iso", "dmg", "pkg", "deb", "rpm", "msi", "apk", "xz", "bz2", "exe", "dll", "so",
    "dylib", "class", "pyc", "o", "obj", "lib", "a", "out", "app", "jar", "ttf", "otf", "woff",
    "woff2", "eot", "bin", "dat", "db", "sqlite",
];

/// Whether `extension` (without the leading dot, any case) names a binary
/// format that should never be indexed.
pub fn is_binary_extension(extension: &str) -> bool {
    let lower = extension.to_ascii_lowercase();
    BINARY_EXTENSIONS.contains(&lower.as_str())
}

fn language_for_extension(extension: &str) -> Option<Language> {
    let language: Language = match extension {
        "rs" => tree_sitter_rust::LANGUAGE.into(),
        "py" => tree_sitter_python::LANGUAGE.into(),
        "js" | "jsx" | "mjs" | "cjs" => tree_sitter_javascript::LANGUAGE.into(),
        "ts" => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        "tsx" => tree_sitter_typescript::LANGUAGE_TSX.into(),
        "go" => tree_sitter_go::LANGUAGE.into(),
        _ => return None,
    };
    Some(language)
}

/// Boundary-aware chunking for a recognized language. Returns `None` if the
/// content can't be parsed at all, so the caller can fall back to
/// [`chunk_text`].
fn chunk_code(content: &str, language: Language, config: &ChunkingConfig) -> Option<Vec<Chunk>> {
    let mut parser = Parser::new();
    parser.set_language(&language).ok()?;
    let tree = parser.parse(content, None)?;

    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Some(Vec::new());
    }

    let mut cursor = tree.root_node().walk();
    let node_end_lines: Vec<usize> = tree
        .root_node()
        .children(&mut cursor)
        .map(|node| node.end_position().row)
        .collect();
    if node_end_lines.is_empty() {
        return Some(chunk_text(content, config));
    }

    let char_span = |start: usize, end: usize| -> usize {
        lines[start..=end].iter().map(|line| line.len() + 1).sum()
    };

    let mut chunks = Vec::new();
    let mut start_line = 0usize;
    let mut node_idx = 0usize;

    while node_idx < node_end_lines.len() {
        let mut end_line = node_end_lines[node_idx];
        node_idx += 1;

        while node_idx < node_end_lines.len() {
            let candidate_end = node_end_lines[node_idx];
            let candidate_lines = candidate_end - start_line + 1;
            let candidate_chars = char_span(start_line, candidate_end);
            if candidate_lines > config.chunk_lines || candidate_chars > config.max_chars {
                break;
            }
            end_line = candidate_end;
            node_idx += 1;
        }

        let window_lines = end_line - start_line + 1;
        let window_chars = char_span(start_line, end_line);
        if window_lines > config.chunk_lines || window_chars > config.max_chars {
            // A single top-level node bigger than the budget: sub-split it
            // as plain text rather than leaving one unbounded chunk.
            let sub_content = lines[start_line..=end_line].join("\n");
            chunks.extend(chunk_text(&sub_content, config).into_iter().map(|mut sub| {
                sub.start_line += start_line as u32;
                sub.end_line += start_line as u32;
                sub
            }));
        } else {
            chunks.push(Chunk {
                content: lines[start_line..=end_line].join("\n"),
                start_line: start_line as u32,
                end_line: end_line as u32,
            });
        }

        if node_idx >= node_end_lines.len() {
            break;
        }
        let overlap_start = (end_line + 1).saturating_sub(config.overlap_lines);
        start_line = overlap_start.max(start_line + 1);
    }

    Some(chunks)
}

/// Plain line/char-budget windowing, used for non-code files and as the
/// fallback when code parsing fails or a single AST node exceeds the
/// budget on its own.
pub fn chunk_text(content: &str, config: &ChunkingConfig) -> Vec<Chunk> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut start = 0usize;

    while start < lines.len() {
        let mut end = start;
        let mut char_count = 0usize;
        while end < lines.len() && end - start < config.chunk_lines {
            let candidate_chars = char_count + lines[end].len() + 1;
            if end > start && candidate_chars > config.max_chars {
                break;
            }
            char_count = candidate_chars;
            end += 1;
        }
        end = end.max(start + 1).min(lines.len());

        chunks.push(Chunk {
            content: lines[start..end].join("\n"),
            start_line: start as u32,
            end_line: (end - 1) as u32,
        });

        if end >= lines.len() {
            break;
        }
        let overlap_start = end.saturating_sub(config.overlap_lines);
        start = overlap_start.max(start + 1);
    }

    chunks
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn tiny_config() -> ChunkingConfig {
        ChunkingConfig {
            chunk_lines: 3,
            overlap_lines: 1,
            max_chars: 1000,
        }
    }

    fn numbered_lines(n: usize) -> String {
        (0..n)
            .map(|i| format!("line{i}"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn chunk_text_empty_content_yields_no_chunks() {
        assert_eq!(chunk_text("", &tiny_config()), Vec::new());
    }

    #[test]
    fn chunk_text_shorter_than_budget_is_one_chunk() {
        let chunks = chunk_text("a\nb", &tiny_config());
        assert_eq!(
            chunks,
            vec![Chunk {
                content: "a\nb".to_string(),
                start_line: 0,
                end_line: 1
            }]
        );
    }

    #[test]
    fn chunk_text_splits_with_overlap() {
        // 7 lines, chunk_lines=3, overlap=1: [0,1,2], [2,3,4], [4,5,6]
        let content = numbered_lines(7);
        let chunks = chunk_text(&content, &tiny_config());

        assert_eq!(chunks.len(), 3);
        assert_eq!((chunks[0].start_line, chunks[0].end_line), (0, 2));
        assert_eq!((chunks[1].start_line, chunks[1].end_line), (2, 4));
        assert_eq!((chunks[2].start_line, chunks[2].end_line), (4, 6));
        assert_eq!(chunks[2].content, "line4\nline5\nline6");
    }

    #[test]
    fn chunk_text_respects_max_chars_over_chunk_lines() {
        let config = ChunkingConfig {
            chunk_lines: 100,
            overlap_lines: 0,
            max_chars: 12,
        };
        // Each line is 5 chars + newline = 6; two lines fit (12), a third would not.
        let content = "aaaaa\nbbbbb\nccccc\nddddd";
        let chunks = chunk_text(content, &config);

        assert_eq!(chunks[0].content, "aaaaa\nbbbbb");
    }

    #[test]
    fn chunk_text_makes_forward_progress_even_with_pathological_config() {
        // overlap_lines >= chunk_lines must not infinite-loop.
        let config = ChunkingConfig {
            chunk_lines: 2,
            overlap_lines: 10,
            max_chars: 1000,
        };
        let content = numbered_lines(10);
        let chunks = chunk_text(&content, &config);

        assert!(!chunks.is_empty());
        assert_eq!(chunks.last().expect("at least one chunk").end_line, 9);
    }

    #[test]
    fn chunk_code_splits_along_function_boundaries() {
        let source = "fn a() {\n    1;\n}\n\nfn b() {\n    2;\n}\n\nfn c() {\n    3;\n}\n";
        let config = ChunkingConfig {
            chunk_lines: 4,
            overlap_lines: 0,
            max_chars: 1000,
        };
        let chunks =
            chunk_code(source, tree_sitter_rust::LANGUAGE.into(), &config).expect("parses");

        assert!(
            chunks.len() >= 2,
            "expected multiple chunks, got {chunks:?}"
        );
        for chunk in &chunks {
            assert!(
                chunk.content.contains("fn "),
                "chunk should align to function boundaries: {chunk:?}"
            );
        }
    }

    #[test]
    fn chunk_code_falls_back_to_text_for_empty_file() {
        let chunks = chunk_code(
            "",
            tree_sitter_rust::LANGUAGE.into(),
            &ChunkingConfig::default(),
        )
        .expect("parses");
        assert_eq!(chunks, Vec::new());
    }

    #[test]
    fn chunk_code_sub_splits_an_oversized_single_node() {
        // One function with many statements, budget far smaller than the whole thing.
        let body: String = (0..50).map(|i| format!("    let x{i} = {i};\n")).collect();
        let source = format!("fn big() {{\n{body}}}\n");
        let config = ChunkingConfig {
            chunk_lines: 10,
            overlap_lines: 2,
            max_chars: 100_000,
        };

        let chunks =
            chunk_code(&source, tree_sitter_rust::LANGUAGE.into(), &config).expect("parses");

        assert!(
            chunks.len() > 1,
            "a 52-line function with a 10-line budget must be sub-split"
        );
        for chunk in &chunks {
            let line_count = chunk.content.lines().count();
            assert!(
                line_count <= config.chunk_lines,
                "chunk has {line_count} lines, over budget: {chunk:?}"
            );
        }
    }

    #[test]
    fn chunk_file_dispatches_by_extension() {
        let rust_chunks = chunk_file("rs", "fn a() {}\n", &ChunkingConfig::default());
        assert_eq!(rust_chunks.len(), 1);

        let text_chunks = chunk_file("md", "# hello\n", &ChunkingConfig::default());
        assert_eq!(text_chunks.len(), 1);
    }

    #[test]
    fn is_binary_extension_matches_known_binary_types_case_insensitively() {
        assert!(is_binary_extension("png"));
        assert!(is_binary_extension("PNG"));
        assert!(is_binary_extension("exe"));
        assert!(!is_binary_extension("rs"));
        assert!(!is_binary_extension("md"));
    }

    proptest! {
        #[test]
        fn chunk_text_is_deterministic_and_covers_every_line(
            line_count in 0usize..40,
            chunk_lines in 1usize..10,
            overlap_lines in 0usize..10,
        ) {
            let content = numbered_lines(line_count);
            let config = ChunkingConfig { chunk_lines, overlap_lines, max_chars: 10_000 };

            let first = chunk_text(&content, &config);
            let second = chunk_text(&content, &config);
            prop_assert_eq!(&first, &second, "chunking must be deterministic");

            if line_count == 0 {
                prop_assert!(first.is_empty());
            } else {
                prop_assert_eq!(first[0].start_line, 0);
                prop_assert_eq!(first.last().expect("non-empty").end_line, line_count as u32 - 1);
                for pair in first.windows(2) {
                    prop_assert!(pair[1].start_line <= pair[0].end_line + 1, "gap between chunks: {:?}", pair);
                    prop_assert!(pair[1].start_line > pair[0].start_line, "no forward progress: {:?}", pair);
                }
            }
        }
    }
}
