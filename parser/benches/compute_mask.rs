use std::hint::black_box;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use llguidance::{
    api::TopLevelGrammar,
    toktrie::{SimpleVob, TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv},
    Matcher, ParserFactory,
};

const BLOG_SCHEMA_JSON: &str = include_str!("../../sample_parser/data/blog.schema.json");
const BENCH_TOKENIZER_DEFAULT_REL: &str = "benches/data/llama3_tokenizer.json";
const BENCH_TOKENIZER_ENV: &str = "LLGUIDANCE_BENCH_TOKENIZER";
#[cfg(feature = "mask_cache")]
const MASK_CACHE_LABEL: &str = "cache_on";
#[cfg(not(feature = "mask_cache"))]
const MASK_CACHE_LABEL: &str = "cache_off";

// Different prefixes representing various parser states
const PREFIX_START: &[u8] = b""; // Start of JSON
const PREFIX_AFTER_KEY: &[u8] = b"{\"title\":"; // After key, expecting value
const PREFIX_IN_STRING: &[u8] = b"{\"title\":\""; // Inside a string value
const PREFIX_MID_STRING: &[u8] = b"{\"title\":\"Hello World"; // Mid-string with content

struct SyntheticTokEnv {
    trie: TokTrie,
}

impl TokenizerEnv for SyntheticTokEnv {
    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        self.trie.greedy_tokenize(s)
    }

    fn tokenize_is_canonical(&self) -> bool {
        false
    }
}

fn synthetic_tok_env(vocab_size: usize) -> TokEnv {
    let eos_token = (vocab_size - 1) as TokenId;
    let mut tokens = Vec::with_capacity(vocab_size);

    for byte in 0u8..=255 {
        tokens.push(vec![byte]);
    }

    let prefixes: &[u8] = b" \"{[\\etaoin";
    for i in 0..(vocab_size - tokens.len() - 1) {
        let mut tok = Vec::with_capacity(5);
        tok.push(prefixes[i % prefixes.len()]);
        tok.extend_from_slice(&(i as u32).to_le_bytes());
        tokens.push(tok);
    }

    tokens.push(b"\xFF<|eos|>".to_vec());
    let trie = TokTrie::from(&TokRxInfo::new(vocab_size as u32, eos_token), &tokens);
    Arc::new(SyntheticTokEnv { trie })
}

fn blog_grammar() -> TopLevelGrammar {
    let schema: serde_json::Value = serde_json::from_str(BLOG_SCHEMA_JSON).unwrap();
    TopLevelGrammar::from_json_schema(schema)
}

fn regex_grammar(regex: &str) -> TopLevelGrammar {
    TopLevelGrammar::from_regex(regex)
}

fn consume_prefix_bytes(matcher: &mut Matcher, tok_env: &TokEnv, prefix: &[u8]) {
    let tokens = tok_env.tokenize_bytes(prefix);
    for tok in tokens {
        let mask = matcher.compute_mask().unwrap();
        assert!(mask.is_allowed(tok), "prefix token {tok} is not allowed");
        matcher.consume_token(tok).unwrap();
    }
}

fn matcher_at_prefix(tok_env: &TokEnv, prefix: &[u8]) -> Matcher {
    let mut factory = ParserFactory::new_simple(tok_env).unwrap();
    factory.quiet();
    let mut matcher = Matcher::new(factory.create_parser(blog_grammar()));
    consume_prefix_bytes(&mut matcher, tok_env, prefix);
    matcher
}

fn regex_matcher_at_prefix(tok_env: &TokEnv, regex: &str, prefix: &[u8]) -> Matcher {
    let mut factory = ParserFactory::new_simple(tok_env).unwrap();
    factory.quiet();
    let mut matcher = Matcher::new(factory.create_parser(regex_grammar(regex)));
    consume_prefix_bytes(&mut matcher, tok_env, prefix);
    matcher
}

fn bench_tokenizer_path() -> PathBuf {
    if let Ok(path) = std::env::var(BENCH_TOKENIZER_ENV) {
        return PathBuf::from(path);
    }

    let manifest_default =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(BENCH_TOKENIZER_DEFAULT_REL);
    if manifest_default.exists() {
        return manifest_default;
    }

    PathBuf::from(BENCH_TOKENIZER_DEFAULT_REL)
}

fn real_tok_env() -> Option<TokEnv> {
    static TOK_ENV: OnceLock<Option<TokEnv>> = OnceLock::new();
    TOK_ENV
        .get_or_init(|| {
            let path = bench_tokenizer_path();
            if !path.exists() {
                return None;
            }
            let btok = toktrie_hf_tokenizers::ByteTokenizer::from_file(path).ok()?;
            btok.into_tok_env(None).ok()
        })
        .clone()
}

fn pick_loop_token(tok_env: &TokEnv, mask: &SimpleVob, forbidden_byte: Option<u8>) -> TokenId {
    let trie = tok_env.tok_trie();
    let eos = trie.eos_token();
    let forbid = forbidden_byte;

    for tok in 0..trie.vocab_size() as TokenId {
        if tok == eos || !mask.is_allowed(tok) {
            continue;
        }
        let bytes = trie.token(tok);
        if bytes.is_empty() || bytes[0] == TokTrie::SPECIAL_TOKEN_MARKER {
            continue;
        }
        if forbid.is_some_and(|b| bytes.contains(&b)) {
            continue;
        }
        return tok;
    }

    panic!("no reusable token found in mask")
}

/// Benchmark compute_mask at different vocabulary sizes.
/// Uses vocab size as throughput metric since larger vocabs require more work.
fn bench_compute_mask(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_mask");

    // Realistic LLM vocabulary sizes (8k to 128k)
    for vocab_size in [8_192, 32_768, 65_536, 128_000] {
        group.throughput(Throughput::Elements(vocab_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(vocab_size),
            &vocab_size,
            |b, &size| {
                let tok_env = synthetic_tok_env(size);
                let mut matcher = matcher_at_prefix(&tok_env, PREFIX_IN_STRING);
                b.iter(|| black_box(matcher.compute_mask().unwrap()))
            },
        );
    }
    group.finish();
}

/// Benchmark compute_mask at different parser positions within the grammar.
/// This reveals if certain grammar states are slower than others.
fn bench_compute_mask_positions(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_mask_positions");
    let vocab_size = 32_768usize;

    let positions = [
        ("start", PREFIX_START),
        ("after_key", PREFIX_AFTER_KEY),
        ("in_string", PREFIX_IN_STRING),
        ("mid_string", PREFIX_MID_STRING),
    ];

    group.throughput(Throughput::Elements(vocab_size as u64));

    for (name, prefix) in positions {
        group.bench_with_input(BenchmarkId::from_parameter(name), &prefix, |b, &prefix| {
            let tok_env = synthetic_tok_env(vocab_size);
            let mut matcher = matcher_at_prefix(&tok_env, prefix);
            b.iter(|| black_box(matcher.compute_mask().unwrap()))
        });
    }
    group.finish();
}

/// Benchmark realistic token generation loop (no rollback).
/// Measures per-token latency during continuous generation.
fn bench_token_generation(c: &mut Criterion) {
    use criterion::BatchSize;

    let mut group = c.benchmark_group("token_generation");
    let num_tokens = 20;

    for vocab_size in [32_768, 65_536, 128_000] {
        group.throughput(Throughput::Elements(num_tokens));
        group.bench_with_input(
            BenchmarkId::from_parameter(vocab_size),
            &vocab_size,
            |b, &size| {
                let tok_env = synthetic_tok_env(size);

                b.iter_batched(
                    || matcher_at_prefix(&tok_env, PREFIX_IN_STRING),
                    |mut m| {
                        for _ in 0..num_tokens {
                            let mask = m.compute_mask().unwrap();
                            // Pick first allowed lowercase letter
                            let tok = (b'a' as TokenId..=b'z' as TokenId)
                                .find(|&t| mask.is_allowed(t))
                                .unwrap_or(b'a' as TokenId);
                            m.consume_token(tok).unwrap();
                        }
                        black_box(m)
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

/// Benchmark cold-start: time to create parser and compute first mask.
/// Important for latency-sensitive applications.
fn bench_first_mask(c: &mut Criterion) {
    let mut group = c.benchmark_group("first_mask");

    for vocab_size in [32_768, 65_536, 128_000] {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(vocab_size),
            &vocab_size,
            |b, &size| {
                let tok_env = synthetic_tok_env(size);
                let grammar = blog_grammar();

                b.iter(|| {
                    let mut factory = ParserFactory::new_simple(&tok_env).unwrap();
                    factory.quiet();
                    let mut matcher = Matcher::new(factory.create_parser(grammar.clone()));
                    black_box(matcher.compute_mask().unwrap())
                })
            },
        );
    }
    group.finish();
}

/// Benchmark generation in permissive regex states where masks are dense.
/// Build with --features mask_cache to compare cache-on vs cache-off.
fn bench_permissive_regex_generation(c: &mut Criterion) {
    use criterion::BatchSize;

    let mut group = c.benchmark_group("permissive_regex_generation");
    let num_tokens = 64usize;
    let regexes: [(&str, &str, &[u8], Option<u8>); 2] = [
        ("dot_star", ".*", b"", None),
        ("not_x_then_x", "[^x]*x", b"aaaaaaaaaaaaaaaa", Some(b'x')),
    ];

    let tok_envs: Vec<(String, TokEnv)> = if let Some(real) = real_tok_env() {
        vec![("llama3".to_string(), real)]
    } else {
        eprintln!(
            "warning: {} missing; using synthetic vocab for permissive benchmark",
            bench_tokenizer_path().display()
        );
        vec![
            ("synthetic_32768".to_string(), synthetic_tok_env(32_768)),
            ("synthetic_65536".to_string(), synthetic_tok_env(65_536)),
        ]
    };

    group.throughput(Throughput::Elements(num_tokens as u64));
    for (tok_env_name, tok_env) in tok_envs {
        let vocab_size = tok_env.tok_trie().vocab_size();
        for (name, regex, prefix, forbidden_byte) in regexes {
            let bench_name = format!("{name}_{MASK_CACHE_LABEL}_{tok_env_name}");
            group.bench_with_input(
                BenchmarkId::new(bench_name, vocab_size),
                &vocab_size,
                |b, &_size| {
                    let tok_env = tok_env.clone();

                    b.iter_batched(
                        || {
                            let mut m = regex_matcher_at_prefix(&tok_env, regex, prefix);
                            let first_mask = m.compute_mask().unwrap();
                            let target_token =
                                pick_loop_token(&tok_env, &first_mask, forbidden_byte);
                            (m, target_token)
                        },
                        |(mut m, target_token)| {
                            for _ in 0..num_tokens {
                                let mask = m.compute_mask().unwrap();
                                assert!(mask.is_allowed(target_token));
                                m.consume_token(target_token).unwrap();
                            }
                            black_box(m)
                        },
                        BatchSize::SmallInput,
                    )
                },
            );
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(100)
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(5))
        .noise_threshold(0.05);
    targets = bench_compute_mask, bench_compute_mask_positions, bench_token_generation, bench_first_mask, bench_permissive_regex_generation
}
criterion_main!(benches);
