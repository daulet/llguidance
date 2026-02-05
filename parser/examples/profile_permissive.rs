use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Instant};

use anyhow::{bail, ensure, Context, Result};
use llguidance::{
    api::TopLevelGrammar,
    earley::{perf::ParserPerfCounters, ParserStats},
    toktrie::{SimpleVob, TokEnv, TokTrie, TokenId},
    Matcher, ParserFactory,
};

const BENCH_TOKENIZER_DEFAULT_REL: &str = "benches/data/llama3_tokenizer.json";
const BENCH_TOKENIZER_ENV: &str = "LLGUIDANCE_BENCH_TOKENIZER";
#[cfg(feature = "mask_cache")]
const MASK_CACHE_LABEL: &str = "cache_on";
#[cfg(not(feature = "mask_cache"))]
const MASK_CACHE_LABEL: &str = "cache_off";

#[derive(Clone, Copy)]
struct ProfileCase {
    name: &'static str,
    grammar: GrammarSpec,
    prefix: &'static [u8],
    forbidden_byte: Option<u8>,
}

#[derive(Clone, Copy)]
enum GrammarSpec {
    Regex(&'static str),
    Lark(&'static str),
}

const HORIZON_LARK: &str = r#"start: text "x"
text[max_tokens=8192, stop="x"]: /[^x]*/"#;

const CASES: [ProfileCase; 3] = [
    ProfileCase {
        name: "dot_star",
        grammar: GrammarSpec::Regex(".*"),
        prefix: b"",
        forbidden_byte: None,
    },
    ProfileCase {
        name: "not_x_then_x",
        grammar: GrammarSpec::Regex("[^x]*x"),
        prefix: b"aaaaaaaaaaaaaaaa",
        forbidden_byte: Some(b'x'),
    },
    ProfileCase {
        name: "horizon_not_x_then_x",
        grammar: GrammarSpec::Lark(HORIZON_LARK),
        prefix: b"aaaaaaaaaaaaaaaa",
        forbidden_byte: Some(b'x'),
    },
];

#[derive(Clone, Copy, Default)]
struct TimerTotals {
    time_us: usize,
    calls: usize,
}

#[derive(Default)]
struct StepTotals {
    compute_time_us: u128,
    rows: u128,
    cached_rows: u128,
    all_items: u128,
    lexer_cost: u128,
    slices_applied: u128,
    trie_nodes_walked: u128,
}

impl StepTotals {
    fn add(&mut self, stats: &ParserStats) {
        self.compute_time_us += stats.compute_time_us as u128;
        self.rows += stats.rows as u128;
        self.cached_rows += stats.cached_rows as u128;
        self.all_items += stats.all_items as u128;
        self.lexer_cost += stats.lexer_cost as u128;
        self.slices_applied += stats.slices_applied as u128;
        self.trie_nodes_walked += stats.trie_nodes_walked as u128;
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
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

fn load_tok_env(path: &PathBuf) -> Result<TokEnv> {
    if !path.exists() {
        bail!(
            "tokenizer not found at {}; set {} to override",
            path.display(),
            BENCH_TOKENIZER_ENV
        );
    }

    let btok = toktrie_hf_tokenizers::ByteTokenizer::from_file(path)
        .with_context(|| format!("loading tokenizer from {}", path.display()))?;
    btok.into_tok_env(None).context("building TokEnv")
}

fn consume_prefix_bytes(matcher: &mut Matcher, tok_env: &TokEnv, prefix: &[u8]) -> Result<()> {
    let tokens = tok_env.tokenize_bytes(prefix);
    for tok in tokens {
        let mask = matcher.compute_mask()?;
        ensure!(mask.is_allowed(tok), "prefix token {tok} is not allowed");
        matcher.consume_token(tok)?;
    }
    Ok(())
}

fn regex_matcher_at_prefix(
    tok_env: &TokEnv,
    regex: &str,
    prefix: &[u8],
) -> Result<(Matcher, Arc<ParserPerfCounters>)> {
    let mut factory = ParserFactory::new_simple(tok_env)?;
    factory.quiet();
    let perf_counters = factory.perf_counters();
    let grammar = TopLevelGrammar::from_regex(regex);
    let mut matcher = Matcher::new(factory.create_parser(grammar));
    consume_prefix_bytes(&mut matcher, tok_env, prefix)?;
    Ok((matcher, perf_counters))
}

fn grammar_matcher_at_prefix(
    tok_env: &TokEnv,
    grammar: TopLevelGrammar,
    prefix: &[u8],
) -> Result<(Matcher, Arc<ParserPerfCounters>)> {
    let mut factory = ParserFactory::new_simple(tok_env)?;
    factory.quiet();
    let perf_counters = factory.perf_counters();
    let mut matcher = Matcher::new(factory.create_parser(grammar));
    consume_prefix_bytes(&mut matcher, tok_env, prefix)?;
    Ok((matcher, perf_counters))
}

fn pick_loop_token(tok_env: &TokEnv, mask: &SimpleVob, forbidden_byte: Option<u8>) -> TokenId {
    let trie = tok_env.tok_trie();
    let eos = trie.eos_token();

    for tok in 0..trie.vocab_size() as TokenId {
        if tok == eos || !mask.is_allowed(tok) {
            continue;
        }
        let bytes = trie.token(tok);
        if bytes.is_empty() || bytes[0] == TokTrie::SPECIAL_TOKEN_MARKER {
            continue;
        }
        if forbidden_byte.is_some_and(|b| bytes.contains(&b)) {
            continue;
        }
        return tok;
    }

    panic!("no reusable token found in mask");
}

fn snapshot_timers(perf: &ParserPerfCounters) -> HashMap<String, TimerTotals> {
    let mut out = HashMap::new();
    for counter in perf.counters() {
        let (_, time_us, calls) = counter.get();
        out.insert(counter.name().to_string(), TimerTotals { time_us, calls });
    }
    out
}

fn timer_delta(
    before: &HashMap<String, TimerTotals>,
    after: &HashMap<String, TimerTotals>,
    name: &str,
) -> TimerTotals {
    let b = before.get(name).copied().unwrap_or_default();
    let a = after.get(name).copied().unwrap_or_default();
    TimerTotals {
        time_us: a.time_us.saturating_sub(b.time_us),
        calls: a.calls.saturating_sub(b.calls),
    }
}

fn pct(part: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        (part as f64) * 100.0 / (total as f64)
    }
}

fn avg(sum: u128, n: usize) -> f64 {
    if n == 0 {
        0.0
    } else {
        (sum as f64) / (n as f64)
    }
}

fn run_case(
    tok_env: &TokEnv,
    case: ProfileCase,
    warmup_steps: usize,
    measure_steps: usize,
) -> Result<()> {
    let vocab_size = tok_env.tok_trie().vocab_size();
    let (mut matcher, perf_counters) = match case.grammar {
        GrammarSpec::Regex(regex) => regex_matcher_at_prefix(tok_env, regex, case.prefix)?,
        GrammarSpec::Lark(lark) => grammar_matcher_at_prefix(
            tok_env,
            TopLevelGrammar::from_lark(lark.to_string()),
            case.prefix,
        )?,
    };
    let first_mask = matcher.compute_mask()?;
    let target_token = pick_loop_token(tok_env, &first_mask, case.forbidden_byte);

    for _ in 0..warmup_steps {
        let mask = matcher.compute_mask()?;
        ensure!(
            mask.is_allowed(target_token),
            "target token became disallowed"
        );
        matcher.consume_token(target_token)?;
    }

    let before = snapshot_timers(perf_counters.as_ref());
    let mut wall_total_us = 0u128;
    let mut wall_max_us = 0u128;
    let mut step_totals = StepTotals::default();
    let mut total_set_bits = 0u128;

    for _ in 0..measure_steps {
        let t0 = Instant::now();
        let mask = matcher.compute_mask()?;
        let us = t0.elapsed().as_micros();
        wall_total_us += us;
        wall_max_us = wall_max_us.max(us);

        let step_stats = matcher.last_step_stats()?;
        step_totals.add(step_stats);

        total_set_bits += mask.num_set() as u128;
        ensure!(
            mask.is_allowed(target_token),
            "target token became disallowed"
        );
        matcher.consume_token(target_token)?;
    }

    let after = snapshot_timers(perf_counters.as_ref());

    let t_compute_mask = timer_delta(&before, &after, "compute_mask");
    let t_compute_bias = timer_delta(&before, &after, "compute_bias");
    let t_force_bytes = timer_delta(&before, &after, "force_bytes");
    let t_force_bytes_empty = timer_delta(&before, &after, "force_bytes_empty");
    let t_tokenize_ff = timer_delta(&before, &after, "tokenize_ff");
    let t_mask_cache_key = timer_delta(&before, &after, "mask_cache_key");

    let total_mask_us = t_compute_mask.time_us;
    let accounted_us = t_compute_bias.time_us
        + t_force_bytes.time_us
        + t_tokenize_ff.time_us
        + t_mask_cache_key.time_us;
    let other_us = total_mask_us.saturating_sub(accounted_us);
    let target_bytes = tok_env.tok_trie().token(target_token);
    let target_dbg = String::from_utf8_lossy(target_bytes);
    let grammar_label = match case.grammar {
        GrammarSpec::Regex(rx) => format!("regex={rx:?}"),
        GrammarSpec::Lark(_) => "grammar=lark(max_tokens horizon)".to_string(),
    };
    let estimated_cache_hit_ratio = if t_compute_mask.calls == 0 {
        0.0
    } else {
        ((t_compute_mask.calls.saturating_sub(t_compute_bias.calls)) as f64)
            / (t_compute_mask.calls as f64)
    };
    let cache_eligibility_ratio = if t_compute_mask.calls == 0 {
        0.0
    } else {
        (t_mask_cache_key.calls as f64) / (t_compute_mask.calls as f64)
    };

    println!();
    println!(
        "case={} {} prefix_len={} vocab={}",
        case.name,
        grammar_label,
        case.prefix.len(),
        vocab_size
    );
    println!(
        "loop_token={} token_bytes={:?}",
        target_token,
        target_dbg.as_ref()
    );
    println!("steps={} warmup={}", measure_steps, warmup_steps);
    println!(
        "wall_compute_mask: avg={:.2}us max={}us",
        avg(wall_total_us, measure_steps),
        wall_max_us
    );
    println!(
        "timer_compute_mask: avg={:.2}us calls={}",
        avg(t_compute_mask.time_us as u128, measure_steps),
        t_compute_mask.calls
    );
    println!(
        "timer_compute_bias: avg={:.2}us calls={} ({:.1}% of compute_mask time)",
        avg(t_compute_bias.time_us as u128, measure_steps),
        t_compute_bias.calls,
        pct(t_compute_bias.time_us, total_mask_us)
    );
    println!(
        "timer_force_bytes: avg={:.2}us calls={} ({:.1}% of compute_mask time)",
        avg(t_force_bytes.time_us as u128, measure_steps),
        t_force_bytes.calls,
        pct(t_force_bytes.time_us, total_mask_us)
    );
    println!(
        "timer_force_bytes_empty: avg={:.2}us calls={}",
        avg(t_force_bytes_empty.time_us as u128, measure_steps),
        t_force_bytes_empty.calls
    );
    println!(
        "timer_tokenize_ff: avg={:.2}us calls={} ({:.1}% of compute_mask time)",
        avg(t_tokenize_ff.time_us as u128, measure_steps),
        t_tokenize_ff.calls,
        pct(t_tokenize_ff.time_us, total_mask_us)
    );
    println!(
        "timer_mask_cache_key: avg={:.2}us calls={} ({:.1}% of compute_mask time)",
        avg(t_mask_cache_key.time_us as u128, measure_steps),
        t_mask_cache_key.calls,
        pct(t_mask_cache_key.time_us, total_mask_us)
    );
    println!(
        "timer_other: avg={:.2}us ({:.1}% of compute_mask time)",
        avg(other_us as u128, measure_steps),
        pct(other_us, total_mask_us)
    );
    println!(
        "recompute_ratio: compute_bias_calls / compute_mask_calls = {:.3}",
        if t_compute_mask.calls == 0 {
            0.0
        } else {
            (t_compute_bias.calls as f64) / (t_compute_mask.calls as f64)
        }
    );
    println!(
        "cache_counters: estimated_hit_ratio={:.3} key_calls_per_step={:.3}",
        estimated_cache_hit_ratio, cache_eligibility_ratio
    );
    println!(
        "parser_step_avg: compute_time_us={:.2} lexer_cost={:.2} all_items={:.2} trie_nodes_walked={:.2} rows={:.2} cached_rows={:.2}",
        avg(step_totals.compute_time_us, measure_steps),
        avg(step_totals.lexer_cost, measure_steps),
        avg(step_totals.all_items, measure_steps),
        avg(step_totals.trie_nodes_walked, measure_steps),
        avg(step_totals.rows, measure_steps),
        avg(step_totals.cached_rows, measure_steps),
    );
    println!(
        "mask_density: avg_num_set={:.2} ({:.2}% of vocab)",
        avg(total_set_bits, measure_steps),
        pct(
            total_set_bits as usize,
            (measure_steps * vocab_size) as usize
        )
    );

    Ok(())
}

fn main() -> Result<()> {
    let warmup_steps = env_usize("LLGUIDANCE_PROFILE_WARMUP", 512);
    let measure_steps = env_usize("LLGUIDANCE_PROFILE_STEPS", 4096);
    let tokenizer_path = bench_tokenizer_path();
    let tok_env = load_tok_env(&tokenizer_path)?;

    println!(
        "profile_permissive build={} tokenizer={} vocab={}",
        MASK_CACHE_LABEL,
        tokenizer_path.display(),
        tok_env.tok_trie().vocab_size()
    );
    println!(
        "env: {}={} {}={}",
        "LLGUIDANCE_PROFILE_WARMUP", warmup_steps, "LLGUIDANCE_PROFILE_STEPS", measure_steps
    );

    for case in CASES {
        run_case(&tok_env, case, warmup_steps, measure_steps)?;
    }

    Ok(())
}
