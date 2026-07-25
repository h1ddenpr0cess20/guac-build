//! Pure shared token-estimation primitives.
//!
//! This crate is the single source of truth for the bytes/4 heuristic and the
//! derived-display arithmetic that `/context`, `/session-info`, the auto-compact
//! gates, the preflight overflow check, and every client renderer use to talk
//! about context-window usage.

/// Bytes per token under the rough character-based heuristic.
pub const BYTES_PER_TOKEN: u64 = 4;

/// Per-image approximate token cost when summing
/// low-resolution image patches.
pub const IMAGE_TOKEN_ESTIMATE: u64 = 765;

/// Bytes/4 estimate of a string's token count.
#[inline]
pub fn estimate_tokens(s: &str) -> u64 {
    (s.len() as u64) / BYTES_PER_TOKEN
}

/// Inverse of [`estimate_tokens`]: convert a token budget into a character
/// budget. Used by skill discovery to size text passages against the model's
/// context window.
#[inline]
pub fn estimate_chars(tokens: u64) -> u64 {
    tokens.saturating_mul(BYTES_PER_TOKEN)
}

/// Token estimate for `image_count` images at [`IMAGE_TOKEN_ESTIMATE`] each.
#[inline]
pub fn estimate_image_tokens(image_count: u64) -> u64 {
    image_count.saturating_mul(IMAGE_TOKEN_ESTIMATE)
}

/// Usage percentage as `f64`, clamped to `100.0`. Returns `0.0` when
/// `total == 0`.
#[inline]
pub fn usage_percentage(used: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        ((used as f64) / (total as f64) * 100.0).min(100.0)
    }
}

/// Usage percentage rounded to `u8`, clamped to `100`.
#[inline]
pub fn usage_percentage_u8(used: u64, total: u64) -> u8 {
    usage_percentage(used, total).round() as u8
}

/// Integer-arithmetic (truncating) usage percentage, clamped to `100`.
///
/// Differs from [`usage_percentage_u8`] in two ways: no `f64` round-trip,
/// and the result is **truncated** (not rounded).
///
/// Returns `u8` because the result is bounded to `100`. Saturates on
/// overflow via `saturating_mul`.
#[inline]
pub fn usage_percentage_truncated_u8(used: u64, total: u64) -> u8 {
    if total == 0 {
        0
    } else {
        ((used.saturating_mul(100) / total).min(100)) as u8
    }
}

/// `total - used`, saturating at zero. The "free" portion of the context
/// window for `/context` rendering.
#[inline]
pub fn free_tokens(total: u64, used: u64) -> u64 {
    total.saturating_sub(used)
}

/// True when `used >= context_window * threshold_percent / 100`. Returns
/// `false` for `context_window == 0` so callers do not have to special-case
/// missing windows. Computed in integer arithmetic to match the existing
/// auto-compact gate semantics.
#[inline]
pub fn exceeds_threshold(used: u64, context_window: u64, threshold_percent: u8) -> bool {
    if context_window == 0 {
        return false;
    }
    used.saturating_mul(100) >= context_window.saturating_mul(threshold_percent as u64)
}

/// True when `used * 100 >= context_window * threshold_percent - headroom * 100`,
/// the scaled form of [`exceeds_threshold`] minus a token headroom.
/// Returns `false` for `context_window == 0`.
#[inline]
pub fn exceeds_threshold_with_headroom(
    used: u64,
    context_window: u64,
    threshold_percent: u8,
    headroom: u64,
) -> bool {
    if context_window == 0 {
        return false;
    }
    used.saturating_mul(100)
        >= context_window
            .saturating_mul(threshold_percent as u64)
            .saturating_sub(headroom.saturating_mul(100))
}

/// At or below this window size the client treats the model as "small
/// context" and tightens its own budgets. Sized so that every window a hosted
/// frontier model reports stays on the untouched path, while locally served
/// models (LM Studio, Ollama) — which are commonly configured at 4K–32K —
/// land on the scaled path.
pub const SMALL_CONTEXT_WINDOW_TOKENS: u64 = 65_536;

/// Upper bound on the free headroom [`compaction_reserve_tokens`] asks for.
/// Past this, a bigger reserve buys nothing: the threshold percentage is
/// already the binding constraint.
pub const MAX_COMPACTION_RESERVE_TOKENS: u64 = 8_192;

/// Tokens to keep free at the moment auto-compaction fires, so the reply that
/// triggers it — plus the tool result it is reacting to — still fit.
///
/// A quarter of the window, capped at [`MAX_COMPACTION_RESERVE_TOKENS`]. The
/// cap is what keeps large windows on the unscaled path: 8K of 1M is 0.8%, so
/// [`auto_compact_threshold_for_window`] leaves their configured percentage
/// alone.
#[inline]
pub fn compaction_reserve_tokens(context_window: u64) -> u64 {
    (context_window / 4).min(MAX_COMPACTION_RESERVE_TOKENS)
}

/// True when `context_window` is at or below [`SMALL_CONTEXT_WINDOW_TOKENS`].
///
/// `0` (an unknown window) is not small — callers treat it as "no information"
/// and leave their defaults alone rather than clamping to the tightest budget.
#[inline]
pub fn is_small_context_window(context_window: u64) -> bool {
    context_window > 0 && context_window <= SMALL_CONTEXT_WINDOW_TOKENS
}

/// Auto-compact threshold for `context_window`, never looser than `configured`.
///
/// A percentage tuned for a 1M window leaves too few absolute tokens on a small
/// one: 85% of 8K is 1.2K free, which one tool result overruns before
/// compaction can run. This lowers the percentage until at least
/// [`compaction_reserve_tokens`] stay free, and is a no-op above roughly 55K
/// where the configured percentage is already the tighter of the two.
///
/// Returns `configured` unchanged for `context_window == 0`.
#[inline]
pub fn auto_compact_threshold_for_window(context_window: u64, configured: u8) -> u8 {
    if context_window == 0 {
        return configured;
    }
    let reserve = compaction_reserve_tokens(context_window);
    let by_reserve = context_window.saturating_sub(reserve) * 100 / context_window;
    configured.min(by_reserve as u8)
}

/// Floor for [`tool_output_budget_bytes`]. Below this a tool result is too
/// clipped to act on, so a tiny window gets a disproportionate slice rather
/// than an unusable one.
pub const MIN_TOOL_OUTPUT_BUDGET_BYTES: usize = 2_000;

/// Inline tool-result byte cap for `context_window`, never above `default_bytes`.
///
/// One `bash` or MCP result is allowed an eighth of the window; on the 20K-byte
/// default that ceiling only binds below ~40K tokens. Returns `default_bytes`
/// unchanged for windows that are not [`is_small_context_window`], so hosted
/// models keep the tuned default.
#[inline]
pub fn tool_output_budget_bytes(context_window: u64, default_bytes: usize) -> usize {
    if !is_small_context_window(context_window) {
        return default_bytes;
    }
    let by_window = usize::try_from(estimate_chars(context_window / 8)).unwrap_or(default_bytes);
    // `min(default_bytes)` last so the floor can never loosen a cap the host
    // set deliberately below it.
    by_window
        .max(MIN_TOOL_OUTPUT_BUDGET_BYTES)
        .min(default_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_tokens_is_bytes_over_four() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abc"), 0);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens(&"x".repeat(4000)), 1000);
    }

    #[test]
    fn estimate_chars_is_inverse() {
        assert_eq!(estimate_chars(0), 0);
        assert_eq!(estimate_chars(1), 4);
        assert_eq!(estimate_chars(1000), 4000);
    }

    #[test]
    fn estimate_image_tokens_uses_constant() {
        assert_eq!(estimate_image_tokens(0), 0);
        assert_eq!(estimate_image_tokens(1), IMAGE_TOKEN_ESTIMATE);
        assert_eq!(estimate_image_tokens(3), 3 * IMAGE_TOKEN_ESTIMATE);
    }

    #[test]
    fn usage_percentage_clamps_and_handles_zero_total() {
        assert_eq!(usage_percentage(0, 0), 0.0);
        assert_eq!(usage_percentage(50, 100), 50.0);
        assert_eq!(usage_percentage(150, 100), 100.0);
        assert_eq!(usage_percentage(100, 0), 0.0);
    }

    #[test]
    fn usage_percentage_u8_rounds() {
        assert_eq!(usage_percentage_u8(0, 100), 0);
        assert_eq!(usage_percentage_u8(50, 100), 50);
        assert_eq!(usage_percentage_u8(99, 100), 99);
        // 12_700 / 256_000 = 0.04960... -> 5 after rounding
        assert_eq!(usage_percentage_u8(12_700, 256_000), 5);
        assert_eq!(usage_percentage_u8(150, 100), 100);
    }

    /// Half-boundary contract — locks rounding direction. `85 / 200 = 0.425`
    /// becomes `42.5%` which rounds half-up to `43`. The truncating helper
    /// returns `42` for the same input (see `usage_percentage_truncated_u8`).
    #[test]
    fn usage_percentage_u8_rounds_half_up() {
        assert_eq!(usage_percentage_u8(85, 200), 43);
        // 7 / 8 = 0.875, rounds to 88 (truncated would be 87).
        assert_eq!(usage_percentage_u8(7, 8), 88);
    }

    #[test]
    fn usage_percentage_truncated_u8_clamps_and_handles_zero_total() {
        assert_eq!(usage_percentage_truncated_u8(0, 0), 0);
        assert_eq!(usage_percentage_truncated_u8(50, 100), 50);
        assert_eq!(usage_percentage_truncated_u8(150, 100), 100);
        // Large values do not overflow because we use saturating_mul.
        assert_eq!(usage_percentage_truncated_u8(u64::MAX, 1), 100);
    }

    /// Truncation contract — distinguishes this helper from
    /// `usage_percentage_u8`, which rounds. Locks in that
    /// `exceeds_threshold(used, cw, p)` and
    /// `usage_percentage_truncated_u8(used, cw) >= p` agree.
    #[test]
    fn usage_percentage_truncated_u8_truncates_does_not_round() {
        // 85 / 200 = 0.425, truncated -> 42 (rounded would be 43).
        assert_eq!(usage_percentage_truncated_u8(85, 200), 42);
        // 7 / 8 = 0.875, truncated -> 87 (rounded would be 88).
        assert_eq!(usage_percentage_truncated_u8(7, 8), 87);
    }

    #[test]
    fn free_tokens_saturates() {
        assert_eq!(free_tokens(100, 30), 70);
        assert_eq!(free_tokens(100, 100), 0);
        assert_eq!(free_tokens(100, 200), 0);
    }

    #[test]
    fn exceeds_threshold_matches_integer_pct() {
        assert!(!exceeds_threshold(50, 100, 85));
        assert!(exceeds_threshold(85, 100, 85));
        assert!(exceeds_threshold(99, 100, 85));
        assert!(!exceeds_threshold(50, 0, 85));
    }

    /// Strict-boundary contract — pin the `>=` semantics. At cw=1000,
    /// pct=85, `850 * 100 == 1000 * 85` so the gate must fire at exactly
    /// 850 tokens. This is one token earlier than the legacy `>` gate
    /// (`total > cw * pct / 100` which fired at 851).
    #[test]
    fn exceeds_threshold_fires_on_strict_boundary() {
        assert!(exceeds_threshold(850, 1000, 85));
        assert!(!exceeds_threshold(849, 1000, 85));
        // 1000 * 85 / 100 = 850, so 850 is the new strict boundary.
        // Same shape at the other commonly-configured threshold (95%):
        assert!(exceeds_threshold(950, 1000, 95));
        assert!(!exceeds_threshold(949, 1000, 95));
    }

    #[test]
    fn compaction_reserve_is_a_quarter_capped() {
        assert_eq!(compaction_reserve_tokens(0), 0);
        assert_eq!(compaction_reserve_tokens(4_096), 1_024);
        assert_eq!(compaction_reserve_tokens(8_192), 2_048);
        assert_eq!(compaction_reserve_tokens(32_768), 8_192);
        // Capped past 32K — 1M would otherwise reserve 262K.
        assert_eq!(
            compaction_reserve_tokens(1_048_576),
            MAX_COMPACTION_RESERVE_TOKENS
        );
    }

    #[test]
    fn is_small_context_window_boundary() {
        // 0 means "unknown", not "smallest possible".
        assert!(!is_small_context_window(0));
        assert!(is_small_context_window(4_096));
        assert!(is_small_context_window(SMALL_CONTEXT_WINDOW_TOKENS));
        assert!(!is_small_context_window(SMALL_CONTEXT_WINDOW_TOKENS + 1));
        assert!(!is_small_context_window(1_048_576));
    }

    /// The scaling must not touch the windows hosted models report — those are
    /// already tuned at 85%.
    #[test]
    fn auto_compact_threshold_is_noop_for_large_windows() {
        for cw in [65_537_u64, 131_072, 200_000, 256_000, 1_048_576] {
            assert_eq!(
                auto_compact_threshold_for_window(cw, 85),
                85,
                "cw={cw} should keep the configured threshold"
            );
        }
    }

    #[test]
    fn auto_compact_threshold_tightens_small_windows() {
        // Below the cap the reserve is cw/4, so the threshold lands at 75%.
        assert_eq!(auto_compact_threshold_for_window(4_096, 85), 75);
        assert_eq!(auto_compact_threshold_for_window(8_192, 85), 75);
        assert_eq!(auto_compact_threshold_for_window(16_384, 85), 75);
        assert_eq!(auto_compact_threshold_for_window(32_768, 85), 75);
        // Between the cap and the no-op point it graduates back up.
        assert_eq!(auto_compact_threshold_for_window(49_152, 85), 83);
        assert_eq!(auto_compact_threshold_for_window(65_536, 85), 85);
    }

    /// Only ever tightens: a host that already asked for a low threshold keeps
    /// it, and an unknown window changes nothing.
    #[test]
    fn auto_compact_threshold_never_loosens() {
        assert_eq!(auto_compact_threshold_for_window(8_192, 50), 50);
        assert_eq!(auto_compact_threshold_for_window(1_048_576, 50), 50);
        assert_eq!(auto_compact_threshold_for_window(0, 85), 85);
        assert_eq!(auto_compact_threshold_for_window(0, 42), 42);
    }

    /// The resulting threshold must leave at least the reserve free, which is
    /// the whole point of the scaling.
    #[test]
    fn auto_compact_threshold_leaves_reserve_free() {
        for cw in [1_024_u64, 4_096, 8_192, 16_384, 32_768, 65_536] {
            let pct = auto_compact_threshold_for_window(cw, 85);
            let fires_at = cw * u64::from(pct) / 100;
            assert!(
                cw - fires_at >= compaction_reserve_tokens(cw),
                "cw={cw} pct={pct} leaves {} free, want >= {}",
                cw - fires_at,
                compaction_reserve_tokens(cw),
            );
        }
    }

    #[test]
    fn tool_output_budget_is_noop_for_large_windows() {
        assert_eq!(tool_output_budget_bytes(0, 20_000), 20_000);
        assert_eq!(tool_output_budget_bytes(131_072, 20_000), 20_000);
        assert_eq!(tool_output_budget_bytes(1_048_576, 20_000), 20_000);
    }

    #[test]
    fn tool_output_budget_scales_small_windows() {
        // cw/8 tokens, times BYTES_PER_TOKEN.
        assert_eq!(tool_output_budget_bytes(32_768, 20_000), 16_384);
        assert_eq!(tool_output_budget_bytes(16_384, 20_000), 8_192);
        assert_eq!(tool_output_budget_bytes(8_192, 20_000), 4_096);
        // Floored so a tiny window still gets a usable result.
        assert_eq!(
            tool_output_budget_bytes(2_048, 20_000),
            MIN_TOOL_OUTPUT_BUDGET_BYTES
        );
        // 65_536/8*4 = 32_768, above the default — clamped back down.
        assert_eq!(tool_output_budget_bytes(65_536, 20_000), 20_000);
    }

    /// The floor must not raise a cap the host set below it.
    #[test]
    fn tool_output_budget_never_exceeds_default() {
        assert_eq!(tool_output_budget_bytes(4_096, 500), 500);
        assert_eq!(tool_output_budget_bytes(32_768, 1_000), 1_000);
        assert_eq!(tool_output_budget_bytes(8_192, 0), 0);
    }

    /// Property: with `headroom == 0` the helper agrees with
    /// [`exceeds_threshold`] across a representative grid of inputs,
    /// including the non-round windows where floor-divide drifts.
    #[test]
    fn exceeds_threshold_with_headroom_zero_headroom_matches_exceeds_threshold() {
        for cw in [0_u64, 1, 50, 100, 101, 1024, 100_000, 128_001, 1_000_001] {
            for pct in [0_u8, 1, 50, 85, 99, 100] {
                for used in [
                    0_u64,
                    1,
                    cw / 2,
                    cw.saturating_sub(1),
                    cw,
                    cw + 1,
                    cw + 1000,
                ] {
                    assert_eq!(
                        exceeds_threshold_with_headroom(used, cw, pct, 0),
                        exceeds_threshold(used, cw, pct),
                        "mismatch at used={used} cw={cw} pct={pct}",
                    );
                }
            }
        }
    }

    #[test]
    fn exceeds_threshold_with_headroom_subtracts_headroom() {
        // 100K window, 85% threshold = 85_000. Headroom 4_000 -> fires at 81_000.
        assert!(!exceeds_threshold_with_headroom(80_999, 100_000, 85, 4_000));
        assert!(exceeds_threshold_with_headroom(81_000, 100_000, 85, 4_000));
    }

    #[test]
    fn exceeds_threshold_with_headroom_zero_window() {
        assert!(!exceeds_threshold_with_headroom(0, 0, 85, 0));
        assert!(!exceeds_threshold_with_headroom(100, 0, 85, 4_000));
    }

    #[test]
    fn exceeds_threshold_with_headroom_headroom_larger_than_threshold_saturates() {
        // 100K * 85% = 85_000 (8_500_000 scaled). Headroom 1M tokens scales to
        // 100_000_000 — saturating sub yields 0, so any used fires.
        assert!(exceeds_threshold_with_headroom(0, 100_000, 85, 1_000_000));
    }
}
