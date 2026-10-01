//! Run spend ledger (redesign §9 budgets; Task 8 frozen decision 9). Every provider attempt
//! reserves its worst-case cost before it is sent and settles afterwards: on the provider's
//! reported cost (`usage.cost`, or Jev input tokens at the configured price) when there is
//! one, otherwise the reservation is kept as spent. A failed attempt settles on
//! [`failed_attempt_cost`]: free when the provider answered with a status that means the
//! request was not run, else the reservation is kept; failed attempts are totalled apart
//! (`failed_attempts_usd`) so the estimated-vs-actual comparison covers answers only (8a
//! review amendment 5). A reservation that would take charged
//! plus in-flight spend beyond the budget is refused, every reservation (even of zero) is
//! refused once charged plus in-flight spend reaches the budget (so a zero budget allows
//! nothing), and every reservation after cancellation is refused, so a run stops issuing
//! requests at the budget. Shared across worker threads.

use std::fmt;
use std::sync::{Mutex, MutexGuard};

use crate::judge::config::{BYTES_PER_TOKEN, estimate_usd};
use crate::llm::catalogue::Price;

/// Worst-case cost of one chat attempt: `ceil(prompt_bytes / 3)` prompt tokens at the prompt
/// price plus `max_completion_tokens` (reasoning included) at the completion price.
pub fn estimate_llm(prompt_bytes: usize, max_completion_tokens: u32, price: Price) -> f64 {
    prompt_bytes.div_ceil(BYTES_PER_TOKEN) as f64 * price.prompt_usd_per_token
        + f64::from(max_completion_tokens) * price.completion_usd_per_token
}

/// Cost of one Jev attempt: request bytes / 3 input tokens at the Task 7 configured
/// `input_price_usd_per_mtok` (output is free).
pub fn estimate_jev(body_bytes: usize, input_price_usd_per_mtok: f64) -> f64 {
    estimate_usd(body_bytes, input_price_usd_per_mtok)
}

/// Jev's actual cost from reported input tokens.
pub fn jev_cost(input_tokens: u64, input_price_usd_per_mtok: f64) -> f64 {
    input_tokens as f64 * input_price_usd_per_mtok / 1e6
}

/// What a failed attempt is charged: nothing for a received non-2xx status other than 408 and
/// 5xx (the provider refused the request before running it), else unknown, so the
/// reservation is kept (a transport failure, timeout, 408 or 5xx may have run upstream, and a
/// 2xx that could not be read or parsed was answered).
pub fn failed_attempt_cost(http_status: Option<u16>) -> Option<f64> {
    match http_status {
        Some(status) if !(200..300).contains(&status) && status != 408 && status < 500 => Some(0.0),
        _ => None,
    }
}

/// Spend held for one in-flight attempt. Settle it with [`Reservation::settle`]; one dropped
/// unsettled (an early return or a panic) settles as unknown, keeping the reservation charged.
#[must_use = "a reservation must be settled"]
#[derive(Debug)]
pub struct Reservation<'a> {
    ledger: &'a Ledger,
    amount_usd: f64,
    settled: bool,
}

impl Reservation<'_> {
    pub fn amount_usd(&self) -> f64 {
        self.amount_usd
    }

    /// Release the reservation and charge the reported cost, or the whole reservation when
    /// no valid cost was reported.
    pub fn settle(mut self, actual_usd: Option<f64>) {
        self.settled = true;
        self.ledger.record(self.amount_usd, actual_usd, false);
    }

    /// Settle a failed attempt: charge `actual_usd` (see [`failed_attempt_cost`]), else the
    /// whole reservation, into the failed-attempt totals.
    pub fn settle_failed(mut self, actual_usd: Option<f64>) {
        self.settled = true;
        self.ledger.record(self.amount_usd, actual_usd, true);
    }
}

impl Drop for Reservation<'_> {
    fn drop(&mut self) {
        if !self.settled {
            self.settled = true;
            self.ledger.record(self.amount_usd, None, false);
        }
    }
}

/// Why a reservation was refused.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Refusal {
    /// The reservation does not fit in what is left of the budget.
    Budget {
        requested_usd: f64,
        available_usd: f64,
    },
    /// The run was cancelled.
    Cancelled,
    /// A negative or non-finite amount.
    InvalidAmount,
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refusal::Budget {
                requested_usd,
                available_usd,
            } => write!(
                f,
                "budget reached: the next request may cost ${requested_usd:.6}, ${available_usd:.6} left"
            ),
            Refusal::Cancelled => f.write_str("cancelled"),
            Refusal::InvalidAmount => f.write_str("invalid cost estimate"),
        }
    }
}

/// A snapshot of the ledger.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Totals {
    pub budget_usd: f64,
    /// Held by in-flight attempts.
    pub reserved_usd: f64,
    /// Counted against the budget: reported costs, else kept reservations, failed attempts
    /// included.
    pub charged_usd: f64,
    /// Sum of the reservations of settled attempts that were not failures.
    pub estimated_usd: f64,
    /// Sum of the reported costs of those attempts.
    pub actual_usd: f64,
    /// Settled attempts, failed ones included.
    pub attempts: u64,
    /// Settled attempts (not failures) without a reported cost.
    pub unknown_actual: u64,
    /// Settled failed attempts.
    pub failed_attempts: u64,
    /// Charged for failed attempts (kept reservations or reported costs).
    pub failed_attempts_usd: f64,
}

#[derive(Debug, Default)]
struct State {
    totals: Totals,
    cancelled: bool,
}

#[derive(Debug)]
pub struct Ledger {
    state: Mutex<State>,
}

impl Ledger {
    /// A ledger with this budget (a negative or non-finite budget allows nothing).
    pub fn new(budget_usd: f64) -> Self {
        let budget_usd = if budget_usd.is_finite() && budget_usd >= 0.0 {
            budget_usd
        } else {
            0.0
        };
        Self {
            state: Mutex::new(State {
                totals: Totals {
                    budget_usd,
                    ..Totals::default()
                },
                cancelled: false,
            }),
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        // A panicking holder leaves consistent numbers: every update is a single assignment.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Hold `amount_usd` for one attempt, or refuse when it does not fit, the budget is
    /// exhausted (whatever the amount) or the run was cancelled.
    pub fn reserve(&self, amount_usd: f64) -> Result<Reservation<'_>, Refusal> {
        if !amount_usd.is_finite() || amount_usd < 0.0 {
            return Err(Refusal::InvalidAmount);
        }
        let mut state = self.lock();
        if state.cancelled {
            return Err(Refusal::Cancelled);
        }
        let totals = &mut state.totals;
        let available_usd = (totals.budget_usd - totals.charged_usd - totals.reserved_usd).max(0.0);
        let exhausted = totals.charged_usd + totals.reserved_usd >= totals.budget_usd;
        if exhausted || amount_usd > available_usd {
            return Err(Refusal::Budget {
                requested_usd: amount_usd,
                available_usd,
            });
        }
        totals.reserved_usd += amount_usd;
        Ok(Reservation {
            ledger: self,
            amount_usd,
            settled: false,
        })
    }

    fn record(&self, amount_usd: f64, actual_usd: Option<f64>, failed: bool) {
        let actual_usd = actual_usd.filter(|c| c.is_finite() && *c >= 0.0);
        let mut state = self.lock();
        let totals = &mut state.totals;
        totals.reserved_usd = (totals.reserved_usd - amount_usd).max(0.0);
        totals.attempts += 1;
        if failed {
            let charged = actual_usd.unwrap_or(amount_usd);
            totals.charged_usd += charged;
            totals.failed_attempts_usd += charged;
            totals.failed_attempts += 1;
            return;
        }
        totals.estimated_usd += amount_usd;
        match actual_usd {
            Some(cost) => {
                totals.charged_usd += cost;
                totals.actual_usd += cost;
            }
            None => {
                totals.charged_usd += amount_usd;
                totals.unknown_actual += 1;
            }
        }
    }

    /// Refuse every later reservation; in-flight attempts still settle.
    pub fn cancel(&self) {
        self.lock().cancelled = true;
    }

    pub fn is_cancelled(&self) -> bool {
        self.lock().cancelled
    }

    pub fn totals(&self) -> Totals {
        self.lock().totals
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn estimates_use_prompt_bytes_and_max_completion_tokens() {
        let price = Price {
            prompt_usd_per_token: 4e-6,
            completion_usd_per_token: 2e-5,
        };
        // 3000 bytes → 1000 prompt tokens × 4e-6 + 500 × 2e-5.
        let estimate = estimate_llm(3000, 500, price);
        assert!((estimate - 0.014).abs() < 1e-12, "{estimate}");
        assert!(estimate_llm(1, 0, price) > 0.0);
        assert_eq!(estimate_jev(3_000_000, 0.042), 0.042);
        assert!((jev_cost(1_000_000, 0.042) - 0.042).abs() < 1e-15);
    }

    #[test]
    fn reserve_settle_and_refuse() {
        let ledger = Ledger::new(1.0);
        let a = ledger.reserve(0.6).unwrap();
        assert_eq!(ledger.totals().reserved_usd, 0.6);
        // In-flight spend counts: 0.6 + 0.5 > 1.0.
        assert_eq!(
            ledger.reserve(0.5).unwrap_err(),
            Refusal::Budget {
                requested_usd: 0.5,
                available_usd: 0.4
            }
        );
        // Reported cost replaces the reservation.
        a.settle(Some(0.1));
        let totals = ledger.totals();
        assert_eq!(totals.reserved_usd, 0.0);
        assert_eq!(totals.charged_usd, 0.1);
        assert_eq!(totals.actual_usd, 0.1);
        assert_eq!(totals.estimated_usd, 0.6);
        // No reported cost: the reservation is kept.
        let b = ledger.reserve(0.5).unwrap();
        b.settle(None);
        let totals = ledger.totals();
        assert!((totals.charged_usd - 0.6).abs() < 1e-12);
        assert_eq!(totals.unknown_actual, 1);
        assert_eq!(totals.attempts, 2);
        // An invalid reported cost counts as unknown.
        let c = ledger.reserve(0.1).unwrap();
        c.settle(Some(f64::NAN));
        assert_eq!(ledger.totals().unknown_actual, 2);
        assert!(ledger.reserve(0.31).is_err());
        assert_eq!(ledger.reserve(-1.0).unwrap_err(), Refusal::InvalidAmount);
        assert_eq!(
            ledger.reserve(f64::NAN).unwrap_err(),
            Refusal::InvalidAmount
        );
        let zero = ledger.reserve(0.0).unwrap();
        zero.settle(Some(0.0));
    }

    #[test]
    fn zero_reservations_are_refused_once_the_budget_is_exhausted() {
        let ledger = Ledger::new(1.0);
        let all = ledger.reserve(1.0).unwrap();
        // Fully held in flight: nothing more, not even a free attempt.
        assert!(matches!(ledger.reserve(0.0), Err(Refusal::Budget { .. })));
        all.settle(None);
        assert_eq!(ledger.totals().charged_usd, 1.0);
        assert_eq!(
            ledger.reserve(0.0).unwrap_err(),
            Refusal::Budget {
                requested_usd: 0.0,
                available_usd: 0.0
            }
        );
        assert!(Ledger::new(0.0).reserve(0.0).is_err());
    }

    #[test]
    fn an_unsettled_reservation_settles_as_unknown_on_drop() {
        let ledger = Ledger::new(1.0);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _held = ledger.reserve(0.25).unwrap();
            panic!("worker failed mid-attempt");
        }));
        assert!(result.is_err());
        let totals = ledger.totals();
        assert_eq!(totals.reserved_usd, 0.0);
        assert_eq!(totals.charged_usd, 0.25);
        assert_eq!(totals.unknown_actual, 1);
        assert_eq!(totals.attempts, 1);
        // A settled reservation is not charged again when dropped.
        ledger.reserve(0.25).unwrap().settle(Some(0.1));
        assert!((ledger.totals().charged_usd - 0.35).abs() < 1e-12);
        assert_eq!(ledger.totals().attempts, 2);
    }

    #[test]
    fn failed_attempts_are_totalled_apart() {
        for status in [400, 401, 402, 403, 404, 429, 302] {
            assert_eq!(failed_attempt_cost(Some(status)), Some(0.0), "{status}");
        }
        for status in [None, Some(200), Some(408), Some(500), Some(502), Some(599)] {
            assert_eq!(failed_attempt_cost(status), None, "{status:?}");
        }
        let ledger = Ledger::new(1.0);
        ledger
            .reserve(0.2)
            .unwrap()
            .settle_failed(failed_attempt_cost(Some(503)));
        ledger
            .reserve(0.3)
            .unwrap()
            .settle_failed(failed_attempt_cost(Some(429)));
        ledger.reserve(0.1).unwrap().settle(Some(0.05));
        let totals = ledger.totals();
        assert!((totals.charged_usd - 0.25).abs() < 1e-12, "{totals:?}");
        assert_eq!(totals.failed_attempts_usd, 0.2);
        assert_eq!(totals.failed_attempts, 2);
        assert_eq!(totals.attempts, 3);
        // The estimated-vs-actual comparison covers the answer only.
        assert_eq!(totals.estimated_usd, 0.1);
        assert_eq!(totals.actual_usd, 0.05);
        assert_eq!(totals.unknown_actual, 0);
        assert_eq!(totals.reserved_usd, 0.0);
    }

    #[test]
    fn cancellation_refuses_new_reservations_but_settles_in_flight_ones() {
        let ledger = Ledger::new(10.0);
        let in_flight = ledger.reserve(1.0).unwrap();
        ledger.cancel();
        assert!(ledger.is_cancelled());
        assert_eq!(ledger.reserve(0.01).unwrap_err(), Refusal::Cancelled);
        in_flight.settle(Some(0.5));
        assert_eq!(ledger.totals().charged_usd, 0.5);
    }

    #[test]
    fn invalid_budgets_allow_nothing() {
        for budget in [-1.0, f64::NAN, f64::INFINITY] {
            let ledger = Ledger::new(budget);
            assert!(ledger.reserve(0.001).is_err(), "{budget}");
            assert!(ledger.reserve(0.0).is_err(), "{budget}");
        }
    }

    #[test]
    fn concurrent_reservations_never_exceed_the_budget() {
        // 8 workers, each attempt holds 0.01 of a 1.00 budget and reports no cost: exactly
        // 100 attempts fit, whatever the interleaving.
        let ledger = Arc::new(Ledger::new(1.0));
        let workers: Vec<_> = (0..8)
            .map(|_| {
                let ledger = Arc::clone(&ledger);
                thread::spawn(move || {
                    let mut granted = 0u64;
                    while let Ok(reservation) = ledger.reserve(0.01) {
                        let totals = ledger.totals();
                        assert!(totals.charged_usd + totals.reserved_usd <= 1.0 + 1e-9);
                        reservation.settle(None);
                        granted += 1;
                    }
                    granted
                })
            })
            .collect();
        let granted: u64 = workers.into_iter().map(|w| w.join().unwrap()).sum();
        let totals = ledger.totals();
        assert_eq!(granted, totals.attempts);
        assert!((99..=100).contains(&granted), "{granted}");
        assert!(totals.charged_usd <= 1.0 + 1e-9);
        assert_eq!(totals.reserved_usd, 0.0);
    }
}
