//! `snapjudge eval` (redesign §7, §9; Task 8). 8a: configuration, credentials, the request
//! cache with the model catalogue snapshot, and the run spend ledger; 8b: the pipeline
//! (selection, polish, inputs, split, reference, Jev, self-agreement, adjudication). The
//! provider transports live in `crate::llm` and `crate::jev`.

pub mod adjudicate;
pub mod answers;
pub mod cache;
pub mod config;
pub mod definition;
pub mod designer;
pub mod env;
pub mod export;
pub mod inputs;
pub mod ledger;
pub mod metrics;
pub mod pipeline;
pub mod reference;
pub mod run;
pub mod select;
