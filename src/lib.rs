//! snapjudge: find LLM calls that are closed-set decisions and measure whether TypeSafe Jev can take them over.

pub mod adapter;
pub mod cache;
pub mod cli;
pub mod decision;
pub mod eval;
pub mod jev;
pub mod judge;
pub mod llm;
pub mod mcp;
pub mod model;
pub mod report;
pub mod scan;
pub mod tools;
