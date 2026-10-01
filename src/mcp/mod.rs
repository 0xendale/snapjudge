//! Tools-only MCP server over stdio: newline-delimited UTF-8 JSON-RPC 2.0, one
//! message per line, stdout only protocol messages, logs on stderr, exit on EOF. Serves both
//! protocol eras: the handshake era (`initialize`, versions 2025-03-26, 2025-06-18,
//! 2025-11-25) and the 2026-07-28 era (per-request `_meta` protocol version,
//! `server/discover`). Requests are handled one at a time, in order.

use std::io::{self, BufRead, Write};
use std::panic::{self, AssertUnwindSafe};

use serde_json::{Map, Value, json};

use crate::tools::{CallError, Service, specs};

/// Handshake-era versions; an `initialize` asking for another gets [`LATEST_HANDSHAKE`].
pub const HANDSHAKE_VERSIONS: [&str; 3] = ["2025-03-26", "2025-06-18", "2025-11-25"];
pub const LATEST_HANDSHAKE: &str = "2025-11-25";
/// Versions accepted in a request's `_meta` (no handshake).
pub const MODERN_VERSIONS: [&str; 1] = ["2026-07-28"];
pub const META_VERSION: &str = "io.modelcontextprotocol/protocolVersion";
pub const META_CAPABILITIES: &str = "io.modelcontextprotocol/clientCapabilities";
pub const META_SERVER_INFO: &str = "io.modelcontextprotocol/serverInfo";
/// Longest accepted message line (bytes, newline excluded).
pub const MAX_LINE_BYTES: usize = 4 << 20;

pub const PARSE_ERROR: i64 = -32700;
pub const INVALID_REQUEST: i64 = -32600;
pub const METHOD_NOT_FOUND: i64 = -32601;
pub const INVALID_PARAMS: i64 = -32602;
pub const INTERNAL_ERROR: i64 = -32603;
pub const UNSUPPORTED_PROTOCOL_VERSION: i64 = -32022;

const METHODS: [&str; 5] = [
    "initialize",
    "ping",
    "server/discover",
    "tools/list",
    "tools/call",
];

enum Line {
    Text(Vec<u8>),
    TooLong,
}

/// Read one line of at most `max` bytes (without the newline); a longer line is consumed
/// and reported as [`Line::TooLong`]. `None` at EOF.
fn read_line<R: BufRead>(reader: &mut R, max: usize) -> io::Result<Option<Line>> {
    let mut buf = Vec::new();
    let mut too_long = false;
    let mut any = false;
    loop {
        let available = match reader.fill_buf() {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        if available.is_empty() {
            return Ok(match (any, too_long) {
                (false, _) => None,
                (true, true) => Some(Line::TooLong),
                (true, false) => Some(Line::Text(buf)),
            });
        }
        any = true;
        let newline = available.iter().position(|&b| b == b'\n');
        let chunk = &available[..newline.unwrap_or(available.len())];
        if !too_long {
            if buf.len() + chunk.len() > max {
                too_long = true;
                buf = Vec::new();
            } else {
                buf.extend_from_slice(chunk);
            }
        }
        let consumed = newline.map_or(available.len(), |i| i + 1);
        reader.consume(consumed);
        if newline.is_some() {
            return Ok(Some(if too_long {
                Line::TooLong
            } else {
                Line::Text(buf)
            }));
        }
    }
}

fn server_info() -> Value {
    json!({"name": "snapjudge", "title": "snapjudge", "version": env!("CARGO_PKG_VERSION")})
}

fn capabilities() -> Value {
    json!({"tools": {"listChanged": false}})
}

fn error(id: Value, code: i64, message: &str, data: Option<Value>) -> Value {
    let mut error = json!({"code": code, "message": message});
    if let Some(data) = data {
        error["data"] = data;
    }
    json!({"jsonrpc": "2.0", "id": id, "error": error})
}

/// Which era serves a request.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Era {
    Handshake,
    Modern,
}

pub struct Server<'a> {
    service: &'a Service,
    /// Version agreed by the last `initialize`.
    negotiated: Option<String>,
}

impl<'a> Server<'a> {
    pub fn new(service: &'a Service) -> Self {
        Self {
            service,
            negotiated: None,
        }
    }

    /// Serve until EOF. Responses go to `output`, one per line; log lines to `log`.
    pub fn serve<R: BufRead, W: Write, L: Write>(
        &mut self,
        mut input: R,
        mut output: W,
        mut log: L,
    ) -> io::Result<()> {
        while let Some(line) = read_line(&mut input, MAX_LINE_BYTES)? {
            let reply = match line {
                Line::TooLong => Some(error(
                    Value::Null,
                    INVALID_REQUEST,
                    &format!("message exceeds {MAX_LINE_BYTES} bytes"),
                    None,
                )),
                Line::Text(bytes) => self.handle_line(&bytes, &mut log),
            };
            if let Some(reply) = reply {
                let text = serde_json::to_string(&reply).map_err(io::Error::other)?;
                output.write_all(text.as_bytes())?;
                output.write_all(b"\n")?;
                output.flush()?;
            }
        }
        Ok(())
    }

    fn handle_line<L: Write>(&mut self, bytes: &[u8], log: &mut L) -> Option<Value> {
        let bytes = bytes.strip_suffix(b"\r").unwrap_or(bytes);
        if bytes.iter().all(u8::is_ascii_whitespace) {
            return None;
        }
        let Ok(text) = std::str::from_utf8(bytes) else {
            return Some(error(
                Value::Null,
                PARSE_ERROR,
                "Parse error: not UTF-8",
                None,
            ));
        };
        let message: Value = match serde_json::from_str(text) {
            Ok(message) => message,
            Err(e) => {
                return Some(error(
                    Value::Null,
                    PARSE_ERROR,
                    &format!("Parse error: {e}"),
                    None,
                ));
            }
        };
        let Some(object) = message.as_object() else {
            return Some(error(
                Value::Null,
                INVALID_REQUEST,
                "Invalid Request: a message must be one JSON object (no batches)",
                None,
            ));
        };
        let method = object.get("method");
        // Notifications (`notifications/initialized`, `notifications/cancelled`, …) are
        // never answered; requests run to completion before the next line is read, so a
        // cancellation always arrives after its request's response.
        if method.is_some() && !object.contains_key("id") {
            return None;
        }
        // A client response (we send no requests): nothing to answer.
        if method.is_none()
            && object.contains_key("id")
            && (object.contains_key("result") || object.contains_key("error"))
        {
            return None;
        }
        let id = match object.get("id") {
            Some(id @ Value::String(_)) => id.clone(),
            Some(Value::Number(n)) if n.is_i64() || n.is_u64() => Value::Number(n.clone()),
            _ => {
                return Some(error(
                    Value::Null,
                    INVALID_REQUEST,
                    "Invalid Request: id must be a string or an integer",
                    None,
                ));
            }
        };
        if object.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
            return Some(error(
                id,
                INVALID_REQUEST,
                "Invalid Request: jsonrpc must be \"2.0\"",
                None,
            ));
        }
        let Some(method) = method.and_then(Value::as_str) else {
            return Some(error(
                id,
                INVALID_REQUEST,
                "Invalid Request: method must be a string",
                None,
            ));
        };
        let empty = Map::new();
        let params = match object.get("params") {
            None => &empty,
            Some(Value::Object(params)) => params,
            Some(_) => {
                return Some(error(
                    id,
                    INVALID_PARAMS,
                    "Invalid params: params must be an object",
                    None,
                ));
            }
        };
        Some(self.request(id, method, params, log))
    }

    fn request<L: Write>(
        &mut self,
        id: Value,
        method: &str,
        params: &Map<String, Value>,
        log: &mut L,
    ) -> Value {
        if !METHODS.contains(&method) {
            return error(
                id,
                METHOD_NOT_FOUND,
                &format!("Method not found: {method}"),
                None,
            );
        }
        if method == "initialize" {
            return self.initialize(id, params);
        }
        let era = match self.era(method, params) {
            Ok(era) => era,
            Err((code, message, data)) => return error(id, code, &message, data),
        };
        let result = match method {
            "ping" => json!({}),
            "server/discover" => json!({
                "supportedVersions": MODERN_VERSIONS,
                "capabilities": capabilities(),
            }),
            "tools/list" => json!({"tools": specs().into_iter().map(|tool| json!({
                "name": tool.name,
                "title": tool.title,
                "description": tool.description,
                "inputSchema": tool.input_schema,
                "outputSchema": tool.output_schema,
                "annotations": tool.annotations,
            })).collect::<Vec<_>>()}),
            _ => match self.call(params, log) {
                Ok(result) => result,
                Err((code, message)) => return error(id, code, &message, None),
            },
        };
        let result = match (era, result) {
            (Era::Modern, Value::Object(mut result)) => {
                result.insert("resultType".into(), json!("complete"));
                result.insert("_meta".into(), json!({META_SERVER_INFO: server_info()}));
                Value::Object(result)
            }
            (_, result) => result,
        };
        json!({"jsonrpc": "2.0", "id": id, "result": result})
    }

    fn initialize(&mut self, id: Value, params: &Map<String, Value>) -> Value {
        let Some(requested) = params.get("protocolVersion").and_then(Value::as_str) else {
            return error(
                id,
                INVALID_PARAMS,
                "Invalid params: protocolVersion must be a string",
                None,
            );
        };
        let version = if HANDSHAKE_VERSIONS.contains(&requested) {
            requested
        } else {
            LATEST_HANDSHAKE
        };
        self.negotiated = Some(version.to_string());
        json!({"jsonrpc": "2.0", "id": id, "result": {
            "protocolVersion": version,
            "capabilities": capabilities(),
            "serverInfo": server_info(),
        }})
    }

    /// A request carrying `_meta` with a protocol version is served in the 2026-07-28 era;
    /// without it, only `ping` or a request after `initialize` (handshake era).
    fn era(
        &self,
        method: &str,
        params: &Map<String, Value>,
    ) -> Result<Era, (i64, String, Option<Value>)> {
        let meta = params.get("_meta").and_then(Value::as_object);
        match meta.and_then(|meta| meta.get(META_VERSION)) {
            Some(Value::String(version)) => {
                if !MODERN_VERSIONS.contains(&version.as_str()) {
                    return Err((
                        UNSUPPORTED_PROTOCOL_VERSION,
                        "Unsupported protocol version".into(),
                        Some(json!({"supported": MODERN_VERSIONS, "requested": version})),
                    ));
                }
                if !meta
                    .is_some_and(|meta| meta.get(META_CAPABILITIES).is_some_and(Value::is_object))
                {
                    return Err((
                        INVALID_PARAMS,
                        format!("Invalid params: _meta[\"{META_CAPABILITIES}\"] must be an object"),
                        None,
                    ));
                }
                Ok(Era::Modern)
            }
            Some(_) => Err((
                INVALID_PARAMS,
                format!("Invalid params: _meta[\"{META_VERSION}\"] must be a string"),
                None,
            )),
            None if method == "ping" => Ok(Era::Handshake),
            None if self.negotiated.is_some() && method != "server/discover" => Ok(Era::Handshake),
            None => Err((
                INVALID_PARAMS,
                format!(
                    "Invalid params: _meta[\"{META_VERSION}\"] is required (or send initialize first)"
                ),
                None,
            )),
        }
    }

    fn call<L: Write>(
        &self,
        params: &Map<String, Value>,
        log: &mut L,
    ) -> Result<Value, (i64, String)> {
        let Some(name) = params.get("name").and_then(Value::as_str) else {
            return Err((
                INVALID_PARAMS,
                "Invalid params: name must be a string".into(),
            ));
        };
        let empty = json!({});
        let arguments = match params.get("arguments") {
            None => &empty,
            Some(arguments @ Value::Object(_)) => arguments,
            Some(_) => {
                return Err((
                    INVALID_PARAMS,
                    "Invalid params: arguments must be an object".into(),
                ));
            }
        };
        let service = self.service;
        let outcome = panic::catch_unwind(AssertUnwindSafe(|| service.call(name, arguments)))
            .unwrap_or_else(|_| Err(CallError::Internal("the tool panicked".into())));
        let output = match outcome {
            Ok(output) => output,
            Err(e @ CallError::Internal(_)) => return Err((INTERNAL_ERROR, e.to_string())),
            Err(e) => return Err((INVALID_PARAMS, e.to_string())),
        };
        if let Some(line) = &output.log {
            let _ = writeln!(log, "{line}");
        }
        let text = serde_json::to_string(&output.structured)
            .map_err(|e| (INTERNAL_ERROR, format!("Internal error: {e}")))?;
        Ok(json!({
            "content": [{"type": "text", "text": text}],
            "structuredContent": output.structured,
            "isError": output.is_error,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(input: &[u8], max: usize) -> Vec<Option<Vec<u8>>> {
        let mut reader = io::BufReader::with_capacity(4, input);
        let mut out = Vec::new();
        while let Some(line) = read_line(&mut reader, max).unwrap() {
            out.push(match line {
                Line::Text(bytes) => Some(bytes),
                Line::TooLong => None,
            });
        }
        out
    }

    #[test]
    fn lines_are_bounded_and_split_on_newlines() {
        assert_eq!(
            lines(b"ab\n0123456789\ncd", 5),
            vec![Some(b"ab".to_vec()), None, Some(b"cd".to_vec())]
        );
        assert_eq!(lines(b"", 5), Vec::<Option<Vec<u8>>>::new());
        assert_eq!(lines(b"\n\n", 5), vec![Some(vec![]), Some(vec![])]);
        assert_eq!(
            lines(b"12345\n123456", 5),
            vec![Some(b"12345".to_vec()), None]
        );
    }
}
