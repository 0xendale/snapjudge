//! Tiny parser for Python type annotations and simple calls such as
//! `Annotated[int, Field(ge=1, le=5)]`. Text-based so it does not depend on
//! tree-sitter's annotation node shapes.

#[derive(Debug, Clone, PartialEq)]
pub enum PyExpr {
    Name(String),
    Str(String),
    Num(f64),
    List(Vec<PyExpr>),
    Subscript(Box<PyExpr>, Vec<PyExpr>),
    Call {
        func: Box<PyExpr>,
        args: Vec<PyExpr>,
        kwargs: Vec<(String, PyExpr)>,
    },
    Union(Vec<PyExpr>),
    Other(String),
}

impl PyExpr {
    /// Last dotted segment of a name: `typing.Literal` → `Literal`.
    pub fn base_name(&self) -> Option<&str> {
        match self {
            PyExpr::Name(n) => n.rsplit('.').next(),
            _ => None,
        }
    }

    /// Keyword argument of a call expression.
    pub fn kwarg(&self, key: &str) -> Option<&PyExpr> {
        match self {
            PyExpr::Call { kwargs, .. } => kwargs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
}

pub fn parse(text: &str) -> PyExpr {
    let other = || PyExpr::Other(text.trim().to_string());
    let Some(toks) = tokenize(text) else {
        return other();
    };
    let mut p = Parser { toks: &toks, i: 0 };
    match p.union() {
        Some(e) if p.i == toks.len() => e,
        _ => other(),
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Name(String),
    Str(String),
    Num(f64),
    Punct(char),
}

fn tokenize(s: &str) -> Option<Vec<Tok>> {
    let cs: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < cs.len() {
        let c = cs[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if "[](),=|".contains(c) {
            out.push(Tok::Punct(c));
            i += 1;
            continue;
        }
        if c == '.' && cs.get(i + 1) == Some(&'.') && cs.get(i + 2) == Some(&'.') {
            out.push(Tok::Name("...".into()));
            i += 3;
            continue;
        }
        let quote_at = if c == '"' || c == '\'' {
            Some(i)
        } else if "rbuRBU".contains(c) && matches!(cs.get(i + 1), Some('"' | '\'')) {
            Some(i + 1)
        } else {
            None
        };
        if let Some(q) = quote_at {
            let quote = cs[q];
            let mut j = q + 1;
            let mut val = String::new();
            loop {
                let ch = *cs.get(j)?;
                if ch == '\\' {
                    val.push(*cs.get(j + 1)?);
                    j += 2;
                    continue;
                }
                if ch == quote {
                    break;
                }
                val.push(ch);
                j += 1;
            }
            out.push(Tok::Str(val));
            i = j + 1;
            continue;
        }
        if c.is_ascii_digit() || (c == '-' && cs.get(i + 1).is_some_and(|d| d.is_ascii_digit())) {
            let mut j = i + 1;
            while j < cs.len() && (cs[j].is_ascii_digit() || cs[j] == '.' || cs[j] == '_') {
                j += 1;
            }
            let t: String = cs[i..j].iter().filter(|c| **c != '_').collect();
            out.push(Tok::Num(t.parse().ok()?));
            i = j;
            continue;
        }
        if c.is_alphabetic() || c == '_' {
            let mut j = i + 1;
            while j < cs.len() && (cs[j].is_alphanumeric() || cs[j] == '_' || cs[j] == '.') {
                j += 1;
            }
            out.push(Tok::Name(cs[i..j].iter().collect()));
            i = j;
            continue;
        }
        return None;
    }
    Some(out)
}

/// Positional and keyword arguments of a call.
type CallArgs = (Vec<PyExpr>, Vec<(String, PyExpr)>);

struct Parser<'a> {
    toks: &'a [Tok],
    i: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&'a Tok> {
        self.toks.get(self.i)
    }

    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(&Tok::Punct(c)) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn union(&mut self) -> Option<PyExpr> {
        let mut parts = vec![self.postfix()?];
        while self.eat('|') {
            parts.push(self.postfix()?);
        }
        Some(if parts.len() == 1 {
            parts.remove(0)
        } else {
            PyExpr::Union(parts)
        })
    }

    fn postfix(&mut self) -> Option<PyExpr> {
        let mut e = self.atom()?;
        loop {
            if self.eat('[') {
                e = PyExpr::Subscript(Box::new(e), self.items(']')?);
            } else if self.eat('(') {
                let (args, kwargs) = self.call_args()?;
                e = PyExpr::Call {
                    func: Box::new(e),
                    args,
                    kwargs,
                };
            } else {
                return Some(e);
            }
        }
    }

    fn atom(&mut self) -> Option<PyExpr> {
        let t = self.peek()?.clone();
        self.i += 1;
        match t {
            Tok::Name(n) => Some(PyExpr::Name(n)),
            Tok::Num(n) => Some(PyExpr::Num(n)),
            Tok::Str(mut s) => {
                // implicit concatenation: "a" "b" (also splits of triple-quoted strings)
                while let Some(Tok::Str(next)) = self.peek() {
                    s.push_str(next);
                    self.i += 1;
                }
                Some(PyExpr::Str(s))
            }
            Tok::Punct('(') => {
                let e = self.union()?;
                self.eat(')').then_some(e)
            }
            Tok::Punct('[') => Some(PyExpr::List(self.items(']')?)),
            Tok::Punct(_) => None,
        }
    }

    fn items(&mut self, close: char) -> Option<Vec<PyExpr>> {
        let mut items = Vec::new();
        while !self.eat(close) {
            items.push(self.union()?);
            if !self.eat(',') {
                return self.eat(close).then_some(items);
            }
        }
        Some(items)
    }

    fn call_args(&mut self) -> Option<CallArgs> {
        let (mut args, mut kwargs) = (Vec::new(), Vec::new());
        while !self.eat(')') {
            match (self.toks.get(self.i), self.toks.get(self.i + 1)) {
                (Some(Tok::Name(n)), Some(Tok::Punct('='))) => {
                    self.i += 2;
                    kwargs.push((n.clone(), self.union()?));
                }
                _ => args.push(self.union()?),
            }
            if !self.eat(',') {
                return self.eat(')').then_some((args, kwargs));
            }
        }
        Some((args, kwargs))
    }
}

#[cfg(test)]
mod tests {
    use super::PyExpr::*;
    use super::*;

    fn name(n: &str) -> Box<PyExpr> {
        Box::new(Name(n.into()))
    }

    #[test]
    fn literal_and_generics() {
        assert_eq!(
            parse(r#"Literal["a", 'b']"#),
            Subscript(name("Literal"), vec![Str("a".into()), Str("b".into())])
        );
        assert_eq!(
            parse("typing.Optional[int]"),
            Subscript(name("typing.Optional"), vec![Name("int".into())])
        );
        assert_eq!(Name("typing.Optional".into()).base_name(), Some("Optional"));
    }

    #[test]
    fn union_with_none() {
        assert_eq!(
            parse("list[Literal['x', 'y']] | None"),
            Union(vec![
                Subscript(
                    name("list"),
                    vec![Subscript(
                        name("Literal"),
                        vec![Str("x".into()), Str("y".into())]
                    )]
                ),
                Name("None".into())
            ])
        );
    }

    #[test]
    fn annotated_with_field_call() {
        let e = parse(r#"Annotated[int, Field(..., ge=1, le=5, description="Stars")]"#);
        let Subscript(_, args) = &e else {
            panic!("{e:?}")
        };
        assert_eq!(args[1].kwarg("ge"), Some(&Num(1.0)));
        assert_eq!(args[1].kwarg("le"), Some(&Num(5.0)));
        assert_eq!(args[1].kwarg("description"), Some(&Str("Stars".into())));
        assert_eq!(parse("conint(ge=0, le=10)").kwarg("le"), Some(&Num(10.0)));
    }

    #[test]
    fn numbers_strings_and_concatenation() {
        assert_eq!(parse("-1"), Num(-1.0));
        assert_eq!(parse("0.5"), Num(0.5));
        assert_eq!(parse(r#""a" "b""#), Str("ab".into()));
        assert_eq!(parse(r#""""doc""""#), Str("doc".into()));
        assert_eq!(parse(r"'it\'s'"), Str("it's".into()));
        assert_eq!(
            parse("['a', 'b']"),
            List(vec![Str("a".into()), Str("b".into())])
        );
    }

    #[test]
    fn unsupported_text_is_other() {
        assert_eq!(parse("lambda x: x"), Other("lambda x: x".into()));
        assert_eq!(parse("{'a': 1}"), Other("{'a': 1}".into()));
        assert_eq!(parse("Literal['a'"), Other("Literal['a'".into()));
    }
}
