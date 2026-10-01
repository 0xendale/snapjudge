//! Import gate: which LLM SDKs a file imports (spec §4 "SDK import gate").

use std::sync::LazyLock;

use regex::Regex;

use crate::model::{Lang, Sdk};

static PY: LazyLock<Vec<(Sdk, Regex)>> = LazyLock::new(|| {
    let r = |pkg: &str| {
        Regex::new(&format!(
            r"(?m)^[ \t]*(?:from[ \t]+{pkg}(?:\.[\w.]+)?[ \t]+import\b|import[ \t]+{pkg}\b)"
        ))
        .unwrap()
    };
    vec![
        (Sdk::Openai, r("openai")),
        (Sdk::Anthropic, r("anthropic")),
        (Sdk::Langchain, r(r"langchain(?:_\w+)?")),
        (Sdk::Instructor, r("instructor")),
        (Sdk::Litellm, r("litellm")),
    ]
});

static TS: LazyLock<Vec<(Sdk, Regex)>> = LazyLock::new(|| {
    let r = |spec: &str| {
        Regex::new(&format!(
            r#"(?:\bfrom\s*|\bimport\s*|\brequire\(\s*)['"](?:{spec})['"]"#
        ))
        .unwrap()
    };
    vec![
        (Sdk::Openai, r(r#"openai(?:/[^'"]*)?"#)),
        (Sdk::Anthropic, r(r#"@anthropic-ai/sdk(?:/[^'"]*)?"#)),
        (Sdk::AiSdk, r("ai")),
        (
            Sdk::Langchain,
            r(r#"@langchain/[^'"]+|langchain(?:/[^'"]*)?"#),
        ),
    ]
});

/// Top-level Python packages of the supported SDKs (the import gate's names) and the
/// known LangChain integration packages.
const PY_SDK_PACKAGES: &[&str] = &[
    "openai",
    "anthropic",
    "instructor",
    "litellm",
    "langchain",
    "langchain_openai",
    "langchain_anthropic",
    "langchain_core",
    "langchain_community",
];

/// True when the absolute Python module `module` belongs to a known SDK package
/// (`openai`, `openai.types`, `langchain_openai`, ...): never a project file. Other
/// `langchain_*` names may be project modules (see `FileTable::python_module`).
pub fn is_python_sdk_module(module: &str) -> bool {
    let first = module.split('.').next().unwrap_or_default();
    PY_SDK_PACKAGES.contains(&first)
}

/// SDKs imported by a source file, in `Sdk` order.
pub fn imported_sdks(src: &str, lang: Lang) -> Vec<Sdk> {
    let table = if lang == Lang::Python { &*PY } else { &*TS };
    let mut out: Vec<Sdk> = table
        .iter()
        .filter(|(_, re)| re.is_match(src))
        .map(|(sdk, _)| *sdk)
        .collect();
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_imports() {
        let src = "import os\nfrom openai import OpenAI\nimport instructor\nfrom langchain_openai import ChatOpenAI\n";
        assert_eq!(
            imported_sdks(src, Lang::Python),
            vec![Sdk::Openai, Sdk::Langchain, Sdk::Instructor]
        );
        assert_eq!(
            imported_sdks(
                "import anthropic\nfrom litellm import completion\n",
                Lang::Python
            ),
            vec![Sdk::Anthropic, Sdk::Litellm]
        );
        assert_eq!(
            imported_sdks("from openai.types import ChatModel\n", Lang::Python),
            vec![Sdk::Openai]
        );
        assert!(
            imported_sdks(
                "# import openai later\nx = 'from openai import OpenAI'\n",
                Lang::Python
            )
            .is_empty()
        );
    }

    #[test]
    fn python_sdk_modules_are_the_known_packages_only() {
        for module in [
            "openai",
            "openai.types",
            "anthropic",
            "instructor",
            "litellm",
            "langchain",
            "langchain.chat_models",
            "langchain_openai",
            "langchain_anthropic",
            "langchain_core.prompts",
            "langchain_community",
        ] {
            assert!(is_python_sdk_module(module), "{module}");
        }
        for module in [
            "langchain_utils",
            "langchain_google",
            "openai_utils",
            "app.openai",
        ] {
            assert!(!is_python_sdk_module(module), "{module}");
        }
    }

    #[test]
    fn ts_imports() {
        let src = "import OpenAI from 'openai';\nimport { zodResponseFormat } from \"openai/helpers/zod\";\nimport Anthropic from '@anthropic-ai/sdk';\n";
        assert_eq!(
            imported_sdks(src, Lang::Typescript),
            vec![Sdk::Openai, Sdk::Anthropic]
        );
        assert_eq!(
            imported_sdks(
                "import { generateObject } from 'ai';\nimport { ChatOpenAI } from '@langchain/openai';\n",
                Lang::Typescript
            ),
            vec![Sdk::AiSdk, Sdk::Langchain]
        );
        assert_eq!(
            imported_sdks("const OpenAI = require('openai');\n", Lang::Javascript),
            vec![Sdk::Openai]
        );
        assert!(
            imported_sdks(
                "import { ai } from './ai';\nimport x from 'openai-edge-lite';\n",
                Lang::Typescript
            )
            .is_empty()
        );
    }
}
