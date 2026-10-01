use std::fs;
use std::path::{Component, Path, PathBuf};

use crate::eval::inputs::{self, SuppliedRow};

const DEFAULT_OUT: &str = ".snapjudge/eval";

fn relative(value: &str) -> Result<PathBuf, &'static str> {
    let path = Path::new(value);
    let mut depth = 0usize;
    for component in path.components() {
        match component {
            Component::Prefix(_) | Component::RootDir => return Err("must be workspace-relative"),
            Component::ParentDir if depth == 0 => return Err("must stay inside the workspace"),
            Component::ParentDir => depth -= 1,
            Component::Normal(_) => depth += 1,
            Component::CurDir => {}
        }
    }
    Ok(path.to_path_buf())
}

pub(super) fn confined_file(workspace: &Path, value: &str) -> Result<PathBuf, String> {
    let relative = relative(value).map_err(str::to_string)?;
    let canonical =
        fs::canonicalize(workspace.join(relative)).map_err(|error| error.to_string())?;
    if !canonical.starts_with(workspace) || !canonical.is_file() {
        return Err("must be a file inside the workspace".into());
    }
    Ok(canonical)
}

pub(super) fn supplied(
    workspace: &Path,
    path: Option<&str>,
) -> Result<Option<Vec<SuppliedRow>>, String> {
    let Some(path) = path else { return Ok(None) };
    let path = confined_file(workspace, path)?;
    if fs::metadata(&path)
        .map_err(|error| error.to_string())?
        .len()
        > inputs::MAX_INPUTS_BYTES
    {
        return Err(format!("inputs exceed {} bytes", inputs::MAX_INPUTS_BYTES));
    }
    inputs::parse_jsonl(&fs::read_to_string(path).map_err(|error| error.to_string())?).map(Some)
}

pub(super) fn validate_out(
    workspace: &Path,
    value: Option<&str>,
) -> Result<PathBuf, (&'static str, String)> {
    let out = relative(value.unwrap_or(DEFAULT_OUT))
        .map_err(|error| ("path_outside_workspace", error.into()))?;
    let mut target = workspace.to_path_buf();
    for component in out.components() {
        target.push(component);
        if let Ok(metadata) = fs::symlink_metadata(&target)
            && (!metadata.is_dir() || metadata.file_type().is_symlink())
        {
            return Err((
                "artifact_write_failed",
                format!("{} is not a real directory", target.display()),
            ));
        }
    }
    Ok(out)
}

pub(super) fn artifact_paths(workspace: &Path, dir: &Path, policy: bool) -> Vec<String> {
    let mut names = vec!["dataset.jsonl", "results.json", "report.md", "run.json"];
    if policy {
        names.push("policy.json");
    }
    names
        .into_iter()
        .map(|name| {
            let path = dir.join(name);
            path.strip_prefix(workspace)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect()
}
