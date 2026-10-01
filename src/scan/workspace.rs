//! All files of one scan: texts, parsed trees in an append-only arena (so nodes from
//! many files can live together), and per-file symbol scopes with project imports merged.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::rc::Rc;

use rayon::prelude::*;
use typed_arena::Arena;

use crate::scan::function::{self, FnDef};
use crate::scan::modules::{self, FileTable, Import};
use crate::scan::syntax::{self, Index, LocalCache, N, Root};
use crate::scan::walk::SourceFile;

pub struct Workspace<'a> {
    pub files: Vec<SourceFile>,
    /// `None` when the file is not valid UTF-8.
    pub texts: Vec<Option<String>>,
    pub table: FileTable,
    arena: &'a Arena<Root>,
    roots: RefCell<HashMap<usize, &'a Root>>,
    indexes: RefCell<HashMap<usize, FileIndexes<'a>>>,
    scopes: RefCell<HashMap<usize, Rc<Index<'a>>>>,
    imports: RefCell<HashMap<usize, Rc<Vec<Import>>>>,
    locals: RefCell<HashMap<usize, Rc<Vec<String>>>>,
    definitions: RefCell<HashMap<usize, Rc<Vec<FnDef>>>>,
    /// Function locals, shared by every scope (collected once per function).
    locals_cache: Rc<LocalCache<'a>>,
}

#[derive(Clone)]
struct FileIndexes<'a> {
    full: Rc<Index<'a>>,
    module: Rc<Index<'a>>,
}

impl<'a> Workspace<'a> {
    pub fn new(files: Vec<SourceFile>, arena: &'a Arena<Root>) -> Self {
        let texts = files
            .par_iter()
            .map(|f| fs::read_to_string(&f.path).ok())
            .collect();
        let table = FileTable::new(files.iter().map(|f| f.rel.clone()).collect());
        Self {
            files,
            texts,
            table,
            arena,
            roots: RefCell::default(),
            indexes: RefCell::default(),
            scopes: RefCell::default(),
            imports: RefCell::default(),
            locals: RefCell::default(),
            definitions: RefCell::default(),
            locals_cache: Rc::default(),
        }
    }

    pub fn ensure_parsed(&self, ids: &[usize]) {
        for &id in ids {
            self.assert_id(id);
        }
        let todo: Vec<usize> = {
            let roots = self.roots.borrow();
            let mut v: Vec<usize> = ids
                .iter()
                .copied()
                .filter(|i| !roots.contains_key(i) && self.texts[*i].is_some())
                .collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        let (texts, files) = (&self.texts, &self.files);
        let parsed: Vec<(usize, Root)> = todo
            .par_iter()
            .map(|&i| {
                (
                    i,
                    syntax::parse(texts[i].as_deref().unwrap_or_default(), files[i].grammar),
                )
            })
            .collect();
        let arena: &'a Arena<Root> = self.arena;
        let mut roots = self.roots.borrow_mut();
        for (i, root) in parsed {
            roots.insert(i, arena.alloc(root));
        }
    }

    pub fn root(&self, id: usize) -> Option<N<'a>> {
        self.assert_id(id);
        self.roots
            .borrow()
            .get(&id)
            .copied()
            .map(|r: &'a Root| r.root())
    }

    fn indexes(&self, id: usize) -> Option<FileIndexes<'a>> {
        self.assert_id(id);
        if let Some(indexes) = self.indexes.borrow().get(&id) {
            return Some(indexes.clone());
        }
        let built = syntax::indexes(&self.root(id)?);
        let indexes = FileIndexes {
            full: Rc::new(built.full),
            module: Rc::new(built.module),
        };
        self.indexes.borrow_mut().insert(id, indexes.clone());
        Some(indexes)
    }

    /// Project imports of file `id`, with TS default imports resolved to the target's
    /// default-exported name.
    pub fn imports(&self, id: usize) -> Rc<Vec<Import>> {
        self.assert_id(id);
        if let Some(i) = self.imports.borrow().get(&id) {
            return i.clone();
        }
        self.ensure_parsed(&[id]);
        let mut list = match self.root(id) {
            Some(root) => modules::project_imports(
                &root,
                &self.files[id].rel,
                self.files[id].grammar.lang(),
                &self.table,
            ),
            None => Vec::new(),
        };
        for import in &mut list {
            if import.remote.as_deref() != Some("default") {
                continue;
            }
            self.ensure_parsed(&[import.target]);
            if let Some(name) = self
                .root(import.target)
                .and_then(|r| modules::default_export(&r))
            {
                import.remote = Some(name);
            }
        }
        let rc = Rc::new(list);
        self.imports.borrow_mut().insert(id, rc.clone());
        rc
    }

    /// Named function definitions of file `id` (empty when it cannot be parsed).
    pub fn definitions(&self, id: usize) -> Rc<Vec<FnDef>> {
        self.assert_id(id);
        if let Some(defs) = self.definitions.borrow().get(&id) {
            return defs.clone();
        }
        self.ensure_parsed(&[id]);
        let defs = Rc::new(
            self.root(id)
                .map(|root| function::definitions(&root))
                .unwrap_or_default(),
        );
        self.definitions.borrow_mut().insert(id, defs.clone());
        defs
    }

    /// True when file `id` binds `name` at module level (assignment or class).
    pub fn module_has_name(&self, id: usize, name: &str) -> bool {
        self.ensure_parsed(&[id]);
        self.indexes(id)
            .is_some_and(|indexes| indexes.module.has_name(name))
    }

    pub fn imported_locals(&self, id: usize) -> Rc<Vec<String>> {
        self.assert_id(id);
        if let Some(l) = self.locals.borrow().get(&id) {
            return l.clone();
        }
        self.ensure_parsed(&[id]);
        let list = self
            .root(id)
            .map(|r| modules::imported_locals(&r, self.files[id].grammar.lang()))
            .unwrap_or_default();
        let rc = Rc::new(list);
        self.locals.borrow_mut().insert(id, rc.clone());
        rc
    }

    pub fn scope(&self, id: usize) -> Rc<Index<'a>> {
        self.assert_id(id);
        if let Some(s) = self.scopes.borrow().get(&id) {
            return s.clone();
        }
        self.ensure_parsed(&[id]);
        let imports = self.imports(id);
        let targets: Vec<usize> = imports.iter().map(|i| i.target).collect();
        self.ensure_parsed(&targets);
        let own = self.indexes(id);
        let mut merged = match &own {
            Some(indexes) => (*indexes.full).clone(),
            None => Index::default(),
        };
        // Own module-level names win; function locals never shadow an import file-wide.
        for imp in imports.iter() {
            if own
                .as_ref()
                .is_some_and(|indexes| !indexes.module.has_name(&imp.local))
            {
                merged.assigns.remove(&imp.local);
                merged.classes.remove(&imp.local);
            }
        }
        for imp in imports.iter() {
            if merged.has_name(&imp.local) {
                continue;
            }
            let Some(target) = self.indexes(imp.target).map(|indexes| indexes.module) else {
                continue;
            };
            match &imp.remote {
                Some(name) => {
                    if let Some(class) = target.classes.get(name) {
                        merged.insert_imported_class(
                            imp.local.clone(),
                            class.clone(),
                            target.clone(),
                        );
                    }
                    if let Some(value) = target.assigns.get(name) {
                        merged.insert_imported_assign(
                            imp.local.clone(),
                            value.clone(),
                            target.clone(),
                        );
                    }
                }
                None => {
                    for (name, class) in &target.classes {
                        merged.insert_imported_class(
                            format!("{}.{name}", imp.local),
                            class.clone(),
                            target.clone(),
                        );
                    }
                    for (name, value) in &target.assigns {
                        merged.insert_imported_assign(
                            format!("{}.{name}", imp.local),
                            value.clone(),
                            target.clone(),
                        );
                    }
                }
            }
        }
        merged.share_locals(self.locals_cache.clone());
        let rc = Rc::new(merged);
        self.scopes.borrow_mut().insert(id, rc.clone());
        rc
    }

    fn assert_id(&self, id: usize) {
        assert!(id < self.files.len(), "file id {id} out of range");
    }
}

#[cfg(test)]
mod tests;
