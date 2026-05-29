# Client-side converter plugins — contract spec

This document is the **formal contract** between dikw-core and any
client-side converter plugin. It defines what plugins must implement,
how they're discovered, and how `dikw client import` dispatches to them.
Plugin authors looking for a tutorial-style walkthrough (with a working
example and pyproject snippets) should read the
[plugin-author-guide in the dikw-plugins repo](https://github.com/opendikw/dikw-plugins/blob/main/docs/plugin-author-guide.md).

## What a converter does

A converter takes one non-markdown file (`paper.pdf`, `book.epub`,
audio transcript, …) and writes a directory of md+assets that
`dikw client import` can package into the user's `<base>/sources/`.
Conversion runs **in-process inside `dikw client`** — the server never
loads converter dependencies.

## Protocol

Plugins must expose a class satisfying this Protocol:

```python
# from dikw_core.client.converters import Converter

class Converter(Protocol):
    name: str                    # engine label, e.g. "marker", "mineru"
    extensions: tuple[str, ...]  # claimed file suffixes, e.g. (".pdf",)
    def convert(self, input_path: Path, output_dir: Path) -> None: ...
```

- `name` must be a non-empty string, globally unique across plugins
  the user has installed (the dispatch uses it for disambiguation).
- `extensions` must be a non-empty tuple of strings starting with `.`,
  e.g. `(".pdf",)` or `(".epub", ".azw3")`.
- `convert` writes md+assets into `output_dir` and returns `None`.
  Raising any exception is treated as a conversion failure and surfaces
  to the user as `SourceImportError`.

## Discovery

dikw-core loads plugins via the
`dikw.client.converters` entry-points group. A plugin's `pyproject.toml`
declares one entry-point per Converter class:

```toml
[project.entry-points."dikw.client.converters"]
marker = "dikw_converter_pdf:MarkerConverter"
```

The entry-point name (`marker` here) is informational — `Converter.name`
on the loaded class is what dispatch uses.

Discovery is **lazy**: `dikw client` only loads plugins when an actual
non-md file triggers dispatch. Common commands (`status`, `retrieve`,
markdown-only `import`) never pay plugin import cost.

## Selection priority

When multiple plugins claim the same extension, `dikw client` picks one
in this order:

1. `--converter=<name>` CLI flag — one-shot override per invocation.
2. `DIKW_CLIENT_CONVERTER_<EXT>` environment variable, e.g.
   `DIKW_CLIENT_CONVERTER_PDF=marker`.
3. `client.toml` `[default.converters]` entry for the extension:

   ```toml
   # ~/.config/dikw/client.toml  (Windows: %APPDATA%\dikw\client.toml)
   [default.converters]
   ".pdf" = "marker"
   ".epub" = "ebook2md"
   ```

4. If exactly one plugin claims the extension, use it (zero config).
5. Otherwise raise `ConverterError` listing the installed engines and
   the remediation paths.

## Output layout

The convention `dikw client import paper.pdf` expects from a plugin:

```
<output_dir>/
├── <stem>.md            # the converted markdown
└── assets/
    ├── <stem>.<orig>    # original input copied as provenance asset
    ├── figure-1.png     # extracted images
    └── …
```

After conversion, the importer packages this exactly like a
user-authored md tree. On the server it lands as
`<base>/sources/<stem>/<stem>.md` + `<base>/sources/<stem>/assets/*`.

Multi-file output (e.g. one PDF → multiple chapter markdowns) is
allowed — each `*.md` in the output becomes its own package.

## Asset reference rule

Every asset a plugin writes **must be referenced from the generated
markdown** using image syntax — `![alt](assets/foo.png)` or the
Obsidian `![[assets/foo.pdf]]` form. The dikw-core importer extracts
asset references with `md_inspect.extract_image_refs`, which only
recognises image-style refs; a regular markdown link
`[label](assets/foo.pdf)` will **not** include the asset.

For provenance (the original `paper.pdf` alongside the converted
`paper.md`), the convention is an explicit image ref even though the
file isn't an image:

```markdown
# Paper Title

…converted prose…

![original](assets/paper.pdf)
```

Plugins that emit unreferenced files in `assets/` either silently drop
them (if the extension isn't in dikw-core's
`_DEFAULT_ASSET_EXTENSIONS`) or trip the orphan-asset check (if it is).
Always reference what you write.

## Determinism

`convert()` should be deterministic for the same input bytes — the
same PDF in, the same md+assets out. dikw-core's ingest hashes the md
and skips unchanged sources; a converter that embeds timestamps, random
seeds, or run IDs into its output defeats that optimisation and forces
a re-embed on every import.

Where the underlying tool is genuinely non-deterministic (some OCR
pipelines, some LLM-assisted converters), the plugin can either:

- Pin the upstream tool's seed / temperature where exposed, or
- Document the non-determinism so users know re-imports will trigger
  re-embed.

## Failure modes

| Plugin behavior                          | dikw client surfaces                                    |
| ---------------------------------------- | ------------------------------------------------------- |
| Raises any exception during `convert()`  | `SourceImportError(f"converter {name!r} failed: {e}")`  |
| Returns without writing any files        | `SourceImportError("converter produced no output")`     |
| Writes md+assets that fail md_inspect    | Normal pre-flight error chain (frontmatter, asset_missing) |
| Missing `name` / `extensions` attribute  | `ConverterError` at `discover()` time, before dispatch  |

## Staging + cleanup

dikw-core creates a `tempfile.mkdtemp(prefix="dikw-import-")` directory
and passes `<staging>/<stem>` as `output_dir` to the plugin. The
staging directory is cleaned up via `shutil.rmtree` whether the import
succeeds or fails. Plugins should not assume `output_dir` persists
beyond the `convert()` call.

## Versioning

The Protocol shape, entry-points group name, and selection rules are
**public API** of dikw-core. Breaking changes go through a
deprecation cycle and a CHANGELOG entry. The contract module is
[`src/dikw_core/client/converters.py`](../src/dikw_core/client/converters.py)
— the source of truth.
