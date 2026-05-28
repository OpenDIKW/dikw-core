# Bulk literal-string replacement helper for wiki -> knowledge refactor.
#
# Reads each path as UTF-8, applies the rename pairs as literal substring
# replacements (NOT regex), and writes back as UTF-8 without BOM.
#
# Usage:
#   .\scripts\rename_wiki.ps1 -CommitPhase 1
#
# Phases follow the plan in .claude/plans/dikw-core-wiki-velvet-flamingo.md.
# Each phase touches only the renames belonging to that commit boundary.

param(
    [Parameter(Mandatory=$true)]
    [int]$CommitPhase,
    [string[]]$Files = @()
)

$ErrorActionPreference = 'Stop'

# Resolve repo root (script lives in scripts/)
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

function Get-DefaultFiles {
    param([int]$Phase)
    $py = @(
        Get-ChildItem -Path "$RepoRoot\src\dikw_core" -Recurse -Filter '*.py' -File |
            Where-Object { $_.FullName -notmatch '__pycache__' } |
            ForEach-Object { $_.FullName }
    ) + @(
        Get-ChildItem -Path "$RepoRoot\tests" -Recurse -Filter '*.py' -File |
            Where-Object { $_.FullName -notmatch '__pycache__' } |
            ForEach-Object { $_.FullName }
    )
    $py3 = $py + @(
        Get-ChildItem -Path "$RepoRoot\src\dikw_core\storage\migrations" -Recurse -Filter '*.sql' -File |
            ForEach-Object { $_.FullName }
    ) + @(
        Get-ChildItem -Path "$RepoRoot\src\dikw_core\prompts" -Recurse -Filter '*.md' -File |
            ForEach-Object { $_.FullName }
    ) + @(
        Get-ChildItem -Path "$RepoRoot\evals" -Recurse -Filter '*.py' -File |
            Where-Object { $_.FullName -notmatch '__pycache__' } |
            ForEach-Object { $_.FullName }
    )
    switch ($Phase) {
        1 { return $py }
        2 { return $py }
        3 { return $py3 }
        default { throw "Unknown phase: $Phase" }
    }
}

function Get-Pairs {
    param([int]$Phase)
    switch ($Phase) {
        1 {
            # Order matters: longer/more-specific patterns must come before
            # their substrings to avoid double-replacement.
            return @(
                @{ old = 'WIKI_INIT_FILES'; new = 'KNOWLEDGE_INIT_FILES' },
                @{ old = 'WikiLogEntry'; new = 'KnowledgeLogEntry' },
                @{ old = 'WikiPageMeta'; new = 'KnowledgePageMeta' },
                @{ old = 'WikiPage'; new = 'KnowledgePage' },
                @{ old = 'persist_wiki_page'; new = 'persist_knowledge_page' },
                @{ old = 'domains.knowledge.wiki'; new = 'domains.knowledge.page' },
                @{ old = 'domains/knowledge/wiki.py'; new = 'domains/knowledge/page.py' },
                @{ old = 'Layer.WIKI'; new = 'Layer.KNOWLEDGE' }
            )
        }
        3 {
            # Commit 3: on-disk wiki/ -> knowledge/ + SQL wiki_log -> knowledge_log
            # + remaining K-layer identifiers that were not pure class/function
            # renames (wiki_pages field, last_wiki_log_ts, etc).
            return @(
                # SynthReport / progress / storage counts field names.
                @{ old = 'last_wiki_log_ts'; new = 'last_knowledge_log_ts' },
                @{ old = 'wiki_pages_changed'; new = 'knowledge_pages_changed' },
                @{ old = 'wiki_pages'; new = 'knowledge_pages' },
                # Storage protocol method names.
                @{ old = 'append_wiki_log'; new = 'append_knowledge_log' },
                @{ old = 'list_wiki_log'; new = 'list_knowledge_log' },
                # SQL table + index name.
                @{ old = 'wiki_log_ts'; new = 'knowledge_log_ts' },
                @{ old = 'wiki_log'; new = 'knowledge_log' },
                # On-disk path string literals — every place that hardcodes the
                # K-layer directory.
                @{ old = '"wiki/index.md"'; new = '"knowledge/index.md"' },
                @{ old = '"wiki/log.md"'; new = '"knowledge/log.md"' },
                @{ old = '"wiki/entities/.gitkeep"'; new = '"knowledge/entities/.gitkeep"' },
                @{ old = '"wiki/concepts/.gitkeep"'; new = '"knowledge/concepts/.gitkeep"' },
                @{ old = '"wiki/notes/.gitkeep"'; new = '"knowledge/notes/.gitkeep"' },
                @{ old = "'wiki/index.md'"; new = "'knowledge/index.md'" },
                @{ old = "'wiki/log.md'"; new = "'knowledge/log.md'" },
                # Compound Path joins (root / "wiki", base_root / "wiki", etc).
                @{ old = '/ "wiki"'; new = '/ "knowledge"' },
                @{ old = "/ 'wiki'"; new = "/ 'knowledge'" },
                # Page-default path template (page.py default_page_path).
                @{ old = 'f"wiki/{type_to_folder('; new = 'f"knowledge/{type_to_folder(' },
                # parts[0] == "wiki" — used by type_from_path.
                @{ old = 'parts[0] == "wiki"'; new = 'parts[0] == "knowledge"' },
                # Prompts: <page path="wiki/<folder>/<slug>.md" ...>
                @{ old = 'path="wiki/<folder>/<slug>.md"'; new = 'path="knowledge/<folder>/<slug>.md"' },
                # Local variable names (wiki_dir, etc.) — note the leading
                # boundary is enforced by adjacent characters; "wiki_dir" is
                # not a substring of another identifier.
                @{ old = 'wiki_dir'; new = 'knowledge_dir' },
                # Trash subpath used in lint_fix soft-delete.
                @{ old = 'trash/wiki/'; new = 'trash/knowledge/' },
                @{ old = 'trash/wiki'; new = 'trash/knowledge' },
                # Heading literals.
                @{ old = '# Wiki Index'; new = '# Knowledge Index' },
                @{ old = '# Wiki Log'; new = '# Knowledge Log' },
                # Common phrases. Order matters: "wiki pages" then "wiki page"
                # so the longer phrase wins first.
                @{ old = 'wiki pages'; new = 'knowledge pages' },
                @{ old = 'wiki page'; new = 'knowledge page' },
                @{ old = 'wiki tree'; new = 'knowledge tree' },
                @{ old = 'Wiki pages'; new = 'Knowledge pages' },
                @{ old = 'Wiki page'; new = 'Knowledge page' },
                @{ old = 'wiki layer'; new = 'knowledge layer' },
                @{ old = 'K-layer wiki'; new = 'K-layer knowledge' },
                @{ old = 'list of D-layer paths this knowledge page summarises.'; new = 'list of D-layer paths this knowledge page summarises.' }
            )
        }
        2 {
            # Commit 2: base-scope renames (NOT K-layer semantics).
            # ``wiki_root`` etc. actually point at the base root containing
            # wiki/, wisdom/, data/, .dikw/ — rename to base_root.
            return @(
                @{ old = 'DIKW_WIKI_INSTANCE_ID'; new = 'DIKW_BASE_INSTANCE_ID' },
                @{ old = '_WIKI_ID_FILENAME'; new = '_BASE_ID_FILENAME' },
                @{ old = '_wiki_scope_id'; new = '_base_scope_id' },
                @{ old = 'resolve_wiki_root'; new = 'resolve_base_root' },
                @{ old = 'init_test_wiki'; new = 'init_test_base' },
                @{ old = 'init_wiki'; new = 'init_base' },
                @{ old = 'load_wiki'; new = 'load_base' },
                @{ old = 'wiki_root'; new = 'base_root' },
                # Identifier name for the persistent id file (separate from
                # the literal filename string, which is handled in commit 3
                # if needed). The bare ``wiki_id`` symbol shows up in tests
                # / runtime code as a variable.
                @{ old = '"wiki_id"'; new = '"base_id"' },
                # Pytest fixture names. Keep ``client_wiki`` and
                # ``tmp_wiki`` consistent across conftest and consumers.
                @{ old = 'client_wiki'; new = 'client_base' },
                @{ old = 'tmp_wiki'; new = 'tmp_base' }
            )
        }
        default { throw "Unknown phase: $Phase" }
    }
}

if ($Files.Count -eq 0) {
    $Files = Get-DefaultFiles -Phase $CommitPhase
}
$Pairs = Get-Pairs -Phase $CommitPhase

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$touched = 0
$totalReplacements = 0

foreach ($file in $Files) {
    if (-not (Test-Path $file)) { continue }
    $original = [System.IO.File]::ReadAllText($file, $utf8NoBom)
    $content = $original
    $fileReplacements = 0
    foreach ($pair in $Pairs) {
        $count = ([regex]::Matches($content, [regex]::Escape($pair.old))).Count
        if ($count -gt 0) {
            $content = $content.Replace($pair.old, $pair.new)
            $fileReplacements += $count
        }
    }
    if ($content -ne $original) {
        [System.IO.File]::WriteAllText($file, $content, $utf8NoBom)
        $touched += 1
        $totalReplacements += $fileReplacements
        $rel = $file.Substring($RepoRoot.Length).TrimStart('\','/')
        Write-Output "$rel ($fileReplacements)"
    }
}

Write-Output "---"
Write-Output "Touched $touched files, $totalReplacements total replacements (phase $CommitPhase)."
