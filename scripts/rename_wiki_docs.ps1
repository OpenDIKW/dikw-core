# Docs-specific wiki -> knowledge rename. Excludes:
#   - CHANGELOG.md (historical entries preserved per refactor plan)
#   - scripts/* (this directory)
#   - Anything that would touch "wikilink" / "[[...]]" semantics
#
# Run from repo root.

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

$files = @(
    Get-ChildItem -Path "$RepoRoot\docs" -Recurse -Filter '*.md' -File |
        ForEach-Object { $_.FullName }
) + @(
    "$RepoRoot\CLAUDE.md",
    "$RepoRoot\CONTEXT.md",
    "$RepoRoot\GUIDE_FOR_AGENTS.md",
    "$RepoRoot\README.md",
    "$RepoRoot\evals\README.md",
    "$RepoRoot\evals\BASELINES.md",
    "$RepoRoot\.github\pull_request_template.md"
) | Where-Object { Test-Path $_ }

# Order matters — longer/more specific phrases first.
$pairs = @(
    # K-layer directory paths.
    @{ old = '`wiki/`'; new = '`knowledge/`' },
    @{ old = '``wiki/``'; new = '``knowledge/``' },
    @{ old = '`wiki/index.md`'; new = '`knowledge/index.md`' },
    @{ old = '`wiki/log.md`'; new = '`knowledge/log.md`' },
    @{ old = '``wiki/index.md``'; new = '``knowledge/index.md``' },
    @{ old = '``wiki/log.md``'; new = '``knowledge/log.md``' },
    # Paths in compound phrases.
    @{ old = 'wiki/index.md'; new = 'knowledge/index.md' },
    @{ old = 'wiki/log.md'; new = 'knowledge/log.md' },
    @{ old = 'wiki/entities/'; new = 'knowledge/entities/' },
    @{ old = 'wiki/concepts/'; new = 'knowledge/concepts/' },
    @{ old = 'wiki/notes/'; new = 'knowledge/notes/' },
    @{ old = 'wiki/<'; new = 'knowledge/<' },
    @{ old = '`wiki/'; new = '`knowledge/' },
    @{ old = '<base>/wiki/'; new = '<base>/knowledge/' },
    @{ old = 'trash/wiki/'; new = 'trash/knowledge/' },
    # On-disk + headings.
    @{ old = '# Wiki Index'; new = '# Knowledge Index' },
    @{ old = '# Wiki Log'; new = '# Knowledge Log' },
    @{ old = 'Wiki page'; new = 'Knowledge page' },
    @{ old = 'wiki pages'; new = 'knowledge pages' },
    @{ old = 'wiki page'; new = 'knowledge page' },
    @{ old = 'wiki tree'; new = 'knowledge tree' },
    @{ old = 'wiki layer'; new = 'knowledge layer' },
    @{ old = 'K-layer wiki'; new = 'K-layer knowledge' },
    # CLI / API field references that already shipped under new names.
    @{ old = 'WikiPage'; new = 'KnowledgePage' },
    @{ old = 'WikiLogEntry'; new = 'KnowledgeLogEntry' },
    @{ old = 'wiki_log'; new = 'knowledge_log' },
    @{ old = 'wiki_pages_changed'; new = 'knowledge_pages_changed' },
    @{ old = 'wiki_pages'; new = 'knowledge_pages' },
    @{ old = 'last_wiki_log_ts'; new = 'last_knowledge_log_ts' },
    @{ old = 'init_wiki'; new = 'init_base' },
    @{ old = 'load_wiki'; new = 'load_base' },
    @{ old = 'resolve_wiki_root'; new = 'resolve_base_root' },
    @{ old = 'wiki_root'; new = 'base_root' },
    @{ old = 'append_wiki_log'; new = 'append_knowledge_log' },
    @{ old = 'list_wiki_log'; new = 'list_knowledge_log' },
    @{ old = 'persist_wiki_page'; new = 'persist_knowledge_page' },
    @{ old = 'Layer.WIKI'; new = 'Layer.KNOWLEDGE' },
    @{ old = '`<wiki>'; new = '`<base>' },
    @{ old = '<wiki>'; new = '<base>' },
    # docstring -- /v1/base/pages/wiki/...
    @{ old = 'pages/wiki/'; new = 'pages/knowledge/' }
)

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$touched = 0
$total = 0
foreach ($file in $files) {
    $c = [System.IO.File]::ReadAllText($file, $utf8NoBom)
    $orig = $c
    foreach ($p in $pairs) {
        $c = $c.Replace($p.old, $p.new)
    }
    if ($c -ne $orig) {
        [System.IO.File]::WriteAllText($file, $c, $utf8NoBom)
        $touched += 1
        $rel = $file.Substring($RepoRoot.Length).TrimStart('\','/')
        Write-Output $rel
    }
}
Write-Output "---"
Write-Output "Touched $touched files (phase 6 docs)."
