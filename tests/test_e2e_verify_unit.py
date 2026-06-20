"""Unit tests for the pure logic in ``tools/e2e_verify.py``.

The subprocess-driven lifecycles are covered by the ``-m slow`` wrappers
(test_e2e_verify_local.py / _docker.py). Here we pin the deterministic core:
the CLI-coverage guard (the anti-drift mechanism), .env parsing, secret
redaction, key posture, and the provider dikw.yml builder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.e2e_verify import (  # noqa: E402
    CoverageManifest,
    LegResult,
    Redactor,
    Status,
    _harness_compose,
    _required_key_envs,
    assert_full_cli_coverage,
    build_coverage_manifest,
    build_provider_yaml,
    client_leaf_verbs,
    load_dotenv,
    resolve_posture,
)


# --- coverage manifest ---------------------------------------------------- #
def test_client_leaf_verbs_covers_the_known_surface() -> None:
    verbs = client_leaf_verbs()
    # A representative spread across groups + the invokable ``lint`` group root.
    for expected in (
        "info", "status", "health", "check", "retrieve",
        "pages list", "pages get", "pages links", "pages provenance",
        "assets get", "graph get", "import", "ingest", "synth", "eval",
        "lint", "lint propose", "lint proposals", "lint apply",
        "tasks list", "tasks status", "tasks events", "tasks wait", "tasks cancel",
        "wisdom write", "delete", "serve-and-run",
    ):
        assert expected in verbs, f"{expected!r} missing from live client verb tree"


def _legs_for(verbs: set[str]) -> list[LegResult]:
    return [LegResult(v, Status.PASS, verb=v) for v in verbs]


def test_coverage_passes_when_all_required_verbs_executed() -> None:
    manifest = build_coverage_manifest()
    required = manifest.live_leaves - set(manifest.skip)
    leg = assert_full_cli_coverage(manifest, _legs_for(required))
    assert leg.status is Status.PASS, leg.detail


def test_coverage_fails_and_names_an_uncovered_verb() -> None:
    """The anti-drift guarantee: a new (or simply uncovered) verb makes the
    coverage leg RED and names it."""
    manifest = build_coverage_manifest()
    required = manifest.live_leaves - set(manifest.skip)
    dropped = "retrieve"
    leg = assert_full_cli_coverage(manifest, _legs_for(required - {dropped}))
    assert leg.status is Status.FAIL
    assert dropped in leg.detail


def test_coverage_counts_loud_skip_as_covered() -> None:
    manifest = build_coverage_manifest()
    required = manifest.live_leaves - set(manifest.skip)
    legs = [LegResult(v, Status.PASS, verb=v) for v in required if v != "synth"]
    legs.append(LegResult("synth", Status.SKIP, "tier-2: keys absent", verb="synth"))
    leg = assert_full_cli_coverage(manifest, legs)
    assert leg.status is Status.PASS, leg.detail


def test_coverage_fails_on_stale_skip_entry() -> None:
    manifest = CoverageManifest(live_leaves={"info", "status"},
                                skip={"no-such-verb": "gone"})
    leg = assert_full_cli_coverage(manifest, _legs_for({"info", "status"}))
    assert leg.status is Status.FAIL
    assert "no-such-verb" in leg.detail


# --- .env parsing --------------------------------------------------------- #
def test_load_dotenv_parses_and_env_wins(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text(
        "# a comment\n"
        "\n"
        'MINIMAX_API_KEY="sk-minimax-123"\n'
        "GITEE_API_KEY=gitee-456\n"
        "ALREADY_SET=fromfile\n"
        "MALFORMED_NO_EQUALS\n",
        encoding="utf-8")
    env = {"ALREADY_SET": "fromenv"}
    load_dotenv(f, env)
    assert env["MINIMAX_API_KEY"] == "sk-minimax-123"  # quotes stripped
    assert env["GITEE_API_KEY"] == "gitee-456"
    assert env["ALREADY_SET"] == "fromenv"  # pre-set env wins over file
    assert "MALFORMED_NO_EQUALS" not in env


def test_load_dotenv_noop_when_missing(tmp_path: Path) -> None:
    env = {"X": "1"}
    load_dotenv(tmp_path / "nope.env", env)
    assert env == {"X": "1"}


# --- redaction ------------------------------------------------------------ #
def test_redactor_masks_secrets_longest_first() -> None:
    # Overlapping secrets: the longer fully contains the shorter. Longest-first
    # ordering must mask the long one whole, leaving no fragment behind.
    redact = Redactor(["sk-abcdef-long", "sk-abc"])
    out = redact("llm=sk-abcdef-long embed=sk-abc")
    assert "sk-abcdef-long" not in out and "sk-abc" not in out
    assert "def-long" not in out  # the long secret wasn't half-masked
    assert out.count("***") == 2


def test_redactor_ignores_empty() -> None:
    redact = Redactor([])
    assert redact("nothing to hide") == "nothing to hide"


# --- key posture ---------------------------------------------------------- #
def test_resolve_posture() -> None:
    required = ["MINIMAX_API_KEY", "GITEE_API_KEY"]
    full = resolve_posture({"MINIMAX_API_KEY": "a", "GITEE_API_KEY": "b"}, required)
    assert full.has_keys and not full.missing
    none = resolve_posture({}, required)
    assert not none.has_keys
    assert "MINIMAX_API_KEY" in none.missing and "GITEE_API_KEY" in none.missing


def test_required_key_envs_reads_profile_config() -> None:
    """The real-leg gate keys off the profile's vendor-canonical key vars,
    not a hardcoded ANTHROPIC_API_KEY/DIKW_EMBEDDING_API_KEY pair."""
    profile = (
        Path(__file__).parent / "fixtures" / "live-minimax-gitee.dikw.yml"
    )
    assert _required_key_envs(profile) == ["MINIMAX_API_KEY", "GITEE_API_KEY"]


# --- provider dikw.yml builder ------------------------------------------- #
def test_build_provider_yaml_local_is_sqlite_no_secrets() -> None:
    out = build_provider_yaml(mode="local", observe=False, env={})
    doc = yaml.safe_load(out)
    assert doc["storage"]["backend"] == "sqlite"
    assert doc["provider"]["llm"] == "anthropic_compat"
    assert doc["provider"]["embedding_model"] == "Qwen3-Embedding-0.6B"
    assert "telemetry" not in doc or not doc["telemetry"]["enabled"]
    # The provider block names key env vars (e.g. MINIMAX_API_KEY) — those are
    # variable NAMES, not values. What must never serialize is a secret VALUE.
    assert doc["provider"]["llm_api_key_env"] == "MINIMAX_API_KEY"
    assert doc["provider"]["embedding_api_key_env"] == "GITEE_API_KEY"
    assert "sk-" not in out


def test_build_provider_yaml_docker_is_postgres() -> None:
    out = build_provider_yaml(mode="docker", observe=False, env={})
    doc = yaml.safe_load(out)
    assert doc["storage"]["backend"] == "postgres"
    assert "postgres:5432" in doc["storage"]["dsn"]


def test_build_provider_yaml_observe_wires_telemetry() -> None:
    local = yaml.safe_load(build_provider_yaml(mode="local", observe=True, env={}))
    assert local["telemetry"]["enabled"] is True
    assert local["telemetry"]["endpoint"] == "http://localhost:4318"
    docker = yaml.safe_load(build_provider_yaml(mode="docker", observe=True, env={}))
    assert docker["telemetry"]["endpoint"] == "http://otel-collector:4318"


# --- docker compose generator ------------------------------------------- #
def test_harness_compose_runs_as_host_uid_with_writable_base() -> None:
    """The server must write the ``./base`` bind mount as the host user, else a
    native-Linux UID mismatch makes import/synth/wisdom fail."""
    doc = yaml.safe_load(
        _harness_compose(12345, observe=False, provider_key_envs=["MINIMAX_API_KEY", "GITEE_API_KEY"])
    )
    svc = doc["services"]["dikw-core-local"]
    assert svc["user"] == "${DIKW_E2E_UID:-1000}:${DIKW_E2E_GID:-1000}"
    assert svc["environment"]["HOME"] == "/tmp"
    # The profile's vendor-canonical key vars are passed through to the container.
    assert svc["environment"]["MINIMAX_API_KEY"] == "${MINIMAX_API_KEY:-}"
    assert svc["environment"]["GITEE_API_KEY"] == "${GITEE_API_KEY:-}"


def test_harness_compose_observe_joins_obs_network() -> None:
    """--observe must put the server on the observability stack's network so it
    can resolve ``otel-collector``; without --observe the file stays single-network."""
    plain = _harness_compose(12345, observe=False, provider_key_envs=["MINIMAX_API_KEY"])
    assert "external" not in plain and "dikw-e2e-obs_default" not in plain

    doc = yaml.safe_load(
        _harness_compose(12345, observe=True, provider_key_envs=["MINIMAX_API_KEY"])
    )
    assert doc["networks"]["obs"]["external"] is True
    assert doc["networks"]["obs"]["name"] == "dikw-e2e-obs_default"
    assert doc["services"]["dikw-core-local"]["networks"] == ["default", "obs"]


def test_build_provider_yaml_env_overrides() -> None:
    env = {
        "DIKW_E2E_LLM": "openai_compat",  # exercises the protocol-switch override line
        "DIKW_E2E_LLM_MODEL": "MiniMax-M3",
        "DIKW_E2E_LLM_BASE_URL": "https://example.test/v1",
        "DIKW_E2E_EMBEDDING_DIM": "512",
        "DIKW_E2E_EMBEDDING_BATCH": "8",
    }
    doc = yaml.safe_load(build_provider_yaml(mode="local", observe=False, env=env))
    assert doc["provider"]["llm"] == "openai_compat"
    assert doc["provider"]["llm_model"] == "MiniMax-M3"
    assert doc["provider"]["llm_base_url"] == "https://example.test/v1"
    assert doc["provider"]["embedding_dim"] == 512
    assert doc["provider"]["embedding_batch_size"] == 8
