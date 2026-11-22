#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Adapted from OpenSDLC managed-agent-platform; modified for OpenDIKW issue forms and rollout.
"""Exercise the issue workflow with a persistent in-memory GitHub timeline."""

import copy
import json
import subprocess
import unittest
from itertools import product
from pathlib import Path
from unittest.mock import patch

from issue_labels import reconcile, sweep

REPO = "owner/repo"
FORM = "### Priority\n\np0\n\n### Difficulty\n\neasy\n\n### What happened?\n\nSteps to reproduce."
BOT = {"__typename": "Bot", "login": "github-actions"}
HUMAN = {"__typename": "User", "login": "maintainer"}
ROLLOUT = "2024-09-01T00:00:00Z"
OLD = "2024-08-31T23:59:59Z"


def connection(nodes, cursor=None):
    return {"nodes": nodes, "pageInfo": {"hasNextPage": cursor is not None, "endCursor": cursor}}


class GitHub:
    def __init__(self, labels=(), kind="Task", body=None, number=23, state="OPEN", created=ROLLOUT):
        self.labels = set(labels)
        self.kind = kind
        self.body = body
        self.number = number
        self.state = state
        self.created = created
        self.repo = REPO
        self.is_pr = False
        self.events = []
        self.event_time = ROLLOUT
        self.history_since = None
        self.writes = []
        self.before_write = None
        self.lose_reply = False
        self.reject_type = False
        self.per_page = 100

    def label(self, name, applied, actor):
        if applied:
            self.labels.add(name)
        else:
            self.labels.discard(name)
        self.events.append(
            {
                "__typename": "LabeledEvent" if applied else "UnlabeledEvent",
                "actor": actor,
                "label": {"name": name},
                "createdAt": self.event_time,
            }
        )

    def set_type(self, value, actor):
        if self.kind == value:
            return
        previous = self.kind
        self.kind = value
        event = {
            "actor": actor,
            "issueType": {"name": value or previous},
            "createdAt": self.event_time,
        }
        if value is None:
            event["__typename"] = "IssueTypeRemovedEvent"
        elif previous is None:
            event["__typename"] = "IssueTypeAddedEvent"
        else:
            event.update(__typename="IssueTypeChangedEvent", prevIssueType={"name": previous})
        self.events.append(event)

    def history(self, cursor=None):
        offset = int(cursor or 0)
        end = offset + self.per_page
        events = [
            event
            for event in self.events
            if not self.history_since or event["createdAt"] >= self.history_since
        ]
        return connection(
            copy.deepcopy(events[offset:end]), str(end) if end < len(events) else None
        )

    def snapshot(self):
        return {
            "id": "I" + str(self.number),
            "number": self.number,
            "repository": {"nameWithOwner": self.repo},
            "state": self.state,
            "createdAt": self.created,
            "issueType": {"name": self.kind} if self.kind else None,
            "body": self.body,
            "labels": connection(
                [{"name": name, "id": "id-" + name} for name in sorted(self.labels)]
            ),
            "timelineItems": self.history(),
        }

    def rest(self):
        return {
            "number": self.number,
            "node_id": "I" + str(self.number),
            "state": self.state.lower(),
            "repository_url": "https://api.github.com/repos/" + self.repo,
            "created_at": self.created,
            "type": {"name": self.kind} if self.kind else None,
            "body": self.body,
            "labels": [{"name": n, "node_id": "id-" + n} for n in self.labels],
        }

    def __call__(self, endpoint, *args, payload=None):
        path = f"repos/{REPO}/issues/{self.number}"
        if endpoint == "graphql" and "mutation" not in payload["query"]:
            variables = payload["variables"]
            self.history_since = (
                variables.get("rollout") if "since: $rollout" in payload["query"] else None
            )
            if variables.get("historyCursor"):
                return {
                    "data": {"node": {"timelineItems": self.history(variables["historyCursor"])}}
                }
            return {
                "data": {
                    "repository": {"issueOrPullRequest": {} if self.is_pr else self.snapshot()}
                }
            }
        if self.before_write:
            hook, self.before_write = self.before_write, None
            hook()
        if endpoint == path and args == ("--method", "PATCH"):
            self.writes.append(("type", payload["type"]))
            if not self.reject_type:
                self.set_type(payload["type"], BOT)
            result = self.rest()
        elif endpoint == path + "/labels":
            for name in payload["labels"]:
                self.writes.append(("add", name))
                self.label(name, True, BOT)
            result = self.rest()["labels"]
        else:
            assert endpoint == "graphql" and "removeLabelsFromLabelable" in payload["query"], (
                endpoint
            )
            for label_id in payload["variables"]["input"]["labelIds"]:
                name = label_id.removeprefix("id-")
                self.writes.append(("remove", name))
                self.label(name, False, BOT)
            result = {"data": {"removeLabelsFromLabelable": {"clientMutationId": None}}}
        if self.lose_reply:
            self.lose_reply = False
            raise RuntimeError("write committed but reply lost")
        return result


class IssueLabelsTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("issue_labels.ROLLOUT", ROLLOUT))

    def run_issue(self, gh):
        with patch("issue_labels.api", side_effect=gh):
            reconcile(REPO, gh.number)

    def test_valid_type_recovers_displaced_human_choice_on_a_fresh_run(self):
        gh = GitHub(["p2", "easy"], kind=None)
        gh.set_type("Bug", HUMAN)
        gh.set_type("Feature", BOT)  # Committed before the previous runner died.
        self.run_issue(gh)
        self.assertEqual(gh.kind, "Bug")
        self.run_issue(gh)
        self.assertEqual(gh.writes, [("type", "Bug")])
        gh.set_type("Feature", HUMAN)
        self.run_issue(gh)
        self.assertEqual(gh.writes, [("type", "Bug")])

    def test_valid_label_recovers_reselected_human_winner_on_a_fresh_run(self):
        gh = GitHub(["easy"])
        gh.label("p0", True, HUMAN)
        gh.label("p2", True, HUMAN)
        gh.label("p0", False, HUMAN)
        gh.label("p0", True, HUMAN)  # New generation after the bot read p0 + p2.
        gh.label("p0", False, BOT)  # Committed before the previous runner died.
        self.run_issue(gh)
        self.assertEqual(gh.labels, {"p0", "easy"})
        self.run_issue(gh)
        self.assertEqual(gh.writes, [("add", "p0"), ("remove", "p2")])

    def test_first_sweep_respects_choices_recorded_before_activation(self):
        gh = GitHub(["easy"], kind="Bug", created=OLD)
        gh.event_time = OLD
        gh.label("p1", True, HUMAN)
        gh.label("p0", True, HUMAN)
        gh.set_type("Task", BOT)  # An existing workflow predates this reconciler.
        gh.event_time = ROLLOUT
        gh.per_page = 1
        self.run_issue(gh)
        self.assertEqual((gh.kind, gh.labels), ("Task", {"p0", "easy"}))

    def test_queued_event_skips_a_deleted_issue_and_reports_other_errors(self):
        target_path = ["repository", "issueOrPullRequest"]
        for error_type, path in (
            ("NOT_FOUND", target_path),
            ("FORBIDDEN", target_path),
            ("NOT_FOUND", ["repository"]),
        ):
            with self.subTest(error_type=error_type, path=path):
                response = {
                    "data": {"repository": {"issueOrPullRequest": None}},
                    "errors": [{"type": error_type, "path": path}],
                }
                failure = subprocess.CalledProcessError(
                    1, ["gh", "api"], output=json.dumps(response)
                )
                with patch("issue_labels.api", side_effect=failure):
                    if error_type == "NOT_FOUND" and path == target_path:
                        reconcile(REPO, 1)
                    else:
                        with self.assertRaises(subprocess.CalledProcessError):
                            reconcile(REPO, 1)

    def test_stale_repair_cannot_restore_a_type_after_a_maintainer_clears_it(self):
        for body, fallback in ((None, "Feature"), (FORM, "Bug")):
            with self.subTest(fallback=fallback):
                gh = GitHub(["p2", "easy"], kind=None, body=body)
                gh.set_type("Task", HUMAN)
                gh.set_type("Feature", BOT)
                gh.before_write = lambda issue=gh: issue.set_type(None, HUMAN)
                self.run_issue(gh)
                self.run_issue(gh)
                self.assertEqual(gh.kind, fallback)

    def test_stale_repair_cannot_restore_a_withdrawn_priority(self):
        gh = GitHub(["easy"])
        gh.label("p0", True, HUMAN)
        gh.label("p0", False, BOT)

        def withdraw():
            gh.label("p0", True, HUMAN)
            gh.label("p0", False, HUMAN)

        gh.before_write = withdraw
        self.run_issue(gh)
        self.run_issue(gh)
        self.assertEqual(gh.labels, {"p1", "easy"})

    def test_interleaved_type_write_and_lost_reply(self):
        gh = GitHub(["p2", "easy"], kind=None)
        gh.before_write = lambda: gh.set_type("Task", HUMAN)
        gh.lose_reply = True
        with self.assertRaisesRegex(RuntimeError, "reply lost"):
            self.run_issue(gh)
        self.assertEqual(gh.kind, "Feature")
        self.run_issue(gh)
        self.assertEqual(gh.kind, "Task")

    def test_cleanup_withdrawn_winner_restores_the_other_human_choice(self):
        gh = GitHub(["medium"])
        gh.label("p2", True, HUMAN)
        gh.label("p0", True, HUMAN)
        gh.before_write = lambda: gh.label("p0", False, HUMAN)
        self.run_issue(gh)
        self.assertEqual(gh.labels, {"p2", "medium"})

    def test_reselected_loser_between_read_and_remove_survives_a_lost_reply(self):
        gh = GitHub(["hard"])
        gh.label("p0", True, HUMAN)
        gh.label("p2", True, HUMAN)

        def reselect():
            gh.label("p0", False, HUMAN)
            gh.label("p0", True, HUMAN)

        gh.before_write = reselect
        gh.lose_reply = True
        with self.assertRaisesRegex(RuntimeError, "reply lost"):
            self.run_issue(gh)
        self.assertEqual(gh.labels, {"p2", "hard"})
        self.run_issue(gh)
        self.assertEqual(gh.labels, {"p0", "hard"})

    def test_legitimate_cleanup_and_manual_clear_do_not_resurrect_invalidated_choices(self):
        gh = GitHub(["medium"], kind=None, body=FORM)
        gh.set_type("Task", HUMAN)
        gh.set_type(None, HUMAN)
        gh.label("p2", True, HUMAN)
        gh.label("p1", True, HUMAN)
        gh.label("p2", False, BOT)  # Removed the loser while the human winner remained.
        gh.label("p1", False, HUMAN)
        self.run_issue(gh)
        self.assertEqual((gh.kind, gh.labels), ("Bug", {"p0", "medium"}))
        self.run_issue(gh)
        self.assertEqual(gh.writes, [("type", "Bug"), ("add", "p0")])

    def test_replacement_and_bot_fallback_preserve_the_latest_human_selection(self):
        gh = GitHub(["medium", "legacy, label"])
        gh.label("p0", True, HUMAN)
        gh.label("p2", True, HUMAN)
        gh.label("p1", True, BOT)
        gh.per_page = 1
        self.run_issue(gh)
        self.assertEqual(gh.labels, {"medium", "p2"})
        self.assertCountEqual(
            gh.writes, [("remove", "legacy, label"), ("remove", "p1"), ("remove", "p0")]
        )

    def test_native_type_and_each_label_dimension_are_independent(self):
        cases = [
            (None, [], None, "Feature", {"p1", "hard"}),
            ("Bug", ["p0"], None, "Bug", {"p0", "hard"}),
            (None, ["enhancement", "easy"], FORM, "Feature", {"p0", "easy"}),
            ("Task", ["bug", "p2"], FORM, "Task", {"p2", "easy"}),
            (None, ["bug", "medium"], None, "Bug", {"p1", "medium"}),
            (None, ["feature"], FORM, "Feature", {"p0", "easy"}),
            (None, [], FORM.replace("What happened?", "Task details"), "Task", {"p0", "easy"}),
            (None, [], FORM.replace("p0", "invalid"), "Bug", {"p1", "easy"}),
            (None, [], "Example:\n" + FORM, "Feature", {"p1", "hard"}),
            (None, [], "### What happened?\n\n" + FORM, "Feature", {"p1", "hard"}),
            ("Bug", ["p2", "medium"], FORM.replace("\n", "\r\n"), "Bug", {"p2", "medium"}),
        ]
        for kind, labels, body, expected_type, expected_labels in cases:
            with self.subTest(kind=kind, labels=labels, body=body):
                gh = GitHub(labels, kind=kind, body=body)
                self.run_issue(gh)
                self.assertEqual((gh.kind, gh.labels), (expected_type, expected_labels))

    def test_maintainer_replaces_default_difficulty_without_changing_priority(self):
        gh = GitHub(["p0"], kind="Bug")
        self.run_issue(gh)
        gh.label("medium", True, HUMAN)
        self.run_issue(gh)
        self.assertEqual((gh.kind, gh.labels), ("Bug", {"p0", "medium"}))
        writes = list(gh.writes)
        self.run_issue(gh)
        self.assertEqual(gh.writes, writes)

    def test_rejected_type_does_not_remove_migration_evidence(self):
        gh = GitHub(["bug", "p2", "easy"], kind=None)
        gh.reject_type = True
        with self.assertRaisesRegex(RuntimeError, "type"):
            self.run_issue(gh)
        self.assertEqual(gh.labels, {"bug", "p2", "easy"})

    def test_only_eligible_issues_are_written(self):
        gh = GitHub(kind=None, state="CLOSED", created=OLD)
        self.run_issue(gh)
        gh.created, gh.is_pr = ROLLOUT, True
        self.run_issue(gh)
        gh.is_pr, gh.repo = False, "elsewhere/repo"
        self.run_issue(gh)
        self.assertEqual(gh.writes, [])
        gh.repo = REPO
        self.run_issue(gh)
        self.assertEqual((gh.kind, gh.labels), ("Feature", {"p1", "hard"}))

    def test_all_template_headings_and_answers_match_the_policy(self):
        folder = Path(__file__).resolve().parents[1] / "ISSUE_TEMPLATE"
        kinds = []
        for path in sorted(folder.glob("*.yml")):
            if path.name == "config.yml":
                continue
            template = path.read_text(encoding="utf-8")
            kind = next(
                line.removeprefix("type: ")
                for line in template.splitlines()
                if line.startswith("type: ")
            )
            kinds.append(kind)
            fields = [
                field
                for field in template.split("  - type: ")[1:]
                if not field.startswith("markdown\n")
            ]
            headings = [
                next(
                    line.strip().removeprefix("label: ")
                    for line in field.splitlines()
                    if line.strip().startswith("label: ")
                )
                for field in fields
            ]
            for field, choices in zip(
                fields[:2], (("p0", "p1", "p2"), ("easy", "medium", "hard")), strict=True
            ):
                self.assertTrue(field.startswith("dropdown\n"))
                self.assertIn("required: true", field)
                self.assertEqual(
                    [
                        line.strip()[2:]
                        for line in field.splitlines()
                        if line.strip().startswith("- ")
                    ],
                    list(choices),
                )
            for priority, difficulty in product(("p0", "p1", "p2"), ("easy", "medium", "hard")):
                values = (priority, difficulty, "Details.")
                body = "\r\n\r\n".join(
                    f"### {h}\r\n\r\n{v}" for h, v in zip(headings[:3], values, strict=True)
                )
                gh = GitHub(kind=None, body=body)
                self.run_issue(gh)
                self.assertEqual((gh.kind, gh.labels), (kind, {priority, difficulty}))
        self.assertCountEqual(kinds, ["Bug", "Feature", "Task"])

    def test_sweep_repairs_valid_stale_state_continues_after_failure_and_honors_cutoff(self):
        failed = GitHub(["p2", "easy"], kind=None, number=4)
        failed.reject_type = True
        stale = GitHub(["p1", "hard"], kind=None, number=3)
        stale.set_type("Task", HUMAN)
        stale.set_type("Bug", BOT)
        recent = GitHub(number=2, state="CLOSED")
        another_failure = GitHub(kind=None, number=6, state="CLOSED")
        another_failure.reject_type = True
        old = GitHub(number=1, state="CLOSED", created=OLD)
        all_issues = [failed, stale, recent, another_failure, old]
        pages = []

        def github(endpoint, *args, payload=None):
            if endpoint == "graphql" and "state" in payload["variables"]:
                state, cursor = payload["variables"]["state"], payload["variables"]["cursor"]
                pages.append((state, cursor))
                if state == "OPEN":
                    rows, next_page = ([failed], "open") if cursor is None else ([stale], None)
                else:
                    rows, next_page = (
                        ([recent, another_failure], "closed")
                        if cursor is None
                        else ([old], "never")
                    )
                return {
                    "data": {
                        "repository": {
                            "issues": connection([g.snapshot() for g in rows], next_page)
                        }
                    }
                }
            for gh in all_issues:
                if endpoint.startswith(f"repos/{REPO}/issues/{gh.number}") or (
                    endpoint == "graphql" and payload["variables"].get("number") == gh.number
                ):
                    return gh(endpoint, *args, payload=payload)
            self.fail(f"Unexpected call to {endpoint}")

        with (
            patch("issue_labels.api", side_effect=github),
            self.assertRaisesRegex(RuntimeError, "#4.*#6"),
        ):
            sweep(REPO)
        self.assertEqual(
            pages, [("OPEN", None), ("OPEN", "open"), ("CLOSED", None), ("CLOSED", "closed")]
        )
        self.assertEqual(stale.kind, "Task")
        self.assertEqual(recent.labels, {"p1", "hard"})
        self.assertEqual(old.writes, [])

    def test_all_triggers_share_one_repository_queue(self):
        workflow = Path(__file__).resolve().parents[1] / "workflows" / "issue-labels.yml"
        block = (
            workflow.read_text(encoding="utf-8").split("\nconcurrency:\n", 1)[1].split("\n\n", 1)[0]
        )
        settings = dict(line.strip().split(": ", 1) for line in block.splitlines())
        self.assertTrue(settings["group"])
        self.assertNotIn("${{", settings["group"])
        self.assertEqual(settings["queue"], "max")
        self.assertEqual(settings["cancel-in-progress"], "false")


if __name__ == "__main__":
    unittest.main()
