#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Adapted from OpenSDLC managed-agent-platform; modified for OpenDIKW issue forms and rollout.
"""Keep the native issue type and one label each for priority and difficulty."""

import json
import os
import re
import subprocess
from datetime import datetime

ROLLOUT = os.environ.get("ISSUE_TRIAGE_ROLLOUT", "")
GROUPS = (("p1", "p0", "p2"), ("hard", "medium", "easy"))
FORM_TYPES = {
    "What happened?": "Bug",
    "What problem does this solve?": "Feature",
    "Task details": "Task",
}
LEGACY_TYPES = {"bug": "Bug", "enhancement": "Feature", "feature": "Feature"}
FORM = re.compile(
    r"\A### Priority\n\n([^\n]*)\n\n### Difficulty\n\n([^\n]*)\n\n"
    r"### (" + "|".join(map(re.escape, FORM_TYPES)) + r")(?:\n|$)"
)
BOT = {"__typename": "Bot", "login": "github-actions"}

# GitHub returns the timeline in order. Use the same snapshot for a webhook
# and a sweep, including on retries where the current fields already look valid.
HISTORY = """
    timelineItems(first: 100, after: $historyCursor, itemTypes: [
        LABELED_EVENT, UNLABELED_EVENT, ISSUE_TYPE_ADDED_EVENT, ISSUE_TYPE_CHANGED_EVENT, ISSUE_TYPE_REMOVED_EVENT
    ]) {
        pageInfo { hasNextPage endCursor }
        nodes {
            __typename
            ... on LabeledEvent { createdAt actor { __typename login } label { name } }
            ... on UnlabeledEvent { createdAt actor { __typename login } label { name } }
            ... on IssueTypeAddedEvent { createdAt actor { __typename login } issueType { name } }
            ... on IssueTypeChangedEvent { createdAt actor { __typename login } issueType { name } prevIssueType { name } }
            ... on IssueTypeRemovedEvent { createdAt actor { __typename login } issueType { name } }
        }
    }
"""
SNAPSHOT = (
    """fragment Triage on Issue {
    id number state createdAt body repository { nameWithOwner } issueType { name }
    labels(first: 100) { nodes { id name } pageInfo { hasNextPage } }
"""
    + HISTORY
    + "}"
)


def api(endpoint, *args, payload=None):
    command = ["gh", "api", endpoint, *args]
    if payload is not None:
        command.extend(["--input", "-"])
    result = subprocess.run(
        command,
        input=json.dumps(payload) if payload is not None else None,
        stdout=subprocess.PIPE,
        check=True,
        encoding="utf-8",
    )
    return json.loads(result.stdout)


def read_query(query, **variables):
    return api("graphql", payload={"query": query, "variables": variables})["data"]


def filed_since_rollout(timestamp):
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00")) >= datetime.fromisoformat(
        ROLLOUT.replace("Z", "+00:00")
    )


def triage_write(event):
    # Earlier Actions workflows made real choices before this policy existed.
    return event["actor"] == BOT and filed_since_rollout(event["createdAt"])


def issue_history(issue):
    page = issue["timelineItems"]
    events = list(page["nodes"])
    while page["pageInfo"]["hasNextPage"]:
        query = (
            """query IssueTriageHistory($id: ID!, $historyCursor: String) {
            node(id: $id) { ... on Issue { """
            + HISTORY
            + "} } }"
        )
        page = read_query(query, id=issue["id"], historyCursor=page["pageInfo"]["endCursor"])[
            "node"
        ]["timelineItems"]
        events.extend(page["nodes"])
    return events


def desired_type(current, labels, form, events):
    fallback = FORM_TYPES[form[3]] if form else "Feature"
    changes = [event for event in events if "issueType" in event]
    chosen = current
    if changes:
        first = changes[0]
        # Recover the type that preceded the first recorded transition.
        chosen = (first.get("prevIssueType") or {}).get("name")
        if first["__typename"] == "IssueTypeRemovedEvent":
            chosen = (first["issueType"] or {}).get("name")
    for event in changes:
        if triage_write(event) and chosen in FORM_TYPES.values():
            continue  # A stale repair cannot replace a newer choice or explicit clear.
        if event["__typename"] == "IssueTypeRemovedEvent":
            chosen = fallback
        else:
            chosen = (event["issueType"] or {}).get("name")
    if chosen in FORM_TYPES.values():
        return chosen
    legacy = {LEGACY_TYPES[name] for name in labels if name in LEGACY_TYPES}
    if len(legacy) == 1:
        return next(iter(legacy))
    return fallback


def desired_label(group, current, answer, events):
    # Undo the recorded changes to seed imported or pre-rollout labels.
    changes = [event for event in events if (event.get("label") or {}).get("name") in group]
    rank = dict.fromkeys(current.intersection(group), (True, 0))
    for event in reversed(changes):
        name = event["label"]["name"]
        if event["__typename"] == "LabeledEvent":
            rank.pop(name, None)
        else:
            rank[name] = (True, 0)

    def winner():
        return max((name for name in group if name in rank), key=rank.get, default=None)

    withdrawn = set()
    for index, event in enumerate(changes, 1):
        name = event["label"]["name"]
        bot = triage_write(event)
        if event["__typename"] == "LabeledEvent":
            if not bot:
                withdrawn.discard(name)
                rank[name] = (True, index)
            elif name not in rank and name not in withdrawn:
                rank[name] = (False, index)
        elif not bot or name != winner():
            if not bot:
                withdrawn.add(name)
            # Cleanup can remove a loser. A stale deletion of the current
            # winner cannot cancel its newer human assignment.
            rank.pop(name, None)
    return winner() or (answer if answer in group else group[0])


def reconcile_snapshot(repository, issue, *, write):
    if (
        not issue
        or issue["repository"]["nameWithOwner"].lower() != repository.lower()
        or (issue["state"] == "CLOSED" and not filed_since_rollout(issue["createdAt"]))
    ):
        return False
    path = f"repos/{repository}/issues/{issue['number']}"
    if issue["labels"]["pageInfo"]["hasNextPage"]:
        pages = api(path + "/labels?per_page=100", "--paginate", "--slurp")
        current = {label["name"]: label["node_id"] for page in pages for label in page}
    else:
        current = {label["name"]: label["id"] for label in issue["labels"]["nodes"]}
    names = set(current)
    events = issue_history(issue)
    form = FORM.match((issue["body"] or "").replace("\r\n", "\n"))
    native = (issue["issueType"] or {}).get("name")
    kind = desired_type(native, names, form, events)
    answers = form.groups()[:2] if form else (None, None)
    desired = {
        desired_label(group, names, answer, events)
        for group, answer in zip(GROUPS, answers, strict=True)
    }
    if kind == native and desired == names:
        return False
    if write:
        if kind != native:
            saved = api(path, "--method", "PATCH", payload={"type": kind})
            if (saved["type"] or {}).get("name") != kind:
                raise RuntimeError(f"GitHub did not apply issue type {kind}")
        missing = sorted(desired - names)
        if missing:
            api(path + "/labels", payload={"labels": missing})
        extra = sorted(names - desired)
        if extra:
            api(
                "graphql",
                payload={
                    "query": """mutation($input: RemoveLabelsFromLabelableInput!) {
                    removeLabelsFromLabelable(input: $input) { clientMutationId }
                }""",
                    "variables": {
                        "input": {
                            "labelableId": issue["id"],
                            "labelIds": [current[name] for name in extra],
                        }
                    },
                },
            )
    return True


def missing_issue(error):
    try:
        errors = json.loads(error.stdout or "{}").get("errors", [])
    except ValueError:
        return False
    return bool(errors) and all(
        item.get("type") == "NOT_FOUND" and item.get("path") == ["repository", "issueOrPullRequest"]
        for item in errors
    )


def reconcile(repository, number):
    owner, name = repository.split("/")
    query = (
        """query IssueTriage($owner: String!, $name: String!, $number: Int!,
                                 $historyCursor: String) {
        repository(owner: $owner, name: $name) { issueOrPullRequest(number: $number) { ...Triage } }
    }"""
        + SNAPSHOT
    )
    # A GITHUB_TOKEN write does not trigger a new issues run. A second read
    # settles a concurrent edit. Future runs can replay even a lost write reply.
    for _ in range(2):
        try:
            issue = read_query(query, owner=owner, name=name, number=number)["repository"][
                "issueOrPullRequest"
            ]
        except subprocess.CalledProcessError as error:
            if missing_issue(error):  # Deleted while the event waited in the queue.
                return
            raise
        if not reconcile_snapshot(repository, issue, write=True):
            break


def eligible_issues(repository):
    owner, name = repository.split("/")
    query = (
        """query IssueTriageSweep($owner: String!, $name: String!, $state: IssueState!, $cursor: String,
                                     $closedSince: DateTime, $historyCursor: String) {
        repository(owner: $owner, name: $name) {
            issues(states: [$state], filterBy: {since: $closedSince}, first: 25, after: $cursor,
                   orderBy: {field: CREATED_AT, direction: DESC}) {
                nodes { ...Triage } pageInfo { hasNextPage endCursor }
            }
        }
    }"""
        + SNAPSHOT
    )
    for state in ("OPEN", "CLOSED"):
        cursor = None
        while True:
            page = read_query(
                query,
                owner=owner,
                name=name,
                state=state,
                cursor=cursor,
                closedSince=ROLLOUT if state == "CLOSED" else None,
            )["repository"]["issues"]
            for issue in page["nodes"]:
                if state == "CLOSED" and not filed_since_rollout(issue["createdAt"]):
                    return
                yield issue
            if not page["pageInfo"]["hasNextPage"]:
                break
            cursor = page["pageInfo"]["endCursor"]


def sweep(repository):
    failures = []
    inspected = 0
    repaired = 0
    for issue in eligible_issues(repository):
        inspected += 1
        try:
            if reconcile_snapshot(repository, issue, write=False):
                reconcile(repository, issue["number"])
                repaired += 1
        except Exception as error:
            failures.append(f"#{issue['number']} ({error})")
    print(f"Checked {inspected} eligible issues in {repository}, reconciled {repaired}.")
    if failures:
        raise RuntimeError("Failed to classify " + ", ".join(failures))


if __name__ == "__main__":
    if not ROLLOUT:
        raise SystemExit("Set ISSUE_TRIAGE_ROLLOUT to the UTC activation time.")
    repo = os.environ["GITHUB_REPOSITORY"]
    if number := os.environ.get("ISSUE_NUMBER"):
        reconcile(repo, int(number))
    else:
        sweep(repo)
