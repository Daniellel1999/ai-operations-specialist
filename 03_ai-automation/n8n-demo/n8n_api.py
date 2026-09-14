"""Talk to an n8n instance from the command line.

Read-only against the public API (`/api/v1`, `X-N8N-API-KEY` header), plus
triggering a workflow through its webhook URL. Nothing here creates, edits or
deletes a workflow — the GUI is where you build, this is where you inspect.

    python n8n_api.py list-workflows
    python n8n_api.py get-workflow 7 --nodes
    python n8n_api.py list-executions --status error
    python n8n_api.py trigger summarize --json '{"text": "hello"}'
"""

import argparse
import json
import os
import sys

import requests

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


class N8nError(Exception):
    """Something the student needs to fix — printed as one line, no traceback."""


def base_url():
    url = os.environ.get("N8N_BASE_URL")
    if not url:
        raise N8nError(
            "N8N_BASE_URL is not set. Copy .env.example to .env and fill it in."
        )
    return url.rstrip("/")


def api_key():
    key = os.environ.get("N8N_API_KEY")
    if not key:
        raise N8nError(
            "N8N_API_KEY is not set. Create one in n8n under "
            "Settings -> n8n API, then put it in .env."
        )
    return key


def get(path, params=None):
    """GET {base}/api/v1{path} with the API key attached."""
    url = f"{base_url()}/api/v1{path}"
    try:
        response = requests.get(
            url, headers={"X-N8N-API-KEY": api_key()}, params=params, timeout=30
        )
    except requests.RequestException as exc:
        raise N8nError(f"Could not reach {url}: {exc}") from exc

    if response.status_code == 401:
        raise N8nError("401 Unauthorized — N8N_API_KEY is wrong, expired, or revoked.")
    if response.status_code == 404:
        raise N8nError(f"404 Not Found — nothing at {url}")
    if not response.ok:
        raise N8nError(f"{response.status_code} from {url}: {response.text[:500]}")
    return response.json()


def get_all(path, params=None, limit=None):
    """Follow n8n's cursor pagination until the results run out.

    Each page is {"data": [...], "nextCursor": "..."}; nextCursor is null on
    the last page. `limit` caps the total number of items returned.
    """
    params = dict(params or {})
    params["limit"] = min(limit, 250) if limit else 250
    items = []
    while True:
        page = get(path, params)
        items.extend(page.get("data", []))
        cursor = page.get("nextCursor")
        if not cursor or (limit and len(items) >= limit):
            break
        params["cursor"] = cursor
    return items[:limit] if limit else items


def dump(value):
    print(json.dumps(value, indent=2, ensure_ascii=False))


def table(rows, headers):
    """Print aligned columns. Everything is stringified first."""
    rows = [["" if cell is None else str(cell) for cell in row] for row in rows]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))


def cmd_list_workflows(args):
    params = {}
    if args.active:
        params["active"] = "true"
    workflows = get_all("/workflows", params, limit=args.limit)
    if args.raw:
        return dump(workflows)
    if not workflows:
        print("No workflows. Build one in the n8n editor first.")
        return
    table(
        [
            [w.get("id"), "active" if w.get("active") else "inactive", w.get("name")]
            for w in workflows
        ],
        ["ID", "STATUS", "NAME"],
    )


def cmd_get_workflow(args):
    workflow = get(f"/workflows/{args.id}")
    if args.raw:
        return dump(workflow)
    if args.nodes:
        table(
            [[n.get("name"), n.get("type")] for n in workflow.get("nodes", [])],
            ["NODE", "TYPE"],
        )
        return
    print(f"id:      {workflow.get('id')}")
    print(f"name:    {workflow.get('name')}")
    print(f"active:  {'yes' if workflow.get('active') else 'no'}")
    print(f"nodes:   {len(workflow.get('nodes', []))}")
    print(f"updated: {workflow.get('updatedAt')}")


def cmd_list_executions(args):
    params = {}
    if args.workflow_id:
        params["workflowId"] = args.workflow_id
    if args.status:
        params["status"] = args.status
    executions = get_all("/executions", params, limit=args.limit)
    if args.raw:
        return dump(executions)
    if not executions:
        print("No executions yet. Trigger the workflow and look again.")
        return
    table(
        [
            [e.get("id"), e.get("workflowId"), e.get("status"), e.get("startedAt")]
            for e in executions
        ],
        ["ID", "WORKFLOW", "STATUS", "STARTED"],
    )


def cmd_get_execution(args):
    # includeData pulls every node's input/output — big, but it's the whole
    # point when you're debugging why a run did the wrong thing.
    execution = get(
        f"/executions/{args.id}",
        {"includeData": "true"} if args.data else None,
    )
    if args.raw or args.data:
        return dump(execution)
    print(f"id:       {execution.get('id')}")
    print(f"workflow: {execution.get('workflowId')}")
    print(f"status:   {execution.get('status')}")
    print(f"started:  {execution.get('startedAt')}")
    print(f"stopped:  {execution.get('stoppedAt')}")


def cmd_trigger(args):
    # Webhooks live outside /api/v1 and take no API key — the URL is the secret.
    # /webhook-test/ only answers while the editor has "Listen for test event"
    # running; /webhook/ needs the workflow activated.
    prefix = "webhook-test" if args.test else "webhook"
    url = f"{base_url()}/{prefix}/{args.path.lstrip('/')}"

    payload = None
    if args.json:
        try:
            payload = json.loads(args.json)
        except json.JSONDecodeError as exc:
            raise N8nError(f"--json is not valid JSON: {exc}") from exc

    try:
        response = requests.post(url, json=payload, timeout=60)
    except requests.RequestException as exc:
        raise N8nError(f"Could not reach {url}: {exc}") from exc

    if response.status_code == 404:
        hint = (
            "the editor isn't listening for a test event"
            if args.test
            else "the workflow isn't active, or the path is wrong"
        )
        raise N8nError(f"404 from {url} — {hint}.")

    print(f"POST {url} -> {response.status_code}")
    try:
        dump(response.json())
    except ValueError:
        print(response.text)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Inspect and trigger workflows on an n8n instance.",
        epilog="Reads N8N_BASE_URL and N8N_API_KEY from the environment or a .env file.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add(name, help_text, handler):
        sub = subparsers.add_parser(name, help=help_text, description=help_text)
        sub.add_argument(
            "--raw", action="store_true", help="print the raw JSON response"
        )
        sub.set_defaults(handler=handler)
        return sub

    workflows = add("list-workflows", "List workflows on the instance.", cmd_list_workflows)
    workflows.add_argument(
        "--active", action="store_true", help="only workflows that are activated"
    )
    workflows.add_argument("--limit", type=int, help="stop after N workflows")

    workflow = add("get-workflow", "Show one workflow.", cmd_get_workflow)
    workflow.add_argument("id", help="workflow ID (see list-workflows)")
    workflow.add_argument(
        "--nodes", action="store_true", help="list the workflow's nodes instead"
    )

    executions = add("list-executions", "List past runs.", cmd_list_executions)
    executions.add_argument("--workflow-id", help="only runs of this workflow")
    executions.add_argument(
        "--status", choices=["success", "error", "waiting"], help="filter by outcome"
    )
    executions.add_argument("--limit", type=int, default=20, help="default: 20")

    execution = add("get-execution", "Show one run.", cmd_get_execution)
    execution.add_argument("id", help="execution ID (see list-executions)")
    execution.add_argument(
        "--data", action="store_true", help="include every node's input/output"
    )

    trigger = add("trigger", "POST to a workflow's webhook URL.", cmd_trigger)
    trigger.add_argument("path", help="the webhook path, e.g. 'summarize'")
    trigger.add_argument("--json", help="request body as a JSON string")
    trigger.add_argument(
        "--test", action="store_true", help="use /webhook-test/ instead of /webhook/"
    )

    return parser


def main():
    if load_dotenv:
        load_dotenv()
    args = build_parser().parse_args()
    try:
        args.handler(args)
    except N8nError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
