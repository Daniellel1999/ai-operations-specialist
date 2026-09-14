# n8n-demo

Automating work with AI using n8n — the low-code automation tool pre-installed
on your Network Solutions hosting plan. You deploy the service, put it on your
own domain, build an AI-powered workflow in the GUI, then drive that same
workflow from the REST API with `n8n_api.py`. The point of the session is that
the dashboard and the API are two views of one thing: everything you click, you
can also call.

## Process

```
Network Solutions plan (n8n pre-installed)
 |
 v
Deploy the service -> provisioned hostname
 |
 v
Attach your domain -> https://n8n.your-domain.com (TLS)
 |
 v
Build a workflow in the GUI: Webhook -> AI node -> Respond
 |
 v
Activate it -> production webhook URL goes live
 |
 v
Issue an API key (Settings -> n8n API)
 |
 v
Drive it from code: n8n_api.py list / get / trigger
```

## Steps

### 1. Deploy the n8n service

1. Sign in to the Network Solutions control panel for the plan you bought.
2. Find the pre-installed **n8n** application and start/activate it.
3. Note the **hostname it gives you** — something like
   `your-account.something.networksolutions-hosted.example`. This is your
   instance until you attach your own domain.
4. Open that hostname in a browser. On first load n8n asks you to create the
   **owner account** (email + password). That account owns the instance — use a
   real password manager entry, not a throwaway.

Panels get redesigned; if the wording differs, you are looking for "apps",
"installed applications", or "one-click apps", and then n8n. What you need out
of this step is a URL that loads the n8n editor and an owner login.

### 2. Attach your domain

1. In the Network Solutions **DNS manager** for your domain, add a record for a
   subdomain — `n8n` is the conventional choice:
   - **CNAME** `n8n` -> the provisioned hostname from step 1, or
   - **A** `n8n` -> the instance's IP, if the panel gives you an IP instead.
2. Back in the hosting panel, set `n8n.your-domain.com` as the app's domain so
   the service answers to it and issues a TLS certificate. Wait for the cert —
   this is usually a few minutes, not instant.
3. Check that DNS has propagated:

   ```
   dig +short n8n.your-domain.com
   ```

   You should see the target from step 1. Nothing works until this resolves.
4. Load `https://n8n.your-domain.com` and confirm the padlock — the editor
   should log you in with the owner account from step 1.
5. **Tell n8n its own address.** n8n builds webhook URLs from its configured
   base URL, so until you update it your webhooks will still advertise the old
   hostname. In the panel's environment/config section for the app, set:

   ```
   WEBHOOK_URL=https://n8n.your-domain.com
   N8N_EDITOR_BASE_URL=https://n8n.your-domain.com
   ```

   Restart the app. Then open any Webhook node and confirm the URL it shows now
   starts with your domain.

### 3. Build the workflow in the GUI

This is the AI part. Build a small endpoint that takes text and returns a
structured answer:

1. **Webhook** node (trigger). Method `POST`, path `summarize`. Note both URLs
   it shows you: the **test** URL (`/webhook-test/summarize`, live only while
   you are listening in the editor) and the **production** URL
   (`/webhook/summarize`, live only once the workflow is activated).
2. **An AI node** — either a built-in AI/LLM node, or a generic **HTTP Request**
   node pointed at an LLM API. Feed it `{{ $json.body.text }}` from the webhook
   and prompt it to summarize the text in one sentence and label it with a
   category.
3. **Respond to Webhook** node, returning the model's output as JSON. Without
   this node the webhook replies immediately and you never see the result.
4. Click **Listen for test event**, then fire the test URL (step 5 below,
   with `--test`) and watch the data flow through each node.
5. **Save**, then toggle the workflow **Active**. Only now does
   `/webhook/summarize` answer.

The credentials for whatever LLM you use are configured in the GUI under
**Credentials** — not in this repo, and not in `.env`.

### 4. Issue an API key

1. In the n8n editor: **Settings -> n8n API -> Create an API key**. Copy it now;
   n8n shows it once.
2. Set up your local config:

   ```
   cp 03_ai-automation/n8n-demo/.env.example 03_ai-automation/n8n-demo/.env
   ```

   Fill in `N8N_BASE_URL`, `N8N_API_KEY` and `N8N_WEBHOOK_PATH`.

`.env` is already in the repo's `.gitignore`. The key can read everything in
your instance — it is a password, so don't paste it into chat, screenshots, or a
commit.

### 5. Drive it from the API

Dependencies are declared in the [root `pyproject.toml`](../../pyproject.toml).
From the repo root:

```
uv sync
```

Then, from `03_ai-automation/n8n-demo/`:

```
uv run python n8n_api.py list-workflows
```

```
ID  STATUS    NAME
--------------------------------
7   active    summarize-ticket
8   inactive  draft-reply
```

Everything the script does is a plain HTTP call. The base path is
`/api/v1` and the key goes in an `X-N8N-API-KEY` header — this is the same
request, by hand:

```
curl -H "X-N8N-API-KEY: $N8N_API_KEY" "$N8N_BASE_URL/api/v1/workflows"
```

Commands:

| Command | What it does |
| --- | --- |
| `list-workflows [--active] [--limit N]` | every workflow, ID and active state |
| `get-workflow <id> [--nodes]` | one workflow; `--nodes` lists its nodes |
| `list-executions [--workflow-id ID] [--status success\|error\|waiting]` | past runs |
| `get-execution <id> [--data]` | one run; `--data` includes every node's input/output |
| `trigger <path> [--json '{...}'] [--test]` | POST to the webhook URL |

Add `--raw` to any of them to see the untouched JSON — that JSON is the thing
worth reading, the table is just a convenience.

Trigger the workflow you built:

```
uv run python n8n_api.py trigger summarize --json '{"text": "The payment failed twice and no one replied to my email."}'
```

Use `--test` while the editor is listening; drop it once the workflow is active.
Then look at what happened:

```
uv run python n8n_api.py list-executions --limit 5
uv run python n8n_api.py get-execution 42 --data
```

Two things worth noticing: the workflow you clicked together in the GUI comes
back as JSON from `get-workflow --nodes`, and `--data` shows you exactly what
the AI node received and returned — which is how you debug a bad answer.

## Scope

Read-only plus trigger, deliberately. The script lists, inspects and fires
workflows; it never creates, edits, activates or deletes one, and it doesn't
manage credentials. Building happens in the GUI, where you can see it. Those
write endpoints do exist on the same API if you want them later — the same
header, the same base path, `POST` and `PATCH` instead of `GET`.
