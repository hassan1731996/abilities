# Slack Voice Operator

A voice-first Slack companion for OpenHome. Read and summarise channel activity, catch @mentions, and send messages — all hands-free, with LLM-powered summaries instead of raw message dumps.

## What makes it different from Alexa

| Feature | Alexa Slack Skill | Slack Voice Operator |
|---------|------------------|---------------------|
| Message reading | Reads messages verbatim | LLM condenses to 2–3 sentence summary |
| @mention alerts | Reads all messages | Proactive interrupt only for @mentions, urgency-scored |
| Recipient lookup | Exact handle required | Natural name ("message Jake") fuzzy-matched |
| Channel resolution | Exact channel name | "my product channel" → LLM-matched |
| Background monitoring | None | Daemon polls every 10 min, interrupts only on new mentions |

## Setup

### 1. Create a Slack App

1. Go to [api.slack.com/apps](https://api.slack.com/apps) → **Create New App** → **From scratch**
2. Choose your workspace

### 2. Add OAuth Scopes

Under **OAuth & Permissions → Bot Token Scopes**, add:

| Scope | Purpose |
|-------|---------|
| `channels:history` | Read public channel messages |
| `channels:read` | List public channels |
| `groups:history` | Read private channel messages |
| `groups:read` | List private channels |
| `im:history` | Read direct messages |
| `im:read` | List direct message conversations |
| `chat:write` | Send messages |
| `users:read` | Look up workspace members |

### 3. Install & Get Token

1. **Install to Workspace** (button on the OAuth page)
2. Copy the **Bot User OAuth Token** (`xoxb-...`)
3. In OpenHome platform settings, add a key: `slack_bot_token` = your token

### 4. Invite the Bot to Channels

In each Slack channel you want the ability to read, type:
```
/invite @your-bot-name
```

### 5. First Voice Run

Say any trigger phrase — the ability walks you through a one-time setup: finding your user ID by display name, picking channels to watch for background mention alerts.

## Trigger Phrases

- `check my Slack` / `any Slack messages`
- `any mentions` / `did anyone ping me`
- `what's new in Slack` / `what did I miss on Slack`
- `summarize #engineering` / `what happened in product`
- `message Jake on Slack: I'll be 5 minutes late`
- `send a Slack message to #general`
- `list my Slack channels`
- `change my Slack settings`

## Example Conversations

**Checking mentions:**
> "Any mentions?"
> → "You have 2 mentions in the last 24 hours. Alex pinged you in #engineering asking for a review on PR 47. Sara asked in #product when the design spec will be ready."

**Channel summary:**
> "What's happening in engineering?"
> → "Here's what's happening in #engineering: The team decided to delay the v2 release by one sprint. There's a blocker on the auth service — Ben is investigating. Three PRs are waiting for review."

**Sending a message:**
> "Message Jake: I'll be a few minutes late to standup."
> → "Sending to Jake Smith: 'I'll be a few minutes late to standup' — shall I send it?"
> "Yes."
> → "Sent."

**Background interrupt (no trigger needed):**
> "Heads up — you have 1 urgent Slack mention in #engineering. Say 'check my Slack' for details."

## Storage

All data is persisted in context storage under key `slack_voice_operator`:
- `slack_user_id` — your Slack User ID (resolved by display name on first run)
- `watch_channels` — channel IDs monitored by the background daemon
- `channel_cache` — list of all channels the bot has access to
- `user_cache` — workspace member list for name resolution
- `last_mention_ts` — timestamp of the last processed @mention

## Notes

- The background daemon polls every 10 minutes. It only interrupts for @mentions, never for general channel activity.
- `users.list` fetches up to 200 members. For large workspaces, name matching uses the most common names. If a name isn't found, try the exact Slack display name.
- The bot must be **invited** to each channel (`/invite @bot`) — it cannot read channels it's not a member of.
- This ability uses a **Bot Token** (`xoxb-`). User tokens (`xoxp-`) also work if you prefer to send messages as yourself.
