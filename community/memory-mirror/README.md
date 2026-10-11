# Memory Mirror

![Community](https://img.shields.io/badge/OpenHome-Community-orange?style=flat-square)
![Author](https://img.shields.io/badge/Author-@hassan1731996-lightgrey?style=flat-square)

## What It Does
Ask your assistant what it knows about you, where it learned each thing, and fix anything that's wrong, all by voice. Memory Mirror reads every memory note OpenHome feeds into your Agent and tells you what each one says. It names the source behind a belief, and it lets you add corrections or topics to avoid that the Agent picks up within a couple of minutes.

## Suggested Trigger Words
- "memory mirror"
- "what do you know about me"
- "what do you remember about me"
- "why do you think that about me"
- "where did you learn that about me"

Avoid phrases that start with "forget …" or "remember that …". Voice Memory Capture and Social Memory already use them.

## Setup
- No API keys, accounts or external services.
- Works in the Live Editor and on any device.

## How It Works
OpenHome injects every persistent `.md` file in your user storage into the Agent's prompt about every 60–90 seconds. These include the platform's own `user_profile.md` and `user_summary.md`, plus notes written by skills such as Jarvis Prime, timers and reminders, weather, Housemate and Town Hall. Memory Mirror makes that hidden context audible.

| You say | Memory Mirror does |
|---|---|
| "What do you know about me?" | Lists the sources it can see, then goes through them one at a time if you want, in at most three short points each. Details are spoken only when you ask, so nothing personal is read out unprompted in a shared room. |
| "Why do you think I'm a runner?" | Searches those notes and the earlier conversation, then names the source and quotes it. Every quote is checked in code to be word-for-word from that source. If nothing supports the belief, it says so. |
| "That's wrong, I'm vegetarian now" | Turns your words into one neutral fact, asks you to confirm, then saves it to `memory_mirror_corrections.md`. That file tells the Agent to prefer your confirmed corrections over anything else in its context. |
| "Stop bringing up my job search" | Adds a short "avoid unless the user raises it" rule after you confirm. |
| "List my corrections" / "remove the diet one" / "undo" | Manages your own corrections. Undo covers your last 10 changes. |

Say "done", "that's all" or "goodbye" at any point to leave. Two silent turns also end the session.

### Safety and privacy
- **Read-only toward everything else.** It never edits or deletes other skills' notes or the platform-owned `user_profile.md` / `user_summary.md`. The only files it writes are `memory_mirror_state.json` and `memory_mirror_corrections.md`.
- **Other skills' notes are treated as untrusted data.** They are fenced off in prompts with random markers. The intent router only ever sees your own words, and every change requires your spoken "yes". A note can't make Memory Mirror do anything.
- **Corrections are limited.** At most 6 short entries, and the injected file stays under 180 words. Requests that try to instruct or reprogram the Agent are refused.
- **Logs never contain your memory.** They record intents, counts and file names only.
- **Some memory can't be read.** A few skills append directly to the Agent's personality prompt, and the platform can't read that back. When a belief can't be traced, Memory Mirror says so instead of guessing.

### How it differs from existing options
- **Dashboard → Profile** lets you hand-edit two files on a screen. Memory Mirror is voice-first, covers every injected note from any skill, and explains where a belief came from.
- **Voice Memory Capture** and **Private Notes** store notes you dictate. Memory Mirror audits what the assistant already believes about you.
- **Jarvis Prime** and **Social Memory** each manage only their own data. Memory Mirror looks across all of it.

## Example Conversation
> **User:** "What do you know about me?"
> **AI:** "I'm drawing on 3 sources: your OpenHome profile, your conversation summary and your timers and reminders. Want me to start with your OpenHome profile?"
> **User:** "Yes."
> **AI:** "According to your OpenHome profile: mentions that you live in Manchester. Mentions that you enjoy steak dinners. Next source, or are you done?"
> **User:** "Why do you think I like steak?"
> **AI:** "That comes from your OpenHome profile, which says: enjoys steak dinners on Fridays. If that's wrong, say correct it and tell me what's true."
> **User:** "That's wrong, I'm vegetarian now."
> **AI:** "Should I remember that you are vegetarian? Please say yes or no."
> **User:** "Yes."
> **AI:** "Noted: you are vegetarian. I'll use that within a couple of minutes."
> **User:** "That's all."
> **AI:** "Okay. Ask me anytime."
