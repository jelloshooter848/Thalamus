# THALAMUS roadmap

Each phase ships as its own pull request, with tests and a browser check.

| Phase | What it adds | Status |
|---|---|---|
| **1. Quick wins and foundations** | Fast path for grounded web lookups; streamed replies with status notes; conversations that survive restarts; per-service cost ledger | ✅ done |
| **2. Long-term memory of you** | Distilled semantic facts; "sleep" consolidation, with JEV checking each fact against its sources; "What I know about you" tab (edit, pin, delete); "forget that" in chat | next |
| **3. Email triage, watchers and notifications** | Read-only email (Gmail and iCloud/ISP over IMAP, Outlook via Microsoft sign-in); JEV triage of category, importance, needs-reply and scam; Email tab; ntfy phone notifications; watchers; morning brief | planned |
| **4. More senses** | Images (a "visual cortex" description for JEV, the image itself for Claude); PDFs and documents into memory; voice in and out using the phone's built-in speech | planned |
| **5. Evaluation and polish** | `thalamus eval` (THALAMUS vs Claude Opus on your questions); habits (turning repeated slow answers into fast JEV decisions); memory backups; cost charts; conversation history; tray icon | planned |
| **6. Old-mailbox cleanup** | Header-only scan grouped by sender; JEV classifies groups (spam, promotions, keep); review screen; moves to folders with a dry run, log and undo, never deletes; unsubscribe list | planned |

**Decisions so far:**
- Phases run in the order 1 → 6.
- Notifications use ntfy (free).
- Email access is read-only for triage. Only the Phase 6 cleanup, when run, may move mail, and never deletes.
