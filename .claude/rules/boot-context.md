# Boot Context — MANDATORY on every session

## Rule: Read project state BEFORE doing anything

On EVERY session start, read these files in order:
1. `NEXT_STEPS.md` — current status, blockers, what's done
2. `PRD.md` — full PRD including Build Plan table (find ⬜ PRDs)
3. `anima_module.yaml` — module manifest (if exists)
4. `git log --oneline -10` — recent commits from this and other agents

Only AFTER reading all of these should you respond to the user or start working.

If `NEXT_STEPS.md` doesn't exist, create it.
If `PRD.md` has a Build Plan table, identify the first ⬜ PRD.
If git log shows recent commits with PRD numbers, skip those PRDs.

**Why:** Multiple agents work on projects across sessions. Without reading state first, agents duplicate work, miss blockers, and lose context. This rule ensures every agent starts informed.
