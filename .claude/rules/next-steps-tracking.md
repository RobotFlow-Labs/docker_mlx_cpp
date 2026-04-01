# NEXT_STEPS.md — Mandatory Session Tracking

## Rule: ALWAYS read and update NEXT_STEPS.md

**On session start:**
1. FIRST thing you do: read `NEXT_STEPS.md` in the project root
2. If it does not exist, create it
3. Summarize what was left to do before starting any work

**On session end (or when asked to save/commit):**
1. Update `NEXT_STEPS.md` with:
   - Current date (use ISO format: YYYY-MM-DD)
   - What was accomplished this session
   - What is still TODO
   - Any blocking issues
   - Models/datasets needed (with download commands if known)
   - MVP readiness score (0-100%)

**Why:** We run multiple agents across sessions. Without NEXT_STEPS.md, context is lost and work gets duplicated.
