# Spec-Driven Development — Reusable Prompt Template

> Use this as a starting prompt when setting up spec-driven development on any new project.
> Works with Claude Code. Adapted from the CRISIS-BENCH methodology.

---

## Quick Start

Copy the prompt in the next section. Replace `[PLACEHOLDERS]` with your project details. Paste into Claude Code. It will generate your roadmap, CLAUDE.md, and all commands.

---

## The Prompt (copy and customize)

```
I want to build [PROJECT_NAME] using spec-driven, test-driven development.

## Project Summary
[1-2 sentences describing what the project does]

## Tech Stack
- Backend: [e.g., Python 3.12 / FastAPI / uvicorn]
- Database: [e.g., PostgreSQL, SQLite, Firestore]
- Frontend: [e.g., Next.js + TypeScript + Tailwind]
- Deployment: [e.g., Docker Compose, GCP Cloud Run, Railway]
- Testing: [e.g., pytest + pytest-asyncio]
- Linting: [e.g., ruff, line-length: 100]

## Development Methodology

### 1. Create These Files First

**roadmap.md** — Master plan with:
- Tech stack table with rationale for each choice
- Budget estimate (infrastructure costs)
- Phases overview table (phase name, spec count, key output)
- Detailed phase tables with columns: Spec | Spec Location | Depends On | Location | Feature | Notes | Status
- Master Spec Index (flat table of ALL specs with status tracking)

**design.md** — Architecture document with:
- ASCII architecture diagram
- Data flow diagram
- Tech stack rationale table
- Deployment architecture
- Key design decisions table

**.claude/CLAUDE.md** — Claude Code context with:
- Project summary (2 lines)
- Key rules (NEVER do X, ALWAYS do Y)
- Tech stack table
- Project structure tree
- Spec folder convention
- Spec-driven development commands table
- Code standards (async, validation, error handling, etc.)
- Testing conventions

### 2. Create These Claude Code Commands

**.claude/commands/create-spec.md**
- Input: spec ID + slug (e.g., "S1.1 dependency-declaration")
- Action: Read roadmap.md + design.md + requirements.md, create specs/spec-{id}-{slug}/spec.md + checklist.md
- Output: Spec folder with filled-in templates, roadmap updated to "spec-written"

**.claude/commands/implement-spec.md**
- Input: spec ID (e.g., "S1.1")
- Action: Load spec.md + checklist.md, follow TDD (Red -> Green -> Refactor)
- Rules: Write tests FIRST, implement minimal code, update checklist progressively
- Output: Working code + passing tests, roadmap updated to "done"

**.claude/commands/verify-spec.md**
- Input: spec ID
- Action: Check code exists, run tests, check lint, audit tangible outcomes
- Output: Verification report (PASS/FAIL with details)

**.claude/commands/check-spec-deps.md**
- Input: spec ID
- Action: Check all prerequisite specs are "done" with passing tests
- Output: Dependency table (READY/BLOCKING per dep)

**.claude/commands/start-spec-dev.md**
- Input: spec ID
- Action: Full workflow — check deps -> create spec (if needed) -> implement (TDD) -> verify -> explain
- Output: Complete spec with code, tests, explanation, and roadmap updated

**.claude/commands/explain-spec.md**
- Input: spec ID
- Action: After implementation, generate explanation.md documenting:
  - WHY: Why this spec exists, what problem it solves, what requirements it addresses
  - WHAT: What it does, key functions/classes, data flow
  - HOW: How it works, implementation details, error handling
  - CONNECTIONS: Upstream dependencies, downstream dependents, shared resources
- Output: explanation.md in the spec folder

### 3. Spec Folder Structure

specs/
  spec-S1.1-{slug}/
    spec.md        <- Requirements, outcomes, TDD notes
    checklist.md   <- Phase-by-phase implementation tracker
    explanation.md <- Post-implementation: why, what, how, connections

### 4. Workflow

For each spec (in dependency order):
  1. /start-spec-dev S{x}.{y}        <- full automated workflow
     OR do it step by step:
  1. /check-spec-deps S{x}.{y}       <- verify prerequisites met
  2. /create-spec S{x}.{y} {slug}    <- generates spec.md + checklist.md
  3. /implement-spec S{x}.{y}        <- TDD implementation
  4. /verify-spec S{x}.{y}           <- post-implementation audit
  5. /explain-spec S{x}.{y}          <- generate explanation.md
  6. Commit when spec is "done"

### 5. Status Flow

pending -> spec-written -> done

- `pending`: Spec exists in roadmap but no spec.md yet
- `spec-written`: spec.md + checklist.md created
- `done`: Code + tests passing + explanation written, roadmap updated
```

---

## Key Principles

1. **Spec before code** — Never write code without a spec
2. **Tests before implementation** — Red -> Green -> Refactor
3. **One spec at a time** — Complete fully before starting next
4. **Dependencies respected** — /check-spec-deps before /implement-spec
5. **Progressive checklist** — Update as you go, not at the end
6. **Roadmap is truth** — Always reflects current project state
7. **Every FR is testable** — If you can't test it, rewrite the requirement
8. **Mock externals** — Tests never hit real APIs or services
9. **No hardcoded secrets** — All config via environment variables
10. **Explain after building** — Every completed spec gets an explanation.md
