Create a spec for this project.

## Input
- Spec ID and slug: $ARGUMENTS (e.g., "S1.1 project-structure")

## Instructions

1. **Read `roadmap.md`** — find the row matching the spec ID. Extract: Feature, Depends On, Location, Notes.

2. **Read `design.md` and `requirements.md`** — find all relevant sections for this spec's feature. Extract functional requirements, data schemas, architectural details.

3. **Create spec folder**: `specs/spec-{ID}-{slug}/`

4. **Create `spec.md`** using this template:

```markdown
# Spec {ID} -- {Feature Name}

## Overview
[From roadmap row + design.md context. 2-3 sentences on what this spec delivers and why it matters.]

## Dependencies
[From roadmap "Depends On" column. List each dependency with its current status.]

## Target Location
[From roadmap "Location" column. Exact file paths that will be created/modified.]

## Functional Requirements

### FR-1: {Name}
- **What**: Behavior description
- **Inputs**: Parameters, types
- **Outputs**: Return type, side effects
- **Edge cases**: Invalid input, corrupted audio, silence, etc.

### FR-2: {Name}
[Repeat for each requirement]

## Data Models
[Schemas, data formats relevant to this spec]

## Integration Points
[How this spec connects to other parts of the system]

## Tangible Outcomes
- [ ] Outcome 1 (testable and verifiable)
- [ ] Outcome 2 (testable and verifiable)

## Test-Driven Requirements

### Tests to Write First
1. `test_{name}`: Description of what it tests
2. `test_{name}`: Description of what it tests

### Mocking Strategy
- Mock audio I/O, model inference, HuggingFace Hub
- List specific mocks needed

### Coverage
- All public functions must have tests
- All edge cases from FR edge cases must have tests
- Target: >80% code coverage
```

5. **Create `checklist.md`** using this template:

```markdown
# Checklist -- Spec {ID}: {Feature}

## Phase 1: Setup & Dependencies
- [ ] Verify all dependency specs are "done" (run /check-spec-deps)
- [ ] Create target files/directories
- [ ] Install any new dependencies

## Phase 2: Tests First (TDD Red Phase)
- [ ] Create test file: `tests/test_{module}.py`
- [ ] Write failing tests for FR-1
- [ ] Write failing tests for FR-2
- [ ] Run tests — confirm all FAIL (Red)

## Phase 3: Implementation (TDD Green Phase)
- [ ] Implement FR-1 — run tests — expect FR-1 tests pass
- [ ] Implement FR-2 — run tests — expect FR-2 tests pass
- [ ] All tests pass (Green)

## Phase 4: Refactor
- [ ] Clean up code, remove duplication
- [ ] Run ruff — fix any lint issues
- [ ] Run full test suite — all pass

## Phase 5: Integration & Verification
- [ ] Wire into dependent systems if applicable
- [ ] Verify tangible outcomes from spec.md
- [ ] No hardcoded secrets
- [ ] Update roadmap.md status: "pending" -> "spec-written"
```

6. **Update `roadmap.md`** — change the spec's status from `pending` to `spec-written` in both the phase table and the Master Spec Index.
