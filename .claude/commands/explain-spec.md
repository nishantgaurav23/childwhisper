Generate a comprehensive explanation of a completed spec.

## Input
- Spec ID: $ARGUMENTS (e.g., "S1.1")

## Instructions

This command creates `explanation.md` in the spec folder. It documents the "why", "what", and "how" of the spec.

1. **Read the spec files**:
   - `specs/spec-{ID}-*/spec.md` — requirements and outcomes
   - `specs/spec-{ID}-*/checklist.md` — implementation tracker
   - The actual source code files listed in spec.md "Target Location"
   - The test files

2. **Read context files**:
   - `roadmap.md` — understand where this spec sits in the project
   - `design.md` — understand the architectural context
   - `requirements.md` — understand which FRs this spec satisfies

3. **Create `specs/spec-{ID}-{slug}/explanation.md`** with this structure:

```markdown
# Spec {ID}: {Feature Name} — Explanation

## Why This Spec Exists

### The Problem
[What problem does this functionality solve?]

### The Need
[Why was this specific approach chosen?]

### Requirements Addressed
[List the FR-xxx and NFR-xxx requirements this spec satisfies]

## What It Does

### In One Sentence
[Single sentence: "{Spec name} provides {capability} so that {benefit}"]

### Key Capabilities
1. **{Capability 1}**: [What it does]
2. **{Capability 2}**: [What it does]

### Important Functions / Classes
| Function/Class | Purpose | Called By |
|---------------|---------|----------|
| `function_name()` | What it does | Which modules call it |

### Data Flow
[How data enters, is processed, and exits this module]

## How It Works

### Architecture
[Internal design — patterns used, key decisions]

### Implementation Details
[Notable implementation choices]

### Error Handling
[How failures are handled]

## How It Connects to the Project

### Upstream Dependencies
| Dependency | Why Needed |
|-----------|-----------|
| S{x}.{y} | Provides X that this spec uses for Y |

### Downstream Dependents
| Dependent | How It Uses This Spec |
|----------|---------------------|
| S{x}.{y} | Uses X to do Y |

## Testing Summary
- Total tests: X
- Key test scenarios: [list the most important tests]
- Mocks used: [what external services are mocked]
```

4. **Verify the explanation** — ensure all sections are filled with specific, accurate information from the actual code.

## Purpose

This explanation serves multiple goals:
- **Onboarding**: Understand the module without reading all the code
- **Architecture documentation**: Shows how the module fits into the system
- **Decision record**: Captures why specific approaches were chosen
- **Spec-to-code traceability**: Links requirements to implementation
