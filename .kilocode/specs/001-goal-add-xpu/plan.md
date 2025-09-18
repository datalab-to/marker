# Implementation Plan: Add XPU Device Handling and Llama-CPP LLM Service

**Branch**: `001-goal-add-xpu` | **Date**: 2025-09-16 | **Spec**: [.kilocode/specs/001-goal-add-xpu/spec.md](.kilocode/specs/001-goal-add-xpu/spec.md)
**Input**: Feature specification from `.kilocode/specs/001-goal-add-xpu/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
4. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
5. Execute Phase 1 → contracts, data-model.md, and quickstart.md
6. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
7. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
8. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
This feature will add support for XPU devices in addition to existing CUDA and CPU support, and introduce a new Llama-CPP LLM service option. The implementation will involve updating device detection logic, adding XPU-specific torch handling, and creating a new LLM service implementation that integrates with llama-serve backend.

## Technical Context
**Language/Version**: Python 3.10+  
**Primary Dependencies**: PyTorch, llama-cpp-python, requests  
**Storage**: N/A  
**Testing**: pytest (minimal testing focused on functionality verification)  
**Target Platform**: Linux server  
**Project Type**: single  
**Performance Goals**: Get features working, performance testing not required  
**Constraints**: Must maintain backward compatibility with existing CUDA/CPU configurations  
**Scale/Scope**: Minimal changes to core system for 2 specific features

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Simplicity**:
- Projects: [1] (core marker project)
- Using framework directly? (no wrapper classes) ✓
- Single data model? (no DTOs unless serialization differs) ✓
- Avoiding patterns? (no Repository/UoW without proven need) ✓

**Architecture**:
- EVERY feature as library? (no direct app code) ✓
- Libraries listed: [XPU device detection, Llama-CPP service integration]
- CLI per library: [marker commands will support new device/service options]
- Library docs: llms.txt format planned? ✓

**Testing (NON-NEGOTIABLE)**:
- RED-GREEN-Refactor cycle enforced? (test MUST fail first) ✓
- Git commits show tests before implementation? ✓
- Order: Contract→Integration→E2E→Unit strictly followed? ✓
- Real dependencies used? (actual DBs, not mocks) ✓
- Integration tests for: new libraries, contract changes, shared schemas? ✓
- FORBIDDEN: Implementation before test, skipping RED phase ✓

**Observability**:
- Structured logging included? ✓
- Frontend logs → backend? (unified stream) N/A
- Error context sufficient? ✓

**Versioning**:
- Version number assigned? (MAJOR.MINOR.BUILD) 1.0.0
- BUILD increments on every change? ✓
- Breaking changes handled? (parallel tests, migration plan) ✓

## Project Structure

### Documentation (this feature)
```
.kilocode/specs/001-goal-add-xpu/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# Option 1: Single project (DEFAULT)
src/
├── models/
├── services/
├── cli/
└── lib/

tests/
├── contract/
├── integration/
└── unit/
```

**Structure Decision**: DEFAULT to Option 1 as this is a core system enhancement

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - XPU-specific torch installation requirements and basic compatibility
   - Basic llama-serve API endpoints needed for Llama-CPP service integration

2. **Generate and dispatch research agents**:
   ```
   For each unknown in Technical Context:
     Task: "Research minimal XPU device detection and torch compatibility requirements"
     Task: "Research basic llama-serve API endpoints for Llama-CPP service"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - Device: Simple representation of processing units (XPU/CUDA/CPU)
   - LLM Service Configuration: Basic configuration for LLM service selection

2. **Generate API contracts** from functional requirements:
   - Device detection enhancement: Add XPU detection to existing device detection logic
   - LLM service addition: Add Llama-CPP service option to existing LLM service framework

3. **Generate contract tests** from contracts:
   - Minimal tests to verify basic functionality
   - Tests focused on verifying the two new features work

4. **Extract test scenarios** from user stories:
   - Core scenarios for XPU device usage
   - Core scenarios for Llama-CPP service usage

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, agent-specific file

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Focus on minimal implementation tasks for the two specific features
- Prioritize getting features working over extensive testing
- Create simple verification tests to confirm basic functionality
- Tasks will be ordered to implement XPU support first, then Llama-CPP service

**Ordering Strategy**:
- XPU device detection and handling implementation
- Llama-CPP service implementation
- Basic functionality verification with actual files
- Minimal testing to ensure no breaking changes

**Estimated Output**: Small number of focused tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation (execute tasks.md following constitutional principles)  
**Phase 5**: Validation (run tests, execute quickstart.md with actual files)

## Complexity Tracking
*Fill ONLY if Constitution Check has violations that must be justified*

| Violation                  | Why Needed         | Simpler Alternative Rejected Because |
| -------------------------- | ------------------ | ------------------------------------ |
| Minimal testing approach   | User preference    | Extensive testing not required for these changes |


## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [x] Phase 3: Tasks generated (/tasks command)
- [x] Phase 3.2: Tests created (TDD approach)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented

---
*Based on Constitution v2.1.1 - See `.kilocode/rules/constitution.md`*
