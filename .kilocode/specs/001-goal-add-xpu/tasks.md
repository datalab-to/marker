# Tasks: Add XPU Device Handling and Llama-CPP LLM Service

**Input**: Design documents from `.kilocode/specs/001-goal-add-xpu/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → If not found: ERROR "No implementation plan found"
   → Extract: tech stack, libraries, structure
2. Load optional design documents:
   → data-model.md: Extract entities → model tasks
   → contracts/: Each file → contract test task
   → research.md: Extract decisions → setup tasks
3. Generate tasks by category:
   → Setup: project init, dependencies, linting
   → Tests: contract tests, integration tests
   → Core: models, services, CLI commands
   → Integration: DB, middleware, logging
   → Polish: unit tests, performance, docs
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph
7. Create parallel execution examples
8. Validate task completeness:
   → All contracts have tests?
   → All entities have models?
   → All endpoints implemented?
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- **Single project**: `src/`, `tests/` at repository root
- Paths shown below assume single project - adjust based on plan.md structure

## Phase 3.1: Setup
- [x] T001 Review marker directory structure per implementation plan
- [x] T002 Update pyproject.toml with Intel PyTorch endpoint dependencies
- [x] T003 [P] Configure XPU device detection in marker/settings.py

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**
- [x] T004 [P] Create basic functionality verification script in .kilocode/specs/001-goal-add-xpu/test_xpu_llama.py
- [x] T005 [P] Prepare test documents for XPU testing (use existing testfiles/benchmark.pdf)
- [x] T006 [P] Prepare test documents for Llama-CPP testing (use existing testfiles/benchmark.pdf)

## Phase 3.3: Core Implementation (ONLY after tests are failing)
- [x] T007 [P] Create LlamaCPPService class in marker/services/llama_cpp.py
- [x] T008 [P] Extend GPUManager for XPU support in marker/utils/gpu.py
- [x] T009 Update settings.py with XPU device handling logic
- [x] T010 Implement llama-serve API integration in marker/services/llama_cpp.py
- [x] T011 Add XPU availability checking in marker/utils/gpu.py
- [x] T012 Update device selection priority in marker/settings.py

## Phase 3.4: Integration
- [ ] T013 Connect LlamaCPPService to existing LLM processor framework
- [ ] T014 Integrate XPU device handling with existing MPS server functionality
- [ ] T015 Add configuration options for Llama-CPP service parameters
- [ ] T016 Ensure backward compatibility with existing LLM services

## Phase 3.5: Polish
- [ ] T017 Test XPU device detection with actual documents
- [ ] T018 Test Llama-CPP service with actual documents
- [ ] T019 Verify backward compatibility with existing CUDA/CPU processing
- [ ] T020 Update documentation with XPU and Llama-CPP usage instructions
- [ ] T021 Create simple test script for manual verification
- [ ] T022 Run manual testing with actual files
- [ ] T023 Verify no breaking changes to existing functionality

## Dependencies
- Tests (T004-T006) before implementation (T007-T016)
- T007 blocks T010, T013
- T008 blocks T011, T014
- T009 blocks T012
- Implementation before polish (T017-T023)

## Parallel Example
```
# Launch T004-T006 together:
Task: "Create basic functionality verification script in .kilocode/specs/001-goal-add-xpu/test_xpu_llama.py"
Task: "Prepare test documents for XPU testing (use existing testfiles/benchmark.pdf)"
Task: "Prepare test documents for Llama-CPP testing (use existing testfiles/benchmark.pdf)"
```

## Notes
- [P] tasks = different files, no dependencies
- Verify tests fail before implementing
- Commit after each task
- Avoid: vague tasks, same file conflicts
- Focus on actual document testing rather than theoretical unit tests

## Task Generation Rules
*Applied during main() execution*

1. **From Contracts**:
   - Each contract file → contract test task [P]
   - Each endpoint → implementation task
   
2. **From Data Model**:
   - Each entity → model creation task [P]
   - Relationships → service layer tasks
   
3. **From User Stories**:
   - Each story → integration test [P]
   - Quickstart scenarios → validation tasks

4. **Ordering**:
   - Setup → Tests → Models → Services → Endpoints → Polish
   - Dependencies block parallel execution

## Validation Checklist
*GATE: Checked by main() before returning*

- [x] All contracts have corresponding tests
- [x] All entities have model tasks
- [x] All tests come before implementation
- [x] Parallel tasks truly independent
- [x] Each task specifies exact file path
- [x] No task modifies same file as another [P] task