---
name: tasks
mode: orchestrator
description: "Break down the plan into executable tasks. This is the third step in the Spec-Driven Development lifecycle."
allowed-tools: read_file, search_files, list_files, apply_diff, write_to_file, execute_command, switch_mode, new_task, ask_followup_question, attempt_completion, update_todo_list
---

<switch_mode>
  <mode_slug>orchestrator</mode_slug>
  <reason>Always start in Orchestrator mode for high-level planning.</reason>
</switch_mode>

## Steps

Your TODO list is as follows. Use the sections below for detailed instructions.

1. Setup
2. Analyze Feature Specification
3. Task Generation
4. Finalization
5. Report Completion

### 01. Setup

`switch_mode` to `code` mode.

`execute_command` `.kilocode/specs/scripts/check-task-prerequisites.sh --json`
from repo root and parse `FEATURE_DIR` and `AVAILABLE_DOCS` list. All paths must
be absolute.

### 02. Analyze Feature Specification

`switch_mode` to `architect` mode.

Load and analyze available design documents:
 - Always read `plan.md` for tech stack and libraries
 - IF EXISTS: Read `data-model.md` for entities
 - IF EXISTS: Read `contracts/` for API endpoints
 - IF EXISTS: Read `research.md` for technical decisions
 - IF EXISTS: Read `quickstart.md` for test scenarios

Note: Not all projects have all documents. For example:
  - CLI tools might not have `contracts/`
  - Simple libraries might not need `data-model.md`
  - Generate tasks based on what's available

### 03. Task Generation

`switch_mode` to `architect` mode.

1. Generate tasks following the template:
   - Use `.kilocode/specs/templates/tasks-template.md` as the base
   - Replace example tasks with actual tasks based on:
     * **Setup tasks**: Project init, dependencies, linting
     * **Test tasks [P]**: One per contract, one per integration scenario
     * **Core tasks**: One per entity, service, CLI command, endpoint
     * **Integration tasks**: DB connections, middleware, logging
     * **Polish tasks [P]**: Unit tests, performance, docs

2. Task generation rules:
   - Each contract file → contract test task marked [P]
   - Each entity in data-model → model creation task marked [P]
   - Each endpoint → implementation task (not parallel if shared files)
   - Each user story → integration test marked [P]
   - Different files = can be parallel [P]
   - Same file = sequential (no [P])

3. Order tasks by dependencies:
   - Setup before everything
   - Tests before implementation (TDD)
   - Models before services
   - Services before endpoints
   - Core before integration
   - Everything before polish

4. Include parallel execution examples:
   - Group [P] tasks that can run together
   - Show actual Task agent commands

5. Create `FEATURE_DIR/tasks.md` with:
   - Correct feature name from implementation plan
   - Numbered tasks (T001, T002, etc.)
   - Clear file paths for each task
   - Dependency notes
   - Parallel execution guidance

The tasks.md should be immediately executable - each task must be specific
enough that an LLM can complete it without additional context.

### 04. Finalization

`switch_mode` to `architect` mode.

Ask the user to review the `FEATURE_DIR/tasks.md` and confirm it is ready. If
they say `Yes`, update `Progress Tracking` in `FEATURE_DIR/plan.md` to mark
phase 2 and 3 as complete. If they say `No`, ask for specific changes, and
repeat until they say `Yes`.

Update the memory bank.

### 05. Report Completion

`switch_mode` to `orchestrator` mode.

Report completion with `FEATURE_DIR/tasks.md` and readiness for the next phase.

Hint the user that next step is to create to begin executing tasks using `Code`
mode. Include these details in the hint:

> Consider installing the `Code Reviewer`, `Code Simplifier`, and 
> `Test Engineer` modes from the Kilo Code Marketplace. Then from the
> `Orchestrator` mode, you can run a command like this. **REMEMBER** click that
> enhanced prompt button for that extra Kilo power!

```
Implement tasks from `{FEATURE_DIR}/tasks.md` using `code` mode. Each task must
be tracked as a todo. Each todo must be executed using `new_task` and must
maintain it's own todo list. Always validate each task completion with tests.
Use TDD principles. Automatically switch to the `code-reviewer` mode to review complex code changes. Use the `test-engineer` mode to create and run tests.
Simplify complex code with the `code-simplifier` mode. Do multiple passes if 
needed to complete each task fully before moving to the next.

ALWAYS update the tasks list in `{FEATURE_DIR}/tasks.md` as you complete each
task so that it reflects the current state.

NEVER attempt to run a completed task again. Mark it complete and move on to the
next.
```
