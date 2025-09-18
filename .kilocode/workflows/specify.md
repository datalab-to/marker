---
name: specify
mode: orchestrator
description: Create a new feature specification. Use the /specify command to describe what you want to build. Focus on the what and why, not the how or tech stack.
allowed-tools: read_file, search_files, list_files, apply_diff, write_to_file, execute_command, switch_mode, new_task, ask_followup_question, attempt_completion, update_todo_list
argument-hint: <feature_description>
---

<switch_mode>
  <mode_slug>orchestrator</mode_slug>
  <reason>Always start in Orchestrator mode for high-level planning.</reason>
</switch_mode>

## Steps

Your TODO list is as follows. Use the sections below for detailed instructions.

1. Validate Memory Bank is initialized and up to date.
2. Create a new feature specification file using the provided feature description.
3. Specification Initialization: Populate the spec file with initial structure and placeholders.
4. Specification Development Loop: Fine tune the specification by asking the user for clarifications as needed.
5. Finalize the specification and mark it as ready for review.
6. Report completion with branch name and spec file path.

### 01. Validation

CRITICAL: Create a `new_task` in `architect` mode to check the state of the memory bank using instructions from `.kilocode/specs/core/memory-bank-automation.md`

### 02. New Specification Creation

`switch_mode` to `code` mode.

If no `feature_description` is provided, ask the user to provide one.
   
<ask_followup_question>
  <question>What feature are you building? Focus on the what and why, not the how or tech stack.</question>
</ask_followup_question>

use the response as the `feature_description`.

`execute_command` `.kilocode/specs/scripts/new-features.sh --json "{feature_description}"` from repo root and parse its JSON output for `BRANCH_NAME` and `SPEC_FILE`. All file paths must be absolute.

### 03. Specification Initialization

`switch_mode` to `architect` mode.

`read_file` `.kilocode/specs/templates/spec-template.md` to understand required sections.

Write the specification to `SPEC_FILE` using the template structure, replacing placeholders with concrete details derived from the feature description (arguments) while preserving section order and headings. NEVER update `Review & Acceptance Checklist` until the `Finalization`.

### 04. Specification Development Loop

`switch_mode` to `architect` mode.

Fine tune the specification. `read_file` `.kilocode/specs/core/brainstorming-techniques.md` for brainstorming methods and `.kilocode/specs/core/elicitation-methods.md` for elicitation techniques.

Continue to ask the user for clarifications using `ask_followup_question` as needed to fill out the spec.

Once the specification development loop is complete, update the `SPEC_FILE` with all gathered information and change the status to `In Review`. NEVER update `Review & Acceptance Checklist` until the `Finalization`.

### 05. Finalization

`switch_mode` to `architect` mode.

Ask the user to review the `SPEC_FILE` and confirm it is ready. If they say
`Yes`, run the `Review & Acceptance Checklist`. If any fail, ask the user to
continue to review and repeat this review loop util the
`Review & Acceptance Checklist` passes. ONLY after the
`Review & Acceptance Checklist` passes, change the status of the `SPEC_FILE` to `Approved`.

Store this as a task in the memory bank.

### 06. Report Completion

`switch_mode` to `orchestrator` mode.

Report completion with `BRANCH_NAME`, `SPEC_FILE`, and readiness for the next phase.

Hint the user that next step is to create a project plan using 
`/plan` command. Example command:

```
/plan "Implement feature from {SPEC_FILE}" --details "{implementation_details}"
```
