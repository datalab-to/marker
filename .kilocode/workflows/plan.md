---
name: plan
mode: orchestrator
description: "Plan how to implement the specified feature. This is the second step in the Spec-Driven Development lifecycle."
allowed-tools: read_file, search_files, list_files, apply_diff, write_to_file, execute_command, switch_mode, new_task, ask_followup_question, attempt_completion, update_todo_list
argument-hint: <implementation_details>
---

<switch_mode>
  <mode_slug>orchestrator</mode_slug>
  <reason>Always start in Orchestrator mode for high-level planning.</reason>
</switch_mode>

## Steps

Your TODO list is as follows. Use the sections below for detailed instructions.

1. Setup
2. Analyze Feature Specification
3. Plan Development Loop
4. Validation
5. Report Completion

### 01. Setup

`switch_mode` to `code` mode.

`execute_command` `.kilocode/specs/scripts/setup-plan.sh --json` from the repo
root and parse its JSON output for `FEATURE_SPEC`, `IMPL_PLAN`, `SPECS_DIR`,
`BRANCH`. All future file paths must be absolute.

### 02. Analyze Feature Specification

`switch_mode` to `architect` mode.

This step is critical to understand what you are planning for and is only
performed once to load context.

Read and analyze the feature specification to understand:
 - The feature requirements and user stories
 - Functional and non-functional requirements
 - Success criteria and acceptance criteria
 - Any technical constraints or dependencies mentioned

Read the memory bank and constitution to understand constitutional requirements.

### 03. Plan Development Loop

`switch_mode` to `architect` mode.

Execute the implementation plan template:
 - `read_file` `.kilocode/specs/templates/plan-template.md` (already copied to 
   `IMPL_PLAN` path)
 - Set Input path to `FEATURE_SPEC`
 - Run the Execution Flow (main) function steps to complete phases 0 and 1.
 - The template is self-contained and executable
 - Follow error handling and gate checks as specified
 - Let the template guide artifact generation in `SPECS_DIR`:
   * Phase 0 generates `research.md`
   * Phase 1 generates `data-model.md`, `contracts/`, `quickstart.md`
 - Track each phase as a TODO and execute each phase in a `new_task` in
   `architect` mode.
 - Incorporate user-provided details from arguments into Technical Context:
   `{implementation_details}`
 - ALWAYS update `Progress Tracking` in `IMPL_PLAN` as you complete each phase

### 04. Validation

`switch_mode` to `architect` mode.

Verify execution completed:
 - Check Progress Tracking shows phases 0 and 1 as complete
 - Ensure all required artifacts were generated
 - Confirm no ERROR states in execution

Ask the user to review the artifacts from phase 0 and 1 and confirm they are
satisfactory. If they say `Yes`, proceed. If `No`, ask for specific changes,
make updates, and repeat the Validation step until the user is satisfied.

### 05. Report Completion

`switch_mode` to `orchestrator` mode.

Report results with `BRANCH_NAME`, `SPECS_DIR`, and generated artifacts.

Use absolute paths with the repository root for all file operations to avoid
path issues.

Hint the user that next step is to create tasks using the `/tasks` command. Example command:

```
/tasks "Generate tasks for {IMPL_PLAN}" --details "Any additional context"
```