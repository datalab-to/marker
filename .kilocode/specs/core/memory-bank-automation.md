# Memory Bank Automation

## Steps

<switch_mode>
  <mode_slug>architect</mode_slug>
  <reason>Memory management optimization works best with the deep thinking Architect.</reason>
</switch_mode>

Your TODO list is as follows. Use the sections below for detailed instructions.
1. Check if the memory bank is already initialized. 
2. If it is not initialized, ask the user if they want to initialize it.

### 01. Check if the memory bank is already initialized. 

Check if the memory bank is already initialized by checking for the `Core Files` in `.kilocode/rules/memory-bank/`.

If it is, ALWAYS evaluate `Automated Memory Bank Update`.

If it is not, ALWAYS evaluate `02. Initialize Memory Bank`.

### 02. Initialize Memory Bank

<ask_followup_question>
  <question>Would you like me to initialize the Memory Bank?</question>
  <follow_up>
    <suggest>Yes</suggest>
    <suggest>No</suggest>
  </follow_up>
</ask_followup_question>

If the user responds "No", end memory bank automation.

If the user responds "Yes"

Initialize memory bank. For each of the `Core Files` needed, use `update_todo_list` to manage tracking and progress.

After the memory back is initialized, ask the user to verify the content of the files using `ask_followup_question`.

<ask_followup_question>
  <question>Is the Memory Bank initialized correctly and ready?</question>
  <follow_up>
  <suggest>Yes</suggest>
  <suggest>No</suggest>
  </follow_up>
</ask_followup_question>

If the user responds "No", ask them what needs to be changed and update the
files accordingly. Continue this loop until the user responds "Yes".

## Automated Memory Bank Update

<ask_followup_question>
  <question>Would you like me to update the Memory Bank?</question>
  <follow_up>
    <suggest>Yes</suggest>
    <suggest>No</suggest>
  </follow_up>
</ask_followup_question>

If the user responds "No", do end memory bank management automation.

If the user responds "Yes":

1.  Update memory bank. For each of the `Core Files` needed, use 
    `update_todo_list` to manage tracking and progress. 
2.  After the memory back is updated, ask the user to verify the content of the 
    files using `ask_followup_question`.

    <ask_followup_question>
      <question>Is the Memory Bank updated correctly and ready?</question>
      <follow_up>
      <suggest>Yes</suggest>
      <suggest>No</suggest>
      </follow_up>
    </ask_followup_question>
    
   If the user responds "No", ask them what needs to be changed and update the
   files accordingly. Continue this loop until the user responds "Yes".

