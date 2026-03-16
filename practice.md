===============
Completed in previous Homework.

Stage 1:
Create a coding agent which will have access to such tools:


- read file list ("ls" )
- read file content ("cat <filename>)
- replace file content (replace full file content with a new value)
- edit file - replace part of file (file name, line start, line end, new content)
- search file by content "string" ("grep" )
- validate syntax tool (run type check, run compile check for language)

[optional]
- write logs in "logs" folder
- add token count functionality

TASK: implement coding agent which can rename function in the codebase
You can use repository "https://github.com/go-training/helloworld" 

Pay attention to error handling, edit file - should not break the syntax.
Error Handling Requirements:
- Code must compile and work 
- Test how it works if some files are readonly
- Test if code was broken originally.

==========================================================================================
===============
Completed in previous Homework.

Stage 2:
Extend your agent to enforce limits for agent operations.

1) Global timeout limit (e.g 20 seconds)
2) Global token limit (for example 20.000)
3) Tool call limit 100 calls per tool. (ls, cat, ...)
4) [Optional] Token rate limit.
5) [Optional] price limit*

Stage 3:
Run it on the big repository: Typescript.

https://github.com/Kilo-Org/kilocode 

https://github.com/anomalyco/opencode

Pick one folder for your choice like:
https://github.com/anomalyco/opencode/blob/dev/script/
https://github.com/anomalyco/opencode/blob/dev/infra/

You can pick another repository for your language of choice (golang/c++/rust/python etc.)

Task: 
In this folder rename all variables/consts to have a suffix "_changed".
Example const name "branch" rename it to "branch_changed"


=======================
Homework 3
Stage 4: 

Add new tools into our agent (must be just 2 tool): 
 - generate json output
 - generate json output with strict json schema 

Add validations to LLM output (Do we need them as tools? - NO) 
 - validate json output
 - validate json output with strict json schema


Stage 5: 
Execute task: analyse codebase and gather info in form of JSON, store it in file.
  - statistics: 
    - total lines of code
    - total character count
    - total number of functions

  - list of functions, their parameters, param types, return type
  - list of files and their imports, exports, function names.

Give this statistics im 3 forms:
 - do not state JSON output as query param
 - state structured output JSON
 - state structured output JSON with strict JSON Schema.


Optional task : run "Give this statistics im 3 forms" 3-5 times .


===================

Homework 4 Evaluation


Add evaluation to "rename variable task"
Add evaluation to "generate json output with strict schema"

Use each of these graders:
Required:
  - Rubric-based scoring (scale score)
  - Natural language assertions ("Was response polite?")
  - Pairwise comparison (compare two responses)

Optional:  
  - Reference-based evaluation (compare response to a known good one)
  - Multi-judge consensus (get multiple LLM-judges opinions)

Notes: consult with lesson Evaluation "agentic_ai/study_topics/8_evaluation/theory/dima/examples/custom"

====================


Homework 5 Metrics Observability:

- Convert our homework 1-4 into langchain (if it's already not in langchain)
- Create account on Langsmith and store traces  in it.
- Observe how your Homework 4 is executed in the Langsmith observability


