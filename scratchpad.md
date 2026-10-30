# Prompt: Design A Two-LLM Planning And Consensus Workflow

I want to build a workflow that lets two coding LLMs collaborate without me manually prompting every turn.

Current setup:

- I use opencode with GPT 5.5.
- I also use Claude Code with Opus 4.7.
- My current workflow is manual: I discuss a plan with one model, then copy/prompt the other model to review it, then copy the response back, and so on.
- This makes me the bottleneck because every interaction requires me to manually coordinate the next step.

Desired goal:

Design a system or workflow where both LLMs can independently reason about a task, review each other's ideas, respond to critiques, and attempt to form a consensus plan. It is acceptable if they fail to reach consensus, as long as the disagreement is clearly summarized for me to review.

The workflow should support these phases:

1. The user provides the original task/brief.
2. GPT 5.5 and Claude Opus 4.7 each produce independent plans without seeing each other's work.
3. Each model reviews the other model's independent plan.
4. Each model responds to the critique of its own plan.
5. The models continue for a bounded number of rounds if useful.
6. The workflow stops when they reach consensus, fail to reach consensus, ask for human input, hit a max round limit, or I interrupt the process.
7. A final summary is produced with the consensus, unresolved disagreements, reasoning, and recommended human decisions.

Important requirements:

- Both models must form independent initial thoughts before seeing each other's plans.
- The system must preserve a historical record of the whole interaction.
- The record should be readable by a human, not just raw JSON.
- The log should show the sequence clearly: independent plans, first cross-reviews, counter-responses, later rounds, consensus attempt, final summary.
- The models should explain their reasoning, especially for complex topics.
- The system should make disagreement explicit instead of forcing fake consensus.
- I need to be able to monitor the interaction while it is happening.
- I need to be able to pause, stop, or interrupt the interaction if it is going in the wrong direction.
- Planning/review mode should be read-only by default. The models should not edit files or run implementation steps unless explicitly allowed.

Please help design this workflow.

I want you to think through:

- Whether this should be implemented as a wrapper/orchestrator script around both CLIs, an opencode plugin/command, a Claude/opencode skill setup, or some other architecture.
- How to call both models in a non-interactive or automatable way.
- How to structure the prompts for independent planning, cross-review, counter-response, consensus checking, and final summary.
- How to log everything to files in a readable historical format.
- How to support live monitoring and user interruption.
- How to detect consensus, partial consensus, repeated disagreement, or failure to converge.
- How to avoid the two models simply agreeing with each other too easily.
- How to keep the workflow safe, especially when the task is only planning or review.
- What an MVP should look like, and what can be deferred.

Please produce a practical design with clear implementation steps.

Assume this will run on a local developer machine and may be used inside project repositories. Prefer simple, robust mechanisms over complicated infrastructure. File-based logs and control files are acceptable if they make the system easier to build and inspect.

Potential output structure:

1. Recommended architecture.
2. Debate protocol.
3. File and folder layout.
4. Prompt templates for each phase.
5. Logging format.
6. Monitoring/interruption mechanism.
7. Consensus/failure detection.
8. Safety and permission model.
9. MVP implementation plan.
10. Risks, open questions, and alternatives.
