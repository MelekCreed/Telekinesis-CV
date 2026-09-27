# AI Team Protocol (Claude Code + Codex CLI)

Governs how Claude Code and Codex CLI collaborate in this project. Both read this before doing
anything non-trivial. Codex is pointed here from AGENTS.md; Claude Code is pointed here from its
global CLAUDE.md.

## Roles
- Whichever agent the human is actively driving through for a task is the ORCHESTRATOR for that
  task. There is no fixed permanent lead.
- The orchestrator picks ONE implementation lead (writer) per task, based on actual fit for that
  task, not a fixed stereotype. The other agent is the consultant and stays READ-ONLY unless the
  human explicitly authorizes a separate-branch/worktree implementation.

## When to consult the other agent
Consult for: architecture/design decisions, security-sensitive changes, hard-to-diagnose bugs,
substantial diffs before merge, final review. Do NOT consult for routine/mechanical edits.

## Consultation budget
Default: one consultation + one rebuttal round per decision. If the two agents still materially
disagree after the rebuttal, stop looping them against each other and surface the disagreement to
the human with both sides' reasoning, instead of forcing a synthesized answer.

## How to consult (handoff format)
Always include when invoking the other agent:
- Objective
- Constraints
- Relevant files / current diff
- Decisions already made (and why)
- Tests and results so far
- Uncertainties or risks
- What kind of response is wanted (plan / critique / alternative / review)

Require the response to include: risks, recommendation, alternatives considered, confidence level.

## Edit ownership (hard rule)
Only the current writer may modify the working tree. The consultant is read-only for that task
unless the human explicitly says otherwise. If both agents need to implement independently for
comparison, each MUST work in its own git branch or worktree — never the same working tree at the
same time.

## Disclosure
The orchestrator must tell the human which agent wrote the code, which reviewed it, and which
verified/tested it. Never imply that a freshly invoked session (a subprocess `codex exec` / `claude`
call) is the human's currently-open conversation with the other agent — always say when a fresh,
context-less session was spun up. That session has no memory of prior chat history or unsaved
editor state unless it's explicitly included in the handoff.

## Safety limits
- No recursive/indefinite agent-to-agent delegation loops.
- No sandbox-bypass flags on either CLI unless the human explicitly asks for it for that specific
  call.
- Human approval is still required for deployments, destructive operations, purchases,
  credentials, or other consequential external actions — regardless of what either agent
  recommends.

## Shared task state
Use `.ai/` at the project root as the durable handoff record (create it when a project first needs
it):
```
.ai/current-task.md   — what's being worked on right now, by whom
.ai/decisions.md      — running log of material decisions + rationale
.ai/handoffs/         — one file per handoff, using the format above
```
Never put secrets/credentials in these files.

## Non-binding routing prior
Rough default only — override freely based on actual task fit and available context:

| Task | Lead | Consultant |
|---|---|---|
| Product spec / exploration | Claude | Codex critiques feasibility |
| Repo investigation / debugging | Codex | Claude challenges the diagnosis |
| Large multi-file implementation | whichever agent already has the most context | other reviews |
| Architecture decision | both plan independently | orchestrator synthesizes |
| Tests / security / regression review | non-author agent | author responds |
| Final review before merge | Codex | Claude (or human) resolves findings |
