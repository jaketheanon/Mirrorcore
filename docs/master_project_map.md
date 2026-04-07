# MirrorCore Master Project Map

## 1. Project identity

MirrorCore is a deterministic, local-first CLI system designed to learn how the user actually thinks, decides, responds, and debugs.

It is not meant to become a generic chatbot or a vague “assistant.”  
It is meant to become a grounded, testable, user-specific reasoning and response engine that can:

- retrieve relevant past situations
- predict likely responses and decisions
- preserve user-corrected phrasing
- learn safely from feedback
- resist contamination from weak or repetitive memory
- become more realistic over time without becoming less truthful

---

## 2. Long-term end goal

The long-term goal of MirrorCore is a reliable “respond / think / decide like me” system that is:

- local-first
- deterministic where possible
- memory-grounded
- replay-stable
- contradiction-aware
- explainable
- safe to iterate on
- capable of realistic user-specific simulation later

The system should eventually support:

- likely-response generation
- likely-decision modeling
- scenario continuity across repeated situations
- user-specific style and tone fidelity
- safe correction reuse
- deeper conversational and behavioral simulation
- optional GUI / dashboard tooling later without compromising the CLI core

---

## 3. Core principles

These are project-level invariants unless explicitly replaced by a future phase.

### Local-first
The system should work locally and not depend on cloud services for core behavior.

### Deterministic where possible
If logic can be explicit and testable, prefer that over vague probabilistic behavior.

### Debug truthfulness over pretty output
If the system is uncertain, mixed, blocked, or inferring, debug output must reflect that honestly.

### Memory safety over aggressive learning
It is better to under-learn than to poison memory with weak or misleading examples.

### User-corrected lines are high-value evidence
If a user says the action was right but the wording was wrong, that correction must not be casually overwritten by older stitched output.

### Same-thread carryover must stay narrow
Carryover should only generalize when the system has strong evidence that the new prompt is materially the same thread.

### Style should emerge from evidence
The project should not fake personality with decorative writing. Style should come from repeated evidence, correction, and stable behavior patterns.

### Replay stability matters
If a scenario is re-run with the same or obviously related wording, the system should remain stable unless there is a strong reason to change.

### Every phase must be verified
A phase is not done because code changed. A phase is done after:
- tests pass
- manual prompts pass
- debug truthfulness is verified
- contamination risk is checked where relevant

---

## 4. Current project interpretation

MirrorCore has evolved in three major tracks so far:

### A. Troubleshooting intelligence track
Early phases focused on:
- incident detection
- retrieval relevance
- historical fix gating
- pattern learning
- user weighting
- decision model foundations

### B. User modeling and response track
Middle phases focused on:
- interview capture
- style calibration
- personal response generation
- response realism
- conflict realism
- contradiction handling
- continuity across repeated situations

### C. Replay / feedback / wording reliability track
Recent phases focused on:
- escalation hardening
- corrected wording reuse
- replay stability
- same-thread generalization
- output cleanup
- response sharpness

The current direction is:
- preserve deterministic, grounded behavior
- improve respond-like-me
- strengthen memory safety
- gradually move from “likely response” into deeper user-specific behavioral modeling

---

## 5. Current system capabilities

MirrorCore currently supports, in some form:

- memory-backed likely-response generation
- short-term situation carryover
- user feedback on response correctness
- corrected wording reuse in some replay / same-thread cases
- confidence labels and short reasoning summaries
- deterministic CLI workflows
- pytest coverage for major personal-response and carryover behavior
- replay-safe improvements in recent branches
- style realism work in personal-response scenarios
- conflict / boundary handling improvements
- contradiction-aware confidence behavior
- response polishing and wrapper cleanup

---

## 6. Current weak spots

Known recurring risks and unfinished areas:

- some outputs can still sound too soft or generic
- confidence and phrasing are not always fully aligned
- style fidelity is improving but not yet deeply user-specific
- memory row clutter can still accumulate
- thread lifecycle resolution is still incomplete
- contradiction handling may still need broader consistency checks
- response sharpness still needs tuning beyond cleanup
- bundle-level verification workflow is not yet automated enough
- long-term memory compression / synthesis still needs architecture work

---

## 7. Stable behavior that must be preserved

Unless a future phase explicitly replaces them, these behaviors should remain true:

- exact replay should remain stable after correction
- same-thread variants should reuse corrected stance when appropriate
- unrelated prompts should not inherit corrected lines
- cross-domain prompts should remain uncontaminated
- debug fields should remain truthful
- confidence should reflect actual grounding, not cosmetic certainty
- user-approved replacement lines must not be silently degraded
- replay logic must not self-poison short-term memory
- recent output cleanup must not reintroduce duplicate wrappers or redundant second-line phrasing

---

## 8. Phase history

Note:
- These entries reflect the current repo phase branches plus the known intent of the later phases as developed.
- Stable branches / tags are checkpoints, not separate behavior phases.
- phase24-pre-claude-backup is a checkpoint, not a feature phase.
- Phase 49 is active / in progress.

### Phase 21
Purpose: verified pre-export baseline  
What it did: established a checked baseline before later development accelerated.

### Phase 22
Purpose: auto-resolve session handling  
What it did: automatically resolved debugging sessions on successful outcomes so stale sessions did not remain open.

### Phase 22.5
Purpose: improved incident detection  
What it did: strengthened incident detection and recognition.

### Phase 22.6
Purpose: distinguish runtime validation errors  
What it did: separated runtime validation failures from other classes of incidents for better classification.

### Phase 23
Purpose: retrieval relevance  
What it did: improved similar-investigation retrieval relevance so historical results were more useful and less noisy.

### Phase 23.5
Purpose: escalation signal cleanup  
What it did: cleaned duplicate escalation output and reduced signal noise.

### Phase 24
Purpose: pattern learning and historical gating  
What it did: introduced / strengthened cross-session fix-pattern learning and tightened historical fix gating.

### Phase 24 pre-Claude backup
Purpose: checkpoint only  
What it did: preserved a recovery point before more changes. Not a feature phase.

### Phase 25
Purpose: user-specific fix weighting  
What it did: added user-specific fix preference weighting so past successful patterns for this user could influence ranking.

### Phase 25.5
Purpose: brain tightening / guardrails  
What it did: tightened weighting behavior so user preference influenced results without overriding correctness.

### Phase 26
Purpose: decision model  
What it did: introduced or formalized the decision-model layer.

### Phase 27
Purpose: human interview  
What it did: strengthened interview-driven user modeling and preference capture.

### Phase 28
Purpose: style calibration  
What it did: built style-calibration foundations and persona-memory grounding.

### Phase 29
Purpose: personal response  
What it did: added the unified personal-response layer and moved toward “respond like me” behavior.

### Phase 30
Purpose: guided onboarding  
What it did: improved guided onboarding so the system could be shaped more intentionally.

### Phase 30.1
Purpose: interview accuracy  
What it did: tightened interview/session handling and improved preference capture reliability.

### Phase 31
Purpose: conversational router  
What it did: improved routing so user input could be directed into the right subsystem more reliably.

### Phase 32
Purpose: quiet routing  
What it did: reduced noisy routing behavior and improved relevance / gating.

### Phase 33
Purpose: decision intelligence  
What it did: introduced decision ontology and adaptive slot logic for more structured reasoning.

### Phase 34
Purpose: memory relevance  
What it did: tightened memory relevance and repetition control.

### Phase 35
Purpose: response polish  
What it did: improved wording polish and reduced repeated or clumsy guidance.

### Phase 36
Purpose: cross-system integration  
What it did: improved integration across the decision, memory, and response layers.

### Phase 37
Purpose: natural response  
What it did: made likely-user responses feel more natural and less obviously assembled.

### Phase 38
Purpose: active learning  
What it did: added the active-learning loop for likely-response feedback and correction.

### Phase 39
Purpose: action-wording split  
What it did: separated likely action from likely wording.

### Phase 40
Purpose: conflict / boundary depth  
What it did: deepened conflict handling and boundary-setting realism and improved how feedback influenced those outputs.

### Phase 41
Purpose: style realism  
What it did: improved realism in style, especially around conflict and phrasing.

### Phase 42
Purpose: example memory  
What it did: promoted repeated corrections into stronger reusable example memory.

### Phase 43
Purpose: contradiction calibration  
What it did: improved contradiction handling and confidence calibration.

### Phase 44
Purpose: session continuity  
What it did: added escalation-aware continuity and same-session carryover behavior.

### Phase 45
Purpose: escalation hardening  
What it did: hardened escalation-aware continuity, improved carryover behavior in repeated-boundary scenarios, and made replacement feedback matter more in surfaced wording.

### Phase 46
Purpose: feedback prioritization  
What it did: made corrected wording and replacement feedback outrank older awkward phrasing more consistently.

### Phase 47
Purpose: replay hardening  
What it did: stabilized exact replay, improved same-thread variant behavior, preserved corrected wording across repeated prompts, and reduced replay self-poisoning.

### Phase 48
Purpose: response polish, v2  
What it did: cleaned surfaced answer assembly, removed duplicate wrappers and redundant second-line phrasing, and improved readability without changing decision logic.

### Phase 49
Purpose: response sharpness  
What it is intended to do: make surfaced responses firmer, less hedge-heavy, and more intentional when confidence and grounding support that tone, without changing retrieval, carryover, replay, escalation, or debug truthfulness.  
Status: active / in progress

---

## 9. Stable checkpoints and branch conventions

### Stable checkpoints
Stable branches and tags such as phaseXX-stable are milestone checkpoints.  
They are not separate feature phases unless they introduced behavior.

### Phase branch convention
Each phaseXX-* branch should represent:
- one focused phase
- one clear goal
- one bounded set of changes
- one test + manual verification cycle

### Main branch strategy
main does not need to move after every phase.  
It should represent a larger stable bundle when a set of related phases is truly worth promoting.

### Chained phase development
Phase branches can chain from previous phase branches when building a bundle.  
This is acceptable as long as:
- each phase still has a distinct goal
- each phase is verified independently
- the bundle can later be merged intentionally

---

## 10. Required workflow for each phase

Each phase should follow this structure:

1. define the goal
2. define what changes
3. define what must not change
4. define failure modes
5. implement
6. run tests
7. run manual prompts
8. inspect debug truthfulness
9. inspect contamination risk if memory was touched
10. commit the phase branch
11. push the phase branch
12. do not merge to main until the larger bundle is ready

### Required per-phase contract
Each phase should explicitly include:
- goal
- what changes
- what must not change
- failure modes
- required tests
- required manual prompts
- debug fields that matter
- done criteria

---

## 11. Active phase section

### Current phase
Phase 49: response sharpness

### Goal
Make surfaced responses sound sharper, more decisive, and more intentional without changing the underlying decision logic.

### What this phase changes
- final surfaced wording
- hedge reduction when confidence and grounding support it
- phrasing firmness / directness
- cleaner expression of already-selected stance

### What this phase must not change
- retrieval scoring
- carryover thresholds
- escalation logic
- replay protection logic
- debug truthfulness
- confidence calculations unless explicitly justified
- corrected user-approved line fidelity

### Known failure modes
- “I’d likely…” when the answer is already clear
- “I’d probably…” when not needed
- soft, floaty verbs where direct phrasing would work
- moderate-confidence answers that still sound timid
- low-confidence answers becoming falsely certain
- corrected lines being “cleaned up” into less faithful wording

### Required tests
At minimum:
- sharper obligation wording without changing meaning
- sharper spending wording when confidence is moderate
- low-confidence responses remain appropriately cautious
- corrected stance lines remain intact
- replay-stable corrected outputs still stay stable
- no reintroduction of duplicate wrappers or redundant second-line phrasing

### Required manual prompts
- my coworker is pushing again after i already told them no
- same coworker is still pushing after i already said no
- i want to buy something fun but rent is due tomorrow and i am short on money
- my coworker asked me again today to cover a shift but this time i might be able to help later

### Done criteria
- outputs feel sharper and more intentional
- moderate-confidence answers sound less hesitant
- low-confidence answers do not become fake-confident
- corrected lines remain faithful
- tests pass
- manual prompts read better without changing core decision behavior

---

## 12. Next planned phase sequence

This is the recommended next bundle after Phase 49.

### Phase 50: confidence-to-tone calibration
Make tone strength align better with confidence level.

### Phase 51: style fidelity layer
Improve whether outputs feel specifically like the user, not just generically cleaner.

### Phase 52: contradiction and self-consistency hardening
Reduce conflicting outputs across similar prompts and repeated scenarios.

### Phase 53: thread lifecycle and resolution
Teach the system to better distinguish ongoing, resolved, reopened, and changed threads.

### Phase 54: contamination resistance v2
Strengthen stale-row suppression, replay-safe refresh rules, and memory safety.

### Phase 55: feedback interpretation upgrade
Improve understanding of what kind of correction the user gave:
- wrong action
- right action, wrong wording
- right direction, wrong strength
- right stance, wrong tone

### Phase 56: example promotion hardening
Make repeated corrections promote into reusable examples more safely and selectively.

### Phase 57: preference clustering
Learn higher-level stable tendencies from repeated behavior across scenarios.

### Phase 58: counterfactual / alternative response modeling
Model not only the chosen response but plausible rejected alternatives and why they are less “you.”

### Phase 59: replay harness / evaluation pack
Create reusable manual replay packs and stronger regression tooling.

### Phase 60: regression suite expansion
Broaden tests to cover tone, consistency, contradiction handling, and style fidelity more deeply.

---

## 13. Definition of success

MirrorCore is succeeding if:

- outputs are grounded in real stored evidence
- corrected lines persist correctly
- same-thread behavior is stable
- unrelated prompts stay uncontaminated
- style becomes recognizably user-specific
- confidence matches grounding honestly
- the system becomes more useful without becoming less truthful
- manual checks increasingly match lived expectations
- new phases improve realism without quietly breaking the earlier contract

---

## 14. How Cursor should use this file

This file should be treated as project-level source-of-truth for:

- project end goal
- major system direction
- completed phase context
- invariants that must not break
- current active-phase interpretation
- next-phase roadmap

Future implementation prompts should explicitly instruct Cursor to read this file before making changes.

Suggested bootstrap header:

`text
Before making changes, read docs/master_project_map.md and docs/current_phase.md and use them as the source of truth for:
- project end goal
- completed phase context
- invariants that must not break
- active phase expectations
- next-step intent

Do not infer project intent from code alone if these docs are more explicit.