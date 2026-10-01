# extractive-cards: GPT judge vs Claude judge

Same 72 saved answers, same rubric; only the judge model differs.

| System | Judge | Correct | Partial | Incorrect | Unanswerable declined |
|---|---|---|---|---|---|
| baseline | gpt-5.6-luna | 23 | 3 | 4 | 6/6 |
| baseline | claude-opus-5-5 | 20 | 6 | 4 | 6/6 |
| moment | gpt-5.6-luna | 22 | 4 | 4 | 6/6 |
| moment | claude-opus-5-5 | 21 | 4 | 5 | 6/6 |

- Agreement: **93.1%** of 72 verdicts; Cohen's kappa **0.822**
- Headline (more answerable questions correct): GPT judge → **baseline**, Claude judge → **moment**; conclusion CHANGES
- Claude judge errors/refusals: 0

## Disagreements

| Question | System | Type | GPT | Claude | Claude's reason |
|---|---|---|---|---|---|
| e02 | baseline | explain | correct | partial | The answer covers distracting the user, the people nearby and the speaker, but it omits the key point that humans have only one language processor. |
| e04 | baseline | explain | correct | partial | It captures the idea of following one small taped piece and having each piece behave the same way, but it says the push is downward rather than sideways and omits that this is the right way to think about the problem. |
| e10 | baseline | explain | correct | partial | It gives the main reason (it implies the audience stayed only out of politeness), but it leaves out the advice to mouth thank you during the applause rather than making it the last move. |
| e10 | moment | explain | correct | partial | The answer gets the politeness point but leaves out that you can still mouth thank you once the applause starts. |
| l04 | moment | locate | partial | incorrect | The answer's 58:50 is about 35 seconds before the reference start of 59:25, which falls outside the roughly 30-second tolerance. |

## Leakage audit

The judge never sees system names, scores or sources. The answer text itself can still hint at its source; counts of answers containing each signal:

| Signal | Baseline answers | Moment RAG answers |
|---|---|---|
| timestamp (mm:ss) | 0 | 5 |
| Moment RAG gate refusal text | 0 | 4 |
| mentions 'excerpts' | 11 | 1 |
| mentions 'moment(s)' | 1 | 5 |
