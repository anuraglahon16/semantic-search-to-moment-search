# llm-cards: GPT judge vs Claude judge

Same 72 saved answers, same rubric; only the judge model differs.

| System | Judge | Correct | Partial | Incorrect | Unanswerable declined |
|---|---|---|---|---|---|
| baseline | gpt-5.6-luna | 21 | 6 | 3 | 6/6 |
| baseline | claude-opus-5-5 | 19 | 5 | 6 | 6/6 |
| moment | gpt-5.6-luna | 24 | 5 | 1 | 6/6 |
| moment | claude-opus-5-5 | 24 | 4 | 2 | 6/6 |

- Agreement: **88.9%** of 72 verdicts; Cohen's kappa **0.7**
- Headline (more answerable questions correct): GPT judge → **moment**, Claude judge → **moment**; conclusion does not change
- Claude judge errors/refusals: 0

## Disagreements

| Question | System | Type | GPT | Claude | Claude's reason |
|---|---|---|---|---|---|
| e04 | baseline | explain | correct | partial | The answer correctly focuses on the small taped section and notes that each piece behaves the same way, but it misstates the mechanism: the reference says blowing on the piece as it comes over the top pushes it sideways, not that it keeps moving downward. |
| e10 | baseline | explain | correct | partial | It captures the politeness/wanting-to-be-elsewhere point but omits that you can mouth thank you once applause starts rather than making it your last move. |
| e10 | moment | explain | correct | partial | The answer gets the main point that it implies the audience stayed only out of politeness, but it leaves out that you can mouth thank you once the applause starts, just not as your final move. |
| l01 | baseline | locate | partial | incorrect | The answer gives no timestamp, while the reference places the start at about 23:35-24:15. |
| l01 | moment | locate | partial | incorrect | The answer gives 24:58, which is more than 30 seconds past the reference start of about 23:35-24:15. |
| l02 | baseline | locate | partial | incorrect | The answer gives no timestamp, while the reference places the demonstration at about 18:30-19:40, and it adds an unsupported claim that the audience can try the prop afterward. |
| l02 | moment | locate | partial | correct | The given start time of 18:26 is within 30 seconds of the reference's 18:30, though the stated end time runs about a minute past the reference range. |
| l03 | baseline | locate | partial | incorrect | The answer declines to give a location, but the reference places the job talks discussion at about 43:55-45:00. |

## Leakage audit

The judge never sees system names, scores or sources. The answer text itself can still hint at its source; counts of answers containing each signal:

| Signal | Baseline answers | Moment RAG answers |
|---|---|---|
| timestamp (mm:ss) | 0 | 4 |
| Moment RAG gate refusal text | 0 | 4 |
| mentions 'excerpts' | 11 | 0 |
| mentions 'moment(s)' | 1 | 4 |
