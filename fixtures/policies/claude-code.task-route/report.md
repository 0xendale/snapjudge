# snapjudge eval: `claude-code.task-route`

Agreement is measured against the model each repo already uses, not against ground truth.

- Site: `agent:claude-code:task-route`
- Status: completed
- Inputs: synthetic
- Reference: `supplied labels` (in code: none); reconstructed: no; teacher_assumed: no; teacher_override: no
- Jev model: `jev-1.13.0`
- Policy: fixture
- Evidence revision: `427a81a8d1ee6aa6f2605b529f89f360e55fdb697fc143ae3ee1a5ec704b91c0`
- Dataset: `dataset.jsonl` SHA-256 `f004fcf085b88f15047e8916d029b4851ce4ab0d07cd70658dc0072c9abe01fb`

Sample size: 50/50 correct gives a 95% Wilson lower bound of 0.9287, so `--target 0` needs at least 1 accepted held-out rows, all agreeing; a measured policy also needs at least `--min-accepted 50` accepted held-out rows.

## Polished definition

Review the questions and the answer mapping Jev was measured with.

```json
{
  "schema_version": "1.0",
  "id": "claude-code.task-route",
  "site_id": "agent:claude-code:task-route",
  "definition_revision": "df8f4acf4a044d327cc75545d19b0564d33d49cb74e63e869ca35ad54f414aa6",
  "input_schema": {
    "fields": [
      {
        "name": "prompt",
        "description": "The allowlisted user prompt from Claude Code UserPromptSubmit",
        "kind": "string",
        "required": true
      }
    ]
  },
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Read the `prompt` received at Claude Code UserPromptSubmit. Which advisory route best matches the user's immediate request? Choose only one route, and do not treat an incidental mention of a bug, file or link as an instruction.",
      "criteria": {
        "explore": "User wants to find, understand or trace current repository code; there is no request to troubleshoot broken behavior or assess a change.",
        "debug": "User reports incorrect behavior or a failure and asks for diagnosis or repair, including a reproducible regression.",
        "review": "User requests a critical inspection of code, a patch or a pull request to identify defects, with findings as the deliverable.",
        "research": "User asks to consult external documentation, compare outside technologies or investigate third-party examples.",
        "none_of_the_above": "User wants a new feature, prose, refactoring without a failure, operations, or anything not primarily covered by the four advisory routes; includes unclear requests."
      }
    }
  },
  "outputs": [
    {
      "shape": "choice",
      "name": "route",
      "question": "route",
      "required": true
    }
  ]
}
```

## Inputs

| split | unique | valid | teacher_failed | teacher_invalid | jev_failed | not_run |
| --- | --- | --- | --- | --- | --- | --- |
| calibration | 120 | 120 | 0 | 0 | 0 | 0 |
| held-out | 120 | 120 | 0 | 0 | 0 | 0 |

240 rows, 0 invalid, 0 duplicates merged.

## Gate

Threshold t = 0/20 = 0 on every output and label (lowest calibration threshold reaching the target 0).

| k | t | calibration coverage | calibration agreement |
| --- | --- | --- | --- |
| 0 | 0 | 120/120 | 0.9583 (115/120; 95% CI [0.9062, 0.9821]; held_out: false; inputs: synthetic) |
| 1 | 0.05 | 120/120 | 0.9583 (115/120; 95% CI [0.9062, 0.9821]; held_out: false; inputs: synthetic) |
| 2 | 0.1 | 120/120 | 0.9583 (115/120; 95% CI [0.9062, 0.9821]; held_out: false; inputs: synthetic) |
| 3 | 0.15 | 119/120 | 0.9664 (115/119; 95% CI [0.9168, 0.9869]; held_out: false; inputs: synthetic) |
| 4 | 0.2 | 119/120 | 0.9664 (115/119; 95% CI [0.9168, 0.9869]; held_out: false; inputs: synthetic) |
| 5 | 0.25 | 119/120 | 0.9664 (115/119; 95% CI [0.9168, 0.9869]; held_out: false; inputs: synthetic) |
| 6 | 0.3 | 119/120 | 0.9664 (115/119; 95% CI [0.9168, 0.9869]; held_out: false; inputs: synthetic) |
| 7 | 0.35 | 119/120 | 0.9664 (115/119; 95% CI [0.9168, 0.9869]; held_out: false; inputs: synthetic) |
| 8 | 0.4 | 118/120 | 0.9746 (115/118; 95% CI [0.9279, 0.9913]; held_out: false; inputs: synthetic) |
| 9 | 0.45 | 118/120 | 0.9746 (115/118; 95% CI [0.9279, 0.9913]; held_out: false; inputs: synthetic) |
| 10 | 0.5 | 118/120 | 0.9746 (115/118; 95% CI [0.9279, 0.9913]; held_out: false; inputs: synthetic) |
| 11 | 0.55 | 114/120 | 1.0000 (114/114; 95% CI [0.9674, 1.0000]; held_out: false; inputs: synthetic) |
| 12 | 0.6 | 113/120 | 1.0000 (113/113; 95% CI [0.9671, 1.0000]; held_out: false; inputs: synthetic) |
| 13 | 0.65 | 113/120 | 1.0000 (113/113; 95% CI [0.9671, 1.0000]; held_out: false; inputs: synthetic) |
| 14 | 0.7 | 112/120 | 1.0000 (112/112; 95% CI [0.9668, 1.0000]; held_out: false; inputs: synthetic) |
| 15 | 0.75 | 110/120 | 1.0000 (110/110; 95% CI [0.9663, 1.0000]; held_out: false; inputs: synthetic) |
| 16 | 0.8 | 108/120 | 1.0000 (108/108; 95% CI [0.9657, 1.0000]; held_out: false; inputs: synthetic) |
| 17 | 0.85 | 107/120 | 1.0000 (107/107; 95% CI [0.9653, 1.0000]; held_out: false; inputs: synthetic) |
| 18 | 0.9 | 101/120 | 1.0000 (101/101; 95% CI [0.9634, 1.0000]; held_out: false; inputs: synthetic) |
| 19 | 0.95 | 94/120 | 1.0000 (94/94; 95% CI [0.9607, 1.0000]; held_out: false; inputs: synthetic) |
| 20 | 1 | 65/120 | 1.0000 (65/65; 95% CI [0.9442, 1.0000]; held_out: false; inputs: synthetic) |

## Held-out agreement

- `all_required_agree`, gate-passing rows: 0.9583 (115/120; 95% CI [0.9062, 0.9821]; held_out: true; inputs: synthetic)
- Coverage: 120/120 = 1.0000 (teacher_failed 0, teacher_invalid 0, jev_failed 0, not_run 0)
- `all_required_agree`, every valid row (ungated): 0.9583 (115/120; 95% CI [0.9062, 0.9821]; held_out: true; inputs: synthetic)

| output | accepted rows | every valid row |
| --- | --- | --- |
| `route` (choice) | 0.9583 (115/120; 95% CI [0.9062, 0.9821]; held_out: true; inputs: synthetic) | 0.9583 (115/120; 95% CI [0.9062, 0.9821]; held_out: true; inputs: synthetic) |

## Reference self-agreement

not applicable

## Adjudication

not applicable

## Disagreements

| input | tags | gate confidence |
| --- | --- | --- |
| `90ebf28aa3ba` | confident_miss | 0.2700 |
| `a7919887675a` | confident_miss | 0.1900 |
| `e1a00e57621f` | confident_miss | 0.4600 |
| `f4402f68e580` | confident_miss | 0.3700 |
| `fd1bba2ef9e9` | confident_miss | 0.7200 |

## Cost and latency

Held-out rows where Jev was attempted: 120 (0 deferred to the reference).

| path | USD | USD per input | mean latency | latency |
| --- | --- | --- | --- | --- |
| Jev only | $0.002651 | $0.000022 | 351 ms | measured |
| reference only | unknown | unknown | not available (120 rows unmeasured) | measured |
| cascade (modeled) | $0.002651 | $0.000022 | 351 ms | simulated |

Run cost: estimated $0.000242 (worst case $0.005272), actual $0.005303 (0 calls without a reported cost).

| role | model | price (USD) | source | usage basis |
| --- | --- | --- | --- | --- |
| jev | `jev-1.13.0` | $0.0420 per million input tokens | configuration | usage.input_tokens × input_price_usd_per_mtok |

Catalogue: https://openrouter.ai/api/v1/models retrieved 2026-09-30 (SHA-256 `2108e36f5882229d4a29e6048861803f3415f82f04a1bda495457f0933969146`).
