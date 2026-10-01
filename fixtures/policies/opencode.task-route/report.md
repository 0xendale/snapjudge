# snapjudge eval: `opencode.task-route`

Agreement is measured against the model each repo already uses, not against ground truth.

- Site: `agent:opencode:task-route`
- Status: completed
- Inputs: synthetic
- Reference: `supplied labels` (in code: none); reconstructed: no; teacher_assumed: no; teacher_override: no
- Jev model: `jev-1.13.0`
- Policy: fixture
- Evidence revision: `fce6fc21a716d0a73a2e071bd9576ecafd2f4c647d7e56f374062283ece7cb39`
- Dataset: `dataset.jsonl` SHA-256 `eaef6396b5f3129189fb27caded9133f0a680d3e9d26b77fe868974dec1c1047`

Sample size: 50/50 correct gives a 95% Wilson lower bound of 0.9287, so `--target 0` needs at least 1 accepted held-out rows, all agreeing; a measured policy also needs at least `--min-accepted 50` accepted held-out rows.

## Polished definition

Review the questions and the answer mapping Jev was measured with.

```json
{
  "schema_version": "1.0",
  "id": "opencode.task-route",
  "site_id": "agent:opencode:task-route",
  "definition_revision": "3a4bc3c29ff9860516a916cb407fb31d0cb77fd6e3045b234575415dc1bd8206",
  "input_schema": {
    "fields": [
      {
        "name": "task",
        "description": "The task text passed to the OpenCode dispatch tool",
        "kind": "string",
        "required": true
      }
    ]
  },
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Given the `task` text passed to the OpenCode dispatch tool, which specialist should take the first bounded step? Select the main requested action, not incidental words or background context.",
      "criteria": {
        "explore": "Locate or explain existing repository code, call paths, ownership or structure. No failure diagnosis or change review is requested.",
        "debug": "Reproduce, trace and fix a reported failure, crash, regression or incorrect runtime behavior. A failing test needing diagnosis belongs here.",
        "review": "Inspect existing code or a diff for defects, correctness, risks or regressions without being asked to implement a fix.",
        "research": "Gather and compare information outside this repository, including current documentation, external APIs or open-source examples.",
        "none_of_the_above": "The task asks for implementation, writing, design, deployment, administration, or another action rather than exploration, debugging, review or external research; use this also when the requested first action is not clear."
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
| 3 | 0.15 | 120/120 | 0.9583 (115/120; 95% CI [0.9062, 0.9821]; held_out: false; inputs: synthetic) |
| 4 | 0.2 | 120/120 | 0.9583 (115/120; 95% CI [0.9062, 0.9821]; held_out: false; inputs: synthetic) |
| 5 | 0.25 | 120/120 | 0.9583 (115/120; 95% CI [0.9062, 0.9821]; held_out: false; inputs: synthetic) |
| 6 | 0.3 | 119/120 | 0.9664 (115/119; 95% CI [0.9168, 0.9869]; held_out: false; inputs: synthetic) |
| 7 | 0.35 | 119/120 | 0.9664 (115/119; 95% CI [0.9168, 0.9869]; held_out: false; inputs: synthetic) |
| 8 | 0.4 | 118/120 | 0.9746 (115/118; 95% CI [0.9279, 0.9913]; held_out: false; inputs: synthetic) |
| 9 | 0.45 | 118/120 | 0.9746 (115/118; 95% CI [0.9279, 0.9913]; held_out: false; inputs: synthetic) |
| 10 | 0.5 | 117/120 | 0.9744 (114/117; 95% CI [0.9273, 0.9912]; held_out: false; inputs: synthetic) |
| 11 | 0.55 | 114/120 | 1.0000 (114/114; 95% CI [0.9674, 1.0000]; held_out: false; inputs: synthetic) |
| 12 | 0.6 | 114/120 | 1.0000 (114/114; 95% CI [0.9674, 1.0000]; held_out: false; inputs: synthetic) |
| 13 | 0.65 | 114/120 | 1.0000 (114/114; 95% CI [0.9674, 1.0000]; held_out: false; inputs: synthetic) |
| 14 | 0.7 | 113/120 | 1.0000 (113/113; 95% CI [0.9671, 1.0000]; held_out: false; inputs: synthetic) |
| 15 | 0.75 | 110/120 | 1.0000 (110/110; 95% CI [0.9663, 1.0000]; held_out: false; inputs: synthetic) |
| 16 | 0.8 | 106/120 | 1.0000 (106/106; 95% CI [0.9650, 1.0000]; held_out: false; inputs: synthetic) |
| 17 | 0.85 | 104/120 | 1.0000 (104/104; 95% CI [0.9644, 1.0000]; held_out: false; inputs: synthetic) |
| 18 | 0.9 | 98/120 | 1.0000 (98/98; 95% CI [0.9623, 1.0000]; held_out: false; inputs: synthetic) |
| 19 | 0.95 | 88/120 | 1.0000 (88/88; 95% CI [0.9582, 1.0000]; held_out: false; inputs: synthetic) |
| 20 | 1 | 50/120 | 1.0000 (50/50; 95% CI [0.9287, 1.0000]; held_out: false; inputs: synthetic) |

## Held-out agreement

- `all_required_agree`, gate-passing rows: 1.0000 (120/120; 95% CI [0.9690, 1.0000]; held_out: true; inputs: synthetic)
- Coverage: 120/120 = 1.0000 (teacher_failed 0, teacher_invalid 0, jev_failed 0, not_run 0)
- `all_required_agree`, every valid row (ungated): 1.0000 (120/120; 95% CI [0.9690, 1.0000]; held_out: true; inputs: synthetic)

| output | accepted rows | every valid row |
| --- | --- | --- |
| `route` (choice) | 1.0000 (120/120; 95% CI [0.9690, 1.0000]; held_out: true; inputs: synthetic) | 1.0000 (120/120; 95% CI [0.9690, 1.0000]; held_out: true; inputs: synthetic) |

## Reference self-agreement

not applicable

## Adjudication

not applicable

## Disagreements

None among valid held-out rows.

## Cost and latency

Held-out rows where Jev was attempted: 120 (0 deferred to the reference).

| path | USD | USD per input | mean latency | latency |
| --- | --- | --- | --- | --- |
| Jev only | $0.002647 | $0.000022 | 353 ms | measured |
| reference only | unknown | unknown | not available (120 rows unmeasured) | measured |
| cascade (modeled) | $0.002647 | $0.000022 | 353 ms | simulated |

Run cost: estimated $0.000242 (worst case $0.005221), actual $0.005296 (0 calls without a reported cost).

| role | model | price (USD) | source | usage basis |
| --- | --- | --- | --- | --- |
| jev | `jev-1.13.0` | $0.0420 per million input tokens | configuration | usage.input_tokens × input_price_usd_per_mtok |

Catalogue: https://openrouter.ai/api/v1/models retrieved 2026-09-30 (SHA-256 `2108e36f5882229d4a29e6048861803f3415f82f04a1bda495457f0933969146`).
