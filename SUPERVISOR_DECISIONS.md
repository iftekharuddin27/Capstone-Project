# Supervisor Decisions

Recorded for the corrected protocol on 2026-09-08. These decisions supersede unresolved planning assumptions without rewriting historical audit reports.

| Decision | Status | Protocol consequence |
|---|---|---|
| Main class ID `2` means **Sarcastic** | Confirmed | The canonical mapping is `0 = Non-hateful`, `1 = Hateful`, `2 = Sarcastic`. |
| Hate and sarcasm are overlapping axes | Confirmed | A record may be positive on both auxiliary axes; mutually exclusive derivation is invalid. |
| Human annotation columns | Confirmed as `hate_type` and `sarcasm_type` | Candidate files must still pass provenance and stable-join validation before use as gold targets. |
| Bangla GRL/domain-adversarial and leave-one-source-out work | Approved in principle | All five verified sources must be used: `ALERT`, `BD_SHS`, `BenSarc`, `BanglaSarc3`, and `BIDWESH`. |
| Reportable seed choice | Delegated to protocol | Stage 2 uses the predeclared seeds `42`, `123`, and `2026`. |
| Test access | Locked | Test data must not be read, evaluated, or used for selection until the model configuration is finalized. |

Approval of the auxiliary task does not establish a safe row-level join. `ANNOTATION_AUDIT.md` documents why gold multi-task execution remains blocked.
