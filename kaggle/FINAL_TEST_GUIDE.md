# Frozen XLM-R final test: operator guide

The Stage 2 validation runs selected XLM-R for Bangla and English before test access. This final evaluation uses those exact six CPU checkpoints (seeds 42, 123, and 2026 for each language), with their SHA-256 hashes frozen in `corrected_pipeline/final_test.py`. It performs **inference only**. The Stage 2 train/validation configurations stay locked.

## Set up once

1. Extract the companion `final_test_additions.zip` into your existing `D:\Capstone\Capstone-Project` checkout on the `experimental-protocol` branch. It adds files; check `git status` before committing.
2. Run `python -m unittest discover -s tests_stage1a -v` on a Python machine, if available. The Kaggle notebooks also run these tests before inference.
3. Commit and push the added files to the **same** `experimental-protocol` branch. The Kaggle launchers clone that branch; they must be able to find `corrected_pipeline/final_test.py`.
4. Have the six already trained XLM-R checkpoint files accessible as **Kaggle notebook inputs**. You may attach saved Stage 2 notebook output versions if they contain the original `best_checkpoint_cpu.pt`, or upload the checkpoint files to a private Kaggle dataset. No checkpoint needs to be sent to this chat. The evaluator checks every checkpoint against the frozen hash and rejects a wrong file.

## Run Bangla, then English

1. Import `kaggle/14_final_test_bn_xlmr.ipynb` into a **new Kaggle notebook session**. Attach all three Bangla checkpoints as inputs. Enable a GPU in Kaggle Settings.
2. Run the first cell. It clones the updated branch. Run the next cell with `RUN_FINAL_TEST = False` first to see the frozen selections.
3. Copy each checkpoint's real `/kaggle/input/.../best_checkpoint_cpu.pt` path from the Kaggle Input sidebar into `CHECKPOINT_PATHS[42]`, `[123]`, and `[2026]`. The three paths must refer to three different trained checkpoints.
4. Set `RUN_FINAL_TEST = True` and choose **Save Version → Save & Run All**. Wait for all cells to complete. The notebook checks the source and checkpoint hashes, validates the full audited train/validation/test CSVs, and produces results for all three seeds. It creates `corrected_final_test_bn_xlmr_results.zip` in `/kaggle/working`.
5. Download this results ZIP. Keep the original checkpoints untouched. The results ZIP contains metrics, row predictions, data checks, and provenance; it does not contain the 1 GB checkpoints.
6. Repeat steps 1–5 with `kaggle/15_final_test_en_xlmr.ipynb` and the three English checkpoints. Download `corrected_final_test_en_xlmr_results.zip`.

Share only the two **small results ZIPs** for aggregation and the final report. If a run fails, keep the full error text and do not modify the checkpoint, the Stage 2 config, or the test hash to force it to pass.

## Evidence and scope

The result `sarcasm_f1` measures class 2 (Sarcastic) against the other two main classes; it is not an independently annotated, potentially overlapping sarcasm axis. The gold multi-task annotation join is still blocked for lack of stable one-to-one IDs. The final results are for the *already trained* corrected single-task XLM-R model; do not describe them as a newly improved XLM-R architecture. The mDistilBERT vs XLM-R comparisons available so far use Stage 2 **validation**, not final test, unless mDistilBERT is evaluated separately under a similarly frozen test procedure.
