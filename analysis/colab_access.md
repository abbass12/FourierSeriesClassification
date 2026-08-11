# Google Colab GPU Replication Access

**Checked:** 2026-08-11

The full experiment notebook is publicly accessible at:

https://colab.research.google.com/github/abbass12/FourierSeriesClassification/blob/main/notebooks/Full_Experiment_Colab.ipynb

The page advertises an available T4 runtime, but Google requires the account holder to sign in before connecting and executing the notebook. A user-authenticated browser step is therefore required to run the full GPU replication.

No credentials have been entered or requested in this project workspace. Once the user signs in and connects the runtime, the experiment script can be run from the already-open notebook.

## Follow-up check

After the initial user handoff, the Colab page still displayed the `Sign in` control and did not show an authenticated Google session. The browser had reached the university identity-provider settings page, but that alone did not complete a Google Colab login. A completed Google account session is still required before the runtime can be used.

## Authentication and runtime status

The Google Colab page now shows an authenticated Google account session. A T4 connection was initiated on 2026-08-11 and the interface reported that the runtime was allocating. The notebook currently displayed was created before the latest local reproducibility and baseline changes; it must be refreshed or replaced with the current repository state before running a confirmatory experiment.

## Current confirmatory notebook

The versioned `Confirmatory_Validation_Colab.ipynb` is now open with the authenticated account, and a T4 connection was initiated. At the latest check, Colab reported that it was allocating the runtime. The notebook uses a shallow Git clone of the current `main` branch so future execution will retrieve commit `2991445` or later.

## Session-capacity block

Colab displayed a `Too many sessions` dialog when allocating the current T4 runtime. The dialog offers a session-management action that could terminate existing user sessions. No existing session was terminated, because that could disrupt unrelated work. The GPU experiment is blocked until the user identifies a disposable Colab session or the account's session capacity becomes available.

## Session-management attempt

After user confirmation, Colab session management was opened. The dialog did not reveal a named disposable session; instead it displayed `Runtime disconnected` and a disabled `Terminate other sessions` control while loading. No session was terminated. This preserves the user's existing work but leaves T4 allocation blocked until the session list becomes available or capacity is freed elsewhere.

## Allocation retry

The stalled dialog was closed and the T4 connection was retried. Colab again reported that a runtime was allocating. No existing session was terminated during this retry.

## Final retry status

A subsequent allocation retry again returned Colab's `Too many sessions` dialog. Because session management did not expose a terminable session in the available interface, no further session actions were taken. The GPU screening run has not started.
