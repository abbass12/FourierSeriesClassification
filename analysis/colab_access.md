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

## Session-capacity recovery update

On the latest attempt, Colab session management exposed an enabled `Terminate other sessions` control. The user has authorized clearing disposable session capacity for this GPU workflow. The current notebook runtime is disconnected; the next action will terminate the other Colab sessions through this built-in control and then retry the T4 connection.

## Session manager outcome

The authorized termination control was invoked. Colab then reported `Failed to list sessions: Unknown error` and displayed `No active sessions`; the dialog was closed. Although the session list call failed, the reported state indicates that no active sessions remain. The next step is a clean T4 connection retry.

## Post-clearance connection retry

After the session-manager sequence, the T4 connection was retried. Colab showed an `Unable to connect to runtime` notification rather than opening a runtime. This indicates that the active-session condition alone was not sufficient to obtain a free runtime; the remaining blocker is a Colab-side allocation or account-capacity failure.

## Browser-session consequence

Refreshing the Colab notebook after the failed runtime allocation cleared the authenticated browser session. The notebook now shows `Sign in`, so a user-mediated Google sign-in is required before any further runtime connection can be attempted. This is a browser-authentication prerequisite, not a code or notebook error.

## Successful runtime recovery

After user reauthentication, Colab connected to a Python 3 GPU backend labelled `T4 (Python 3)`. Starting the GitHub-hosted bootstrap cell opened Colab's standard untrusted-notebook warning. The code in the cell is the reviewed repository clone and dependency installation command; execution will proceed through the built-in `Run anyway` confirmation.

## Runtime setup status

The notebook is connected to the T4 backend and has cloned the current GitHub repository successfully. The first setup cell remains in progress while installing dependencies from `requirements.txt`; the browser output has not shown an error, but the process has not completed after repeated checks. The next recovery step is to inspect or interrupt that setup command and use the Colab-provided PyTorch environment with only the missing lightweight dependencies, rather than waiting indefinitely for a full requirements reinstall.

## Setup-cell recovery outcome

The full dependency-installation command was interrupted after an extended stall. The Git clone itself completed. A subsequent re-run of the bootstrap cell created a nested checkout at `/content/FourierSeriesClassification/FourierSeriesClassification`; the working directory is now the nested, complete checkout. The next checks will use this actual directory, rely on the Colab-provided PyTorch installation, and install only any missing lightweight packages if an import fails.

## GPU verification

The recovered runtime successfully reported `PyTorch: 2.11.0+cu128`, `CUDA available: True`, and `GPU: Tesla T4`. The free T4 runtime blocker is resolved. The screening command can now be run from the nested repository checkout created during recovery.

## Screening run launched

The GPU screening command started successfully on CUDA with the verified Tesla T4. It uses seeds 11, 23, and 37; 300 synthetic examples per class; 1,500 sample points; 50 Fourier modes; 40 epochs; batch size 64; the trigonometric concentration factor; and combined jump descriptors. The first raw-model training phase for seed 11 reached epoch 40 with 90.67% training accuracy and 93.33% validation accuracy. The multi-model, three-seed workflow is still running.

## Current screening progress

The GPU screening remains active on the Tesla T4. During seed 11, the raw MLP completed 40 epochs (last reported train/validation accuracy: 90.67%/93.33%), the compact CNN completed 40 epochs (88.19%/83.33%), the Fourier MLP stopped early at epoch 37 after a 91.33% validation plateau, and the Fourier-plus-jump MLP has started. Results are not interpreted until all four models and all three seeds complete and the emitted JSON summary is available.

## Seed-level screening progress

Seed 11 completed on the Tesla T4 with held-out test accuracies of 91.67% (raw MLP), 90.00% (compact CNN), 93.33% (Fourier MLP), and 92.67% (Fourier-plus-jump MLP). Seed 23 is in progress: its raw MLP and compact CNN have completed their 40 epochs, and the Fourier MLP has completed its 40th epoch with a reported 93.33% validation accuracy. Final results remain pending all four models for seeds 23 and 37.

## Additional seed-level progress

Seed 23 completed with held-out accuracies of 92.67% (raw MLP), 86.33% (compact CNN), 95.33% (Fourier MLP), and 93.67% (Fourier-plus-jump MLP). Seed 37 is underway: the raw MLP and compact CNN have completed 40 epochs, and the Fourier MLP has completed 40 epochs with 96.00% validation accuracy. The final seed-37 jump-augmented model and aggregate statistical summary are still pending.

## Completed T4 screening result

The free Tesla T4 screening completed successfully in 77.935 seconds. Across seeds 11, 23, and 37, mean held-out accuracy (95% t confidence interval) was: raw MLP 91.78% (90.83%--92.73%), Fourier MLP 93.33% (91.07%--95.60%), Fourier-plus-jump MLP 92.56% (91.23%--93.88%), and compact 1D CNN 89.11% (86.33%--91.89%). Paired Wilcoxon tests were non-significant at this small three-seed screening scale: jump versus raw p=0.25, jump versus Fourier p=0.50, jump versus CNN p=0.25, and Fourier versus raw p=0.25. The result is therefore a successful GPU reproduction/screening only; it does not justify a general performance claim.
