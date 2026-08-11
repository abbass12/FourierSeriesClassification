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
