# Running the suite in the cloud

Three working routes, in order of readiness. A design principle makes all of
them safe: the experiment suite needs generative calls only for the view-design
and topic-naming steps, a handful per corpus. Those run once on a machine that
holds credentials, and their responses land in the committed prompt cache, after
which every experiment is pure CPU over public datasets. No cloud runner ever
needs a secret.

1. **GitHub Actions (live now).** Every push runs the full test suite plus a
   seeded smoke experiment on a clean Ubuntu runner (.github/workflows/ci.yml),
   which is the continuous proof that a stranger's machine can reproduce the
   harness. Full experiments could run the same way via workflow_dispatch, but
   runners are 2-core, so a full corpus pass that takes minutes locally takes
   an hour there. Fine for verification, wasteful for iteration.

2. **A Claude Code cloud session.** Both repositories are on GitHub, so a cloud
   session can clone them and run `uv sync && uv run mcb run <id>` end to end.
   The pre-cached design responses make credentials unnecessary there too.

3. **The dormant H100 VM** (documented in the owner's infrastructure repo) if
   bulk generation ever becomes the bottleneck. Current spend patterns are cents
   per study, so this stays parked.

What does NOT move to the cloud: the private program-state repo path (the
boundary blocklist stays local), and any credential of any kind.

## Fresh-machine setup (spare laptop or any clean box)

```
git clone https://github.com/muralisidfn7/multicard-bench.git
cd multicard-bench
uv sync --python 3.11        # installs everything, ~2 GB with torch
uv run pytest -q             # 66 tests must pass before anything else
uv run mcb run e0_dilution   # first full experiment, no credentials needed
```

Notes for the spare laptop specifically. Apple Silicon or Linux machines can
relax the torch pin in pyproject.toml (it exists for Intel Mac wheels); results
are CPU-deterministic either way. Experiments that include a view-design or
topic-naming step need Vertex credentials in the environment
(VERTEX_AI_SERVICE_ACCOUNT_JSON); everything else runs with none. The private
program-state repo and the boundary blocklist are NOT needed to run experiments,
only to push public artifacts. Corpora download on first use into data/ (Enron
is the big one at ~423 MB via scripts/download_enron.sh; the rest fetch from
Hugging Face automatically).

## Full programme checkout on a second machine (both repos into ~/github_other)

```
mkdir -p ~/github_other && cd ~/github_other
gh repo clone muralisidfn7/multicard-bench
gh repo clone muralisidfn7/the-private-programme-repo
cd multicard-bench
mkdir -p .boundary
echo "$HOME/github_other/the-private-programme-repo" > .boundary/state-repo-path
uv sync --python 3.11
uv run pytest -q
```

Scope honesty: "everything" means these two repos. The guide repos
(agentic-enterprise and its local workbench) are not needed to run experiments;
the workbench is not a git repository at all and exists only on the primary
machine. Optional accelerators: copying data/cache/ from the primary machine
skips re-encoding (first runs otherwise rebuild it deterministically in under an
hour), and copying data/raw/enron_mail_20150507.tar.gz skips the 423 MB
download. Credentials: only view-design and topic-naming steps read
VERTEX_AI_SERVICE_ACCOUNT_JSON from the environment; set it on the second
machine only if those steps will run there. Both clones push and pull the same
origin, so work done on either machine lands in the same place; run the boundary
grep before pushing public artifacts from anywhere.
