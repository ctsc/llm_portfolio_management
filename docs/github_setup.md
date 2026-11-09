# GitHub setup

GitHub (`https://github.com/ctsc/llm_portfolio_management`) hosts the source and runs lint and
tests in GitHub Actions on every push. It never runs the weekly pipeline (decision N2): that is the
Carter's hand-run `scripts/run_weekend.py`. GitLab is the fallback host (`docs/gitlab_setup.md`).

## 1. Remotes

Carter's local checkout has two remotes:

```
origin   https://github.com/ctsc/llm_portfolio_management.git   (primary)
gitlab   https://gitlab.com/cartertierney0/llm_portfolio_management.git   (fallback mirror)
```

Push to both after each commit so the hosts never diverge:

```
git push origin master
git push gitlab master
```

To recreate the remotes on a fresh clone: `git remote add origin <github url>` and
`git remote add gitlab <gitlab url>`. The `.gitattributes` file keeps every text file on LF line
endings, so the Linux runner and a Windows checkout see the same bytes.

## 2. The workflow

`.github/workflows/ci.yml` mirrors `.gitlab-ci.yml` job for job:

| Job | Runs | Commands |
|---|---|---|
| `lint` | every push and pull request | `ruff check`, `mypy` |
| `test` | every push and pull request | `pytest tests/unit` with coverage (artifact `coverage.xml`) |
| `integration` | default branch only | `pytest tests/integration` |
| `e2e` | default branch only | `pytest tests/e2e -q` (the 30-week rehearsal, about 15 to 20 minutes) |
| `slow` | by hand: Actions, "ci", Run workflow, tick "slow" | `pytest -m slow tests` (the 300-stock benchmark) |
| `checks` | every push and pull request | prompt manifest hashes, placebo length, banned words, secret patterns, yamllint |

The `test`, `integration` and `e2e` jobs check out the full history (`fetch-depth: 0`) because the
phase checklist test counts one commit per phase. Python 3.11 with a pip cache is used everywhere.

## 3. Confirm the first run

1. Open the Actions tab; the `ci` workflow starts on the push.
2. Every job must be green before the repository is tagged for the freeze (phase 15 H).
3. No repository secrets or variables are needed. Nothing in CI calls a real service; the
   adapters are mock-tested only.

## 4. Notes

- The repository is public, so standard GitHub-hosted runners are free and there is no monthly
  minute budget to manage; the default-branch run (unit, integration, e2e) takes about 11 minutes
  of wall time because the jobs run in parallel.
- If a push fails the `checks` job after you edit a prompt file, bump the configuration
  `version` in `config/prompts_manifest.yaml`, rename the file to carry the new version, and run
  `python scripts/refresh_manifest.py --write` before committing.
- Going back to GitLab: `git push gitlab master` already keeps the mirror current, and
  `.gitlab-ci.yml` runs the same jobs there (see `docs/gitlab_setup.md`).
