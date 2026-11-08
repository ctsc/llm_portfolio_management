# GitLab setup (fallback host)

The project moved to GitHub (`docs/github_setup.md`) once that account was available again;
GitLab stays configured as the remote `gitlab` and `.gitlab-ci.yml` is kept current, so a return
to GitLab needs no code change. GitLab hosts a mirror of the source and runs lint and tests in CI on every push. It never runs the weekly
pipeline (decision N2): that is the researcher's hand-run `scripts/run_weekend.py` on the desktop.

## 1. Create the project

1. Sign in to GitLab and create a new blank project (private). Do not initialize it with a README.
2. Copy the SSH or HTTPS URL of the empty project.

## 2. Push the repository

From the repository root on the desktop:

```
git remote add origin <project url>
git push -u origin --all
git push origin --tags
```

The `.gitattributes` file keeps every text file on LF line endings, so the Linux runner and the
Windows desktop see the same bytes.

## 3. Confirm the pipeline passes

1. Open CI/CD, Pipelines in the project. The first pipeline starts on the push.
2. Stages: `lint` (ruff, mypy), `test` (pytest over `tests/unit` and `tests/integration` with
   coverage), `checks` (prompt manifest hashes, placebo length, banned words, secret patterns,
   yamllint). The `e2e` job runs only on the default branch.
3. Every stage must be green before the repository is tagged `frozen-v1` (v1 phase 17 E2).

## 4. Notes

- The free tier gives about 400 CI minutes a month. The unit and integration suite takes
  roughly 10 minutes on a shared runner; the e2e job adds about the same, which is why it runs
  only on the default branch.
- No CI variables or secrets are needed. Nothing in CI calls a real service; the adapters are
  mock-tested only.
- Pip downloads are cached per branch under `.cache/pip`.
- If a push fails the `checks` stage after you edit a prompt file, bump the configuration
  `version` in `config/prompts_manifest.yaml`, rename the file to carry the new version, and
  run `python scripts/refresh_manifest.py --write` before committing.
