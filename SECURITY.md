# Security

## Reporting a vulnerability

Report suspected vulnerabilities privately through GitHub's
[security advisory form](https://github.com/mahmoudmayaleh/ORION-v1/security/advisories/new)
rather than in a public issue. Expect an acknowledgement within a few days.

## Scope

This is a research simulator. It does not run as a service, expose a network listener, or
handle third-party data. The parts worth attention are:

- **API keys.** The frontier planner reads `ORION_FRONTIER_API_KEY` from the environment
  and never from a file in the repository. `frontier_backend.assert_key_absent` is called
  before anything is serialized, so a key cannot reach a result file, a telemetry sidecar
  or a log. Report any path that writes a key.
- **The language-model endpoint.** `LLMConfig.base_url` defaults to `localhost`. Pointing
  it at a remote endpoint sends slice specifications to that endpoint.
- **Result files.** These are committed and include a provenance block with the git commit,
  the dirty-file list and the serving model. Check what a result file contains before
  committing one from a private deployment.

## Secrets

Do not commit credentials. `.gitignore` excludes `.env`, `.env.*`, `*.key` and `*_secret*`.
If a credential does reach a commit, treat it as compromised and rotate it. Removing it in
a later commit does not remove it from history, and rewriting history does not remove it
from clones that already exist.
