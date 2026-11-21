# Project-wide instructions

## Git and GitHub workflow

After completing each clearly defined implementation step:

1. Run the relevant tests and checks.
2. Review all modified and newly created files.
3. Run `git status` and inspect the changes.
4. Check that no API keys, passwords, `.env` files, private data, or unnecessary large files are included.
5. Never overwrite or modify the original notebook or existing trained model checkpoint unless I explicitly authorize it.
6. Present a short summary of the changes, test results, files to be committed, and the proposed commit message.
7. Wait for explicit approval before committing or pushing.
8. After approval, stage only the relevant files, create a descriptive commit, and push it to the current development branch.
9. Verify that the push succeeded and report the commit hash.

## Important restrictions

- Never force-push.
- Never use destructive Git commands to discard work.
- Never commit secrets or credentials.
- Never blindly use `git add .`.
- Never push incomplete or untested work.
- Never combine multiple implementation steps into one commit unless explicitly asked.

Follow these instructions for every future task in this repository.
