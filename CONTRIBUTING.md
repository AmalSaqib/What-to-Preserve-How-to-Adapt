# Contributing

Contributions that improve correctness, portability, documentation, and test
coverage are welcome.

## Development workflow

1. Create a focused branch.
2. Keep paper configurations in `reproducibility/experiments.json` rather than
   adding machine-specific shell commands.
3. Add or update a lightweight test for configuration and preprocessing code.
4. Run:

   ```bash
   python -m unittest discover -s reproducibility/tests -v
   python reproducibility/run_experiment.py --list > /dev/null
   python -m compileall -q reproducibility
   git diff --check
   ```

5. Describe whether a change affects checkpoint compatibility or reproduces
   the reported training behavior.

## Data and privacy

Never commit medical images, labels, patient identifiers, per-case predictions,
private access URLs, credentials, checkpoints, or absolute workstation paths.
Use synthetic fixtures for tests. Aggregate result tables must contain no case
identifiers.

## Experimental changes

Changes to block indexing, initialization, optimizer construction, calibration,
or evaluation splits can invalidate direct comparison with existing
checkpoints. Document such changes explicitly and use new plan/experiment IDs
so results cannot silently overwrite earlier runs.

## Style

- Prefer small, testable functions.
- Explain scientific intent and non-obvious invariants in comments.
- Avoid comments that merely repeat the code.
- Keep expensive commands dry-run-by-default.
- Use environment variables or CLI arguments instead of local paths.
