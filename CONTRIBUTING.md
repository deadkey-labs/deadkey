# Contributing to Deadkey

Thank you for helping harden sovereign local AI verification — especially via
adversarial bypass probes in [CHALLENGE.md](CHALLENGE.md)!

## Developer Certificate of Origin (DCO)

Every pull request must include a `Signed-off-by:` line in each commit message
(`git commit -s`), certifying that you wrote or have the right to submit the code
under this repository's **Apache License 2.0**:

```text
Signed-off-by: Your Name <you@example.com>
```

By submitting a contribution with a `Signed-off-by:` trailer, you certify that:
1. The contribution was created in whole or in part by you and you have the right
   to submit it under the Apache License 2.0 (see `LICENSE`); or
2. The contribution is based upon previous work covered under a compatible open
   source license and you have preserved all required notices; and
3. You understand and agree that this project and your contribution are public and
   that a record of the contribution (including your sign-off) is maintained
   indefinitely and may be redistributed consistent with this project's license.

## Quick Verification Before Opening a PR

```bash
python -m bench.demo_break_vs_hold
python -m bench.reproduce_all --quick
python tests/test_ci_portable.py
python tests/test_semantic_critic.py
python tests/test_mcp_and_proxy.py
python tests/test_a2a_and_doctor.py
```
