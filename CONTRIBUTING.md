# Contributing

Hardware reports and narrowly scoped fixes are welcome.

When opening an issue or pull request, include:

- device and total unified memory;
- OS, driver and Docker versions;
- exact model and SGLang revisions;
- launch command and relevant logs;
- prompt length, output length, concurrency and measured tokens/s;
- whether the PLE mmap was on NVMe and whether host or container swap was used.

Do not submit model weights, access tokens, private IP addresses or unrelated
generated outputs. Run these checks before a pull request:

```bash
bash -n run-spark.sh
python3 -m py_compile patch_qwen4_ple_disk_cache.py patch_sm121_qsa_sdpa.py prepare_ple.py
```
