# Security policy

Please do not open a public issue for a vulnerability that could expose model
endpoints, host files or credentials. Report it privately through GitHub's
security advisory interface for this repository.

The included API binds to `127.0.0.1` by default and has no built-in
authentication. Users are responsible for firewalling or authenticating any
non-local deployment.
