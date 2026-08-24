# Contributing Guide

## Developer Setup

Python 3.9–3.11 and TensorFlow 2.15 are required. Install with the
`[dev,doc]` extras (and `[cpu]` or `[gpu]` for TensorFlow):

```sh
git clone https://github.com/Teoroo-CMC/PiNN.git
cd PiNN
pip install -e '.[cpu,dev,doc]'   # or '.[gpu,dev,doc]'
pytest                            # run the tests
mkdocs serve                      # live documentation
```

## Pull Request Checklist

Contributions to the main repo `teoroo-cmc/pinn` shall proceed with a pull
request (PR), which will be reviewed by at least one member of the PiNN team.
Track your development with a fork and start contributing by opening a PR. PR
can be used to discussed new features without implementation. 

GitHub Actions runs the tests on every push and pull request. Documentation is
deployed from `Teoroo-CMC/PiNN` on `master` and version tags; forks still *build*
the docs and the CPU image to catch breakage. Below is a checklist before
merging commits to the master branch:

### Code Quality

- [ ] Test should pass;
- [ ] New code should be tested at least with an integration test (e.g. a layer
      should be used in a test of the potential model);
- [ ] Components with non-trivial expected outputs should be tested with simple
      cases (e.g. neighbor list or equivarient tensor products).

### Documentation

- [ ] Documentation build should pass;
- [ ] New layers should be documented, include equations when necessary;
- [ ] Citations should be added as a Bibtex entry.


### Hygiene Rules

- [ ] **Rebase changes** new changes should always be rebased on the latest
      master branch before merged, see [interactive
      rebase](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History) if you
      need to rewrite history;
- [ ] Follow [PEP 440](https://peps.python.org/pep-0440/) for versioning.
