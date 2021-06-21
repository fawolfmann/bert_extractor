"""Entrypoint for nox."""

import nox


@nox.session(reuse_venv=True, python="3.8")
def tests(session):
    """Run all tests."""
    session.install("poetry")
    session.run("poetry", "install")

    cmd = ["poetry", "run", "pytest", "-n", "auto"]

    if session.posargs:
        cmd.extend(session.posargs)
    session.run(*cmd)


@nox.session(reuse_venv=True, python="3.8")
def lint(session):
    """Run all pre-commit hooks."""
    session.install("poetry")
    session.run("poetry", "install")
    session.run("poetry", "run", "pre-commit", "install")
    session.run(
        "poetry", "run", "pre-commit", "run", "--show-diff-on-failure", "--all-files"
    )
