# Contributing
----------------------------------

1. File an issue to notify the maintainers about what you're working on.
2. Fork the repo, develop and test your code changes, add docs.
3. Make sure that your commit messages clearly describe the changes.
4. Send a pull request.

## File an Issue
----------------------------------

Use the issue tracker to start the discussion. It is possible that someone
else is already working on your idea, your approach is not quite right, or that
the functionality exists already. The ticket you file in the issue tracker will
be used to hash that all out.

## Style Guides
-------------------
1. Write in UTF-8 in Python 3
2. User modular architecture to group similar functions, classes, etc. 
3. Always use 4 spaces for indentation (don't use tabs)
4. Try to limit line length to 80 characters
5. Class names should always be capitalized
6. Function names should always be lowercase
7. Look at the existing style and adhere accordingly

## Fork the Repository
-------------------

Be sure to add the relevant tests before making the pull request. Docs will be
updated automatically when we merge to `main`, but you should also build
the docs yourself and make sure they're readable.
Include the dev and docs dependencies: `poetry install --with dev, docs`

## Write / update tests.
---------------------

Update and write additional tests in the `/tests/` folder of the repository. 
We aim to maintenance a 80% test coverage.

## Update Documentation
---------------------

Be sure to also update any affected documentation. We use `Sphinx` and you can automatically update the API documentation by running:

```bash
poetry run sphinx-apidoc -o docs/ llamea/
```

Also update any of the static files in the `docs` folder if needed.

To view the updated html, run:

```bash
cd docs  
poetry run sphinx-build -b html . _build
```


## Make the Pull Request
---------------------

Once you have made all your changes, tests, and updated the documentation,
make a pull request to move everything back into the main branch of the
`repository`. Be sure to reference the original issue in the pull request.
Expect some back-and-forth with regards to style and compliance of these
rules.