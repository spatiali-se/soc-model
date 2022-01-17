# soc-model

A repository for building, testing and logging SOC prediction models.

## Docker
[//]: # (TODO: Explain that Docker needs to be installed for this to work locally)

Build the docker image running the following command in your terminal from the main project directory:

```
docker build -t mlflow-docker -f Dockerfile .
```

## MLflow

Run the project by executing the following command in your terminal from the main project directory:

```
mlflow run --no-conda .
```

To view results in a graphical interface run `mlflow ui` and open the designated url (default: [`http://127.0.0.1:5000`](http://127.0.0.1:5000))

## Committing to this repository

A commit message should be structured as follows:

```
<type>[optional scope]: <description>
[optional body]

[optional footer(s)]
```

The commit contains the following structural elements, to communicate intent to the consumers of your library:

1. **fix:** a commit of the *type* `fix:` patches a bug in your codebase (this correlates with [`PATCH`](https://semver.org/#summary) in Semantic Versioning).

2. **feat:** a commit of the *type* `feat:` introduces a new feature to the codebase (this correlates with [`MINOR`](https://semver.org/#summary) in Semantic Versioning).

3. **BREAKING CHANGE:** a commit that has a footer `BREAKING CHANGE:`, or appends a `!` after the type/scope, introduces a breaking API change (correlating with [`MAJOR`](https://semver.org/#summary) in Semantic Versioning). A BREAKING CHANGE can be part of commits of any type.

4. *types* other than `fix:` and `feat:` are allowed, for example [@commitlint/config-conventional](https://github.com/conventional-changelog/commitlint/tree/master/%40commitlint/config-conventional) (based on the [the Angular convention](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#-commit-message-guidelines)) recommends `build:`, `chore:`, `ci:`, `docs:`, `style:`, `refactor:`, `perf:`, `test:`, and others.

5. *footers* other than `BREAKING CHANGE: <description>` may be provided and follow a convention similar to [git trailer format](https://git-scm.com/docs/git-interpret-trailers). Additional types are not mandated by the Conventional Commits specification, and have no implicit effect in Semantic Versioning (unless they include a BREAKING CHANGE). A scope may be provided to a commit’s type, to provide additional contextual information and is contained within parenthesis, e.g., `feat(parser): add ability to parse arrays.`

For more information and best practices, see [this link](https://www.conventionalcommits.org/en/v1.0.0/).
