# Factory Pattern

Prototypes of a factory on top of
[registry-pattern](https://github.com/adnanhd/registry-pattern): build an
object, or call a function, from a registry key plus keyword arguments, with
the arguments validated by pydantic. The maintained form of this idea is
`registry.build` / `registry.serialize` in registry-pattern; this repo keeps
the earlier prototypes and does not track the current registry API.

## Layout

- `factory/typ_factory.py`, `factory/fnc_factory.py`: `TypeFactory` and
  `FunctionalFactory`, dataclasses that wrap a `TypeRegistry` or
  `FunctionalRegistry` and plug into pydantic through
  `__get_pydantic_core_schema__`. They import `ObjectConfigMap`, which
  registry-pattern dropped after v0.1.1, so they fail to import against
  current registry-pattern.
- `factory/first_example.py`, `factory/second_example.py`:
  `TypeFactory.instantiate_artifact` and `FunctionFactory.execute_artifact`,
  which build a pydantic model from the target's signature and validate the
  keyword arguments against it. Their `__main__` demos call
  `add_registry_item`, `register_function` and `get_registry_item`, which the
  current registry no longer has; `second_example.py` also needs
  `typing_compat` and `torch`.
- `new-test/`: a self-contained prototype with its own `TypeRegistry`
  (`registry_pattern.py`) and `Factory` (`factory.py`). A `Factory`-annotated
  argument of a `validate_call` function takes `{"type": ..., **kwargs}`,
  validates the kwargs against a model built from the type's `__init__`, and
  serializes the config back.
- `procrastination/`, `new-procrastination/`: scratch experiments
  (validators, model graphs, a serializer demo).

## Install

`setup.py` installs the `factory` package with the dependencies in
`requirements.txt`: `pydantic>=2.8`, `typing_extensions` and
`registry @ git+ssh://...registry-pattern.git@main`. That last line fails:
registry-pattern now builds a distribution named `registry-pattern`, and pip
rejects the name mismatch. The `new-test/` demo needs only pydantic:

```bash
pip install 'pydantic>=2.8'
```

## Example

The one demo that runs as is needs only pydantic:

```bash
cd new-test
python main.py
```

It registers a mock ResNet-18 under `resnet18`, builds it from
`{"type": "resnet18", "pretrained": True, "layers": 18}`, validates it and
prints the serialized config.

## Checks

There are no tests; `pytest` collects nothing but the import errors of
`procrastination/trash/`. `deploy.sh` installs the tools and runs pytest,
pyright, black and flake8:

```bash
pip install pytest pytest-cov black flake8 pyright
pyright
black --check .
flake8 .
```
