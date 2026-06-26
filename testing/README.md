# Synthetic Math Fixtures

This folder is isolated from the whiteboard app runtime. It contains pure Python
fixture generation for handwritten-looking math boards and does not import
frontend, browser, OCR, DBNet, CoMER, server, or model code.

## Setup

```sh
python3 -m pip install -r testing/requirements.txt
```

## Render Fixtures

Render a single mixed-spacing multi-step algebra board:

```sh
python3 testing/render_math_fixture.py --problem algebra_steps --spacing mixed
```

Render a derivative board with explicit non-uniform line gaps:

```sh
python3 testing/render_math_fixture.py --problem derivative_evaluate --line-gaps 64,12,48
```

Render all integral fixtures across every spacing profile:

```sh
python3 testing/render_math_fixture.py --family integral --all-spacings
```

Generated PNG and JSON files are written only under `testing/results/` and are
ignored by git.

## Tests

```sh
python3 -m unittest discover testing
```
