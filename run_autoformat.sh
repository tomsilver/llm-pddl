#!/bin/bash
yapf -i -r --style .style.yapf --exclude '**/third_party' llmclone
yapf -i -r --style .style.yapf tests
yapf -i -r --style .style.yapf *.py
docformatter -i -r . --exclude venv llmclone/third_party
isort .
