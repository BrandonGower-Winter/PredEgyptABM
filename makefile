PATH_TO_PYPY="/usr/bin/pypy3"

install: venv
	. venv/bin/activate; pip3 install -Ur requirements.txt

venv :
	test -d venv || virtualenv -p $(PATH_TO_PYPY) venv

profile: venv
	. venv/bin/activate; python -m vmprof ./src/main.py

clean:
	rm -rf venv
	find -iname "*.pyc" -delete