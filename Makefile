.PHONY: clean env-and-requirements test docker-build docker-push

#################################################################################
#
# Makefile to build the entire digital fingerprint module. Perhaps this could be Gradle:)
#
#################################################################################

NAMESPACE=kaggle
SHELL := /bin/bash
PROJECT_NAME = home-credit
PYTHON_INTERPRETER = python3
PYTHONPATH=./src/:./tests/src



#################################################################################
# Setup
#################################################################################

ifeq (,$(shell conda info --envs | grep $(PROJECT_NAME)))
	HAS_CONDA_ENV=False
else
	HAS_CONDA_ENV=True
endif


# If conda is available then run that, unless its overriden on the command line with FORCE_VENV=True
ifeq (True, $(FORCE_VENV))
	HAS_CONDA=False
else ifeq (,$(shell which conda))
	HAS_CONDA=False
else
	HAS_CONDA=True
endif

# In CI (alpine) we want to run pip3
ifeq (True, $(CI))
	PIP:=pip3
else
	PIP:=pip
endif

# Turn off conda to test venv setup (used by Gitlabs)
#HAS_CONDA=False

## Set up python interpreter environment. If Conda is installed this will be used, if not it will fall back to virtual env.
create_environment:
	@echo ">>> About to create environment: $(PROJECT_NAME)..."
ifeq (True,$(HAS_CONDA))
ifeq (True,$(HAS_CONDA_ENV))
	@echo ">>> Detected conda, found existing conda environment."
else
	@echo ">>> Detected conda, creating conda environment."
	( \
	  conda create -m -y --name $(PROJECT_NAME) python=3.6; \
	)
endif
else
	@echo ">>> check python3 version"
	( \
		$(PYTHON_INTERPRETER) --version; \
	)
	@echo ">>> No conda detected, using VirtualEnv."
	( \
	    $(PIP) install -q virtualenv virtualenvwrapper; \
	    virtualenv venv --python=$(PYTHON_INTERPRETER); \
	)
endif


# Define utility variable to help calling Python from the virtual environment
ifeq (True,$(HAS_CONDA))
    ACTIVATE_ENV := source activate $(PROJECT_NAME)
else
    ACTIVATE_ENV := source venv/bin/activate
endif

# Execute python related functionalities from within the project's environment
define execute_in_env
	$(ACTIVATE_ENV) && $1
endef


## Install Python Dependencies
environment: create_environment
	$(call execute_in_env, which python)
	# $(call execute_in_env, which $(PIP))
	$(call execute_in_env, pip install -r ./requirements.txt)
	$(call execute_in_env, pip install -U pytest)
	# $(call execute_in_env, which $(PIP))
	# $(call execute_in_env, $(PIP) install -r ./requirements.txt)
	# $(call execute_in_env, $(PIP) install -U pytest)



#################################################################################
# Clean
#################################################################################

## Clean any artifacts created
clean:
	( \
	  rm -Rf ./src/build; \
	  rm -Rf ./src/kaggle-home-credit.egg-info; \
	  rm -Rf ./src/dist; \
	  rm -Rf ./venv; \
	)


#################################################################################
# Test / Validate
#################################################################################

## Run the test suite
test: environment
	$(call execute_in_env, PYTHONPATH=${PYTHONPATH} $(PYTHON_INTERPRETER) -m pytest --junitxml=./build/test-results.xml)


## Run test coverage
coverage: environment
	$(call execute_in_env, PYTHONPATH=${PYTHONPATH} python -m pytest --cov=digital_fingerprint)

## Lint using flake8
lint: environment
	$(call execute_in_env, PYTHONPATH=${PYTHONPATH} flake8 src)


## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# Build
#################################################################################

# TODO: This breaks under Conda but need to fix relative dir to full path
## Run the Python setup utils to package a source / binary package
package: environment
	# Run bdist to package
	( pwd; \
	  cd src; \
      source ../venv/bin/activate && ${PYTHON_INTERPRETER} setup.py sdist; \
	)
	# Now copy the version tar to the latest version
	( \
	  rm -rf ./dist; \
	  mv ./src/dist ./dist; \
	  basename `ls  ./dist/*.gz`|xargs -I{} cp ./dist/{} ./dist/kaggle-home-credit.tar.gz \
	)


#################################################################################
# Help
#################################################################################

.DEFAULT_GOAL := help
# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
