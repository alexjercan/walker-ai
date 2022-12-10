.DEFAULT_GOAL := help


### QUICK
# ¯¯¯¯¯¯¯

install: walker.install ## Install

evolve: walker.evolve ## Train

lint: test.lint

export PYTHONPATH=$PYTHONPATH:src

include makefiles/walker.mk
include makefiles/test.mk
include makefiles/format.mk
include makefiles/help.mk
