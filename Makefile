.DEFAULT_GOAL := help


### QUICK
# ¯¯¯¯¯¯¯

install: walker.install ## Install

train: walker.train ## Train

export PYTHONPATH=$PYTHONPATH:src

include makefiles/walker.mk
include makefiles/test.mk
include makefiles/format.mk
include makefiles/help.mk
