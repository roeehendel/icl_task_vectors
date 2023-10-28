import os
from contextlib import AbstractContextManager, ExitStack
from typing import Iterable


class CombinedContextManager(AbstractContextManager):
    def __init__(self, context_managers):
        self.context_managers = context_managers
        self.stack = None

    def __enter__(self):
        self.stack = ExitStack()
        for cm in self.context_managers:
            self.stack.enter_context(cm)
        return self.stack

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stack is not None:
            self.stack.__exit__(exc_type, exc_val, exc_tb)
