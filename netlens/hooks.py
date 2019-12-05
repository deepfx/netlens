from abc import ABC
from typing import Callable, Any, Mapping, Union, Collection

from torch import Tensor, nn

Tensors = Union[Tensor, Collection['Tensors']]
ModuleHookFunc = Callable[[nn.Module, Tensors, Tensors], Any]


def is_collection(x: Any) -> bool:
    return isinstance(x, (tuple, list))


class Hook(ABC):
    def __init__(self, hook_receiver, hook_func, detach: bool = True):
        self.hook_func, self.detach, self.stored = hook_func, detach, None
        self.hook = hook_receiver(self.hook_fn_wrapper)
        self.removed = False

    def hook_fn_wrapper(self, *args):
        out = self.hook_func(*args)
        self.stored = out.detach() if self.detach and isinstance(out, Tensor) else out
        return out

    def remove(self):
        if not self.removed:
            self.hook.remove()
            self.removed = True

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()

    def __del__(self):
        self.remove()


class TensorHook(Hook):
    def __init__(self, t: Tensor, hook_func: Callable[[Tensor], Any], detach: bool = True):
        super(TensorHook, self).__init__(t.register_hook, hook_func, detach)


class ModuleHook(Hook):
    def __init__(self, m: nn.Module, hook_func: ModuleHookFunc, is_forward: bool = True, detach: bool = True):
        f = m.register_forward_hook if is_forward else m.register_backward_hook
        super(ModuleHook, self).__init__(f, hook_func, detach)


class HookDict:
    def __init__(self, hooks: Mapping[str, Hook] = None):
        self.hooks = hooks or {}

    def __getitem__(self, key):
        return self.hooks[key]

    def __setitem__(self, key, hook):
        self.hooks[key] = hook

    def __delitem__(self, key):
        del self.hooks[key]

    def __len__(self):
        return len(self.hooks)

    def __iter__(self):
        return iter(self.hooks)

    @property
    def stored(self):
        return {key: o.stored for key, o in self.hooks.items()}

    def get_stored(self, key):
        return self.hooks[key].stored if key in self.hooks else None

    def remove(self):
        for h in self.hooks.values():
            h.remove()

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()

    def __del__(self):
        self.remove()

    def add_gradient_captor_hook(self, key: str, t: Tensor):
        """
        A convenience method that for the given tensor, adds a hook that just captures its gradient.
        """
        self.hooks[key] = TensorHook(t, lambda grad: grad)

    @classmethod
    def from_tensors(cls, ts: Mapping[str, Tensor], hook_func: Callable[[Tensor], Any], detach: bool = True):
        return cls({key: TensorHook(t, hook_func, detach) for key, t in ts.items()})

    @classmethod
    def from_modules(cls, ms: Mapping[str, nn.Module], hook_func: ModuleHookFunc, is_forward: bool = True, detach: bool = True):
        return cls({key: ModuleHook(m, hook_func, is_forward, detach) for key, m in ms.items()})
