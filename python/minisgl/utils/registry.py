from typing import Callable, Generic, Iterable, List, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """注册表：按名称注册和查找组件（用于插件化架构）"""

    def __init__(self, type: str):
        self._registry = {}  # 内部注册表字典
        self._type = type  # 组件类型描述（用于错误提示）

    def register(self, name: str) -> Callable[[T], None]:
        """返回一个装饰器，用于将组件注册到指定名称下"""
        if name in self._registry:
            raise KeyError(f"{self._type} '{name}' is already registered.")  # 防止重复注册

        def decorator(item: T) -> None:
            self._registry[name] = item

        return decorator

    def __getitem__(self, name: str) -> T:
        """按名称查找已注册的组件"""
        if name not in self._registry:
            raise KeyError(f"Unsupported {self._type}: {name}")
        return self._registry[name]

    def supported_names(self) -> List[str]:
        """获取所有已注册的组件名称列表"""
        return list(self._registry.keys())

    def assert_supported(self, names: str | Iterable[str]) -> None:
        """断言指定的名称是否为已注册的支持项（用于参数校验）"""
        if isinstance(names, str):
            names = [names]
        for name in names:
            if name not in self._registry:
                from argparse import ArgumentTypeError

                raise ArgumentTypeError(
                    f"Unsupported {self._type}: {name}. "
                    f"Supported items: {self.supported_names()}"
                )
