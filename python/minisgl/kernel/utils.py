from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, List, NamedTuple, Tuple, TypeAlias, Union

if TYPE_CHECKING:
    from tvm_ffi import Module

KERNEL_PATH = pathlib.Path(__file__).parent / "csrc"  # CUDA/C++ 源码目录
DEFAULT_INCLUDE = [str(KERNEL_PATH / "include")]  # 默认头文件搜索路径
DEFAULT_CFLAGS = ["-std=c++20", "-O3"]  # 默认 C++ 编译选项
DEFAULT_CUDA_CFLAGS = ["-std=c++20", "-O3", "--expt-relaxed-constexpr"]  # 默认 CUDA 编译选项
DEFAULT_LDFLAGS = []  # 默认链接选项
CPP_TEMPLATE_TYPE: TypeAlias = Union[int, float, bool]  # C++ 模板参数允许的 Python 类型


class CppArgList(list[str]):
    """C++ 模板参数列表，支持格式化为逗号分隔的字符串"""
    def __str__(self) -> str:
        return ", ".join(self)


class KernelConfig(NamedTuple):
    """CUDA kernel 启动配置"""
    num_threads: int    # 每个 block 的线程数
    max_occupancy: int  # 最大占用率
    use_pdl: bool       # 是否使用 PDL (Programmatic Dependent Launch)

    @property
    def template_args(self) -> str:
        """将配置格式化为 C++ 模板参数字符串"""
        pdl = "true" if self.use_pdl else "false"
        return f"{self.num_threads},{self.max_occupancy},{pdl}"


def _make_name(*args: str) -> str:
    """生成唯一的模块名称（以 minisgl__ 为前缀）"""
    return "minisgl__" + "_".join(str(arg) for arg in args)


def _make_wrapper(tup: Tuple[str, str]) -> str:
    """生成 TVM FFI 导出包装器代码"""
    export_name, kernel_name = tup
    return f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({export_name}, ({kernel_name}));"


def make_cpp_args(*args: CPP_TEMPLATE_TYPE) -> CppArgList:
    """将 Python 参数转换为 C++ 模板参数字符串列表"""
    def _convert(arg: CPP_TEMPLATE_TYPE) -> str:
        if isinstance(arg, bool):
            return "true" if arg else "false"
        if isinstance(arg, (int, float)):
            return str(arg)
        raise TypeError(f"Unsupported argument type for cpp template: {type(arg)}")

    return CppArgList(_convert(arg) for arg in args)


def load_aot(
    *args: str,                    # 模块名称组件
    cpp_files: List[str] | None = None,    # C++ 源文件列表（路径相对于 csrc/src）
    cuda_files: List[str] | None = None,   # CUDA 源文件列表
    extra_cflags: List[str] | None = None,        # 额外的 C++ 编译选项
    extra_cuda_cflags: List[str] | None = None,   # 额外的 CUDA 编译选项
    extra_ldflags: List[str] | None = None,        # 额外的链接选项
    extra_include_paths: List[str] | None = None,  # 额外的头文件搜索路径
    build_directory: str | None = None,            # 构建目录
) -> Module:
    """加载预编译的 AOT（Ahead-of-Time）模块"""
    from tvm_ffi.cpp import load

    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    cpp_files = [str((KERNEL_PATH / "src" / f).resolve()) for f in cpp_files]  # 拼接完整路径
    cuda_files = [str((KERNEL_PATH / "src" / f).resolve()) for f in cuda_files]

    return load(
        _make_name(*args),
        cpp_files=cpp_files,
        cuda_files=cuda_files,
        extra_cflags=DEFAULT_CFLAGS + extra_cflags,
        extra_cuda_cflags=DEFAULT_CUDA_CFLAGS + extra_cuda_cflags,
        extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
        extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
        build_directory=build_directory,
    )


def load_jit(
    *args: str,                    # 模块名称组件
    cpp_files: List[str] | None = None,    # C++ 源文件列表（路径相对于 csrc/jit）
    cuda_files: List[str] | None = None,   # CUDA 源文件列表
    cpp_wrappers: List[Tuple[str, str]] | None = None,   # C++ 包装器列表
    cuda_wrappers: List[Tuple[str, str]] | None = None,  # CUDA 包装器列表
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    build_directory: str | None = None,
) -> Module:
    """加载 JIT（Just-in-Time）编译的内联模块"""
    from tvm_ffi.cpp import load_inline

    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    cpp_wrappers = cpp_wrappers or []
    cuda_wrappers = cuda_wrappers or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    # 包含 C++ 文件（通过 #include 指令）
    cpp_paths = [(KERNEL_PATH / "jit" / f).resolve() for f in cpp_files]
    cpp_sources = [f'#include "{path}"' for path in cpp_paths]
    cpp_sources += [_make_wrapper(tup) for tup in cpp_wrappers]

    # 包含 CUDA 文件
    cuda_paths = [(KERNEL_PATH / "jit" / f).resolve() for f in cuda_files]
    cuda_sources = [f'#include "{path}"' for path in cuda_paths]
    cuda_sources += [_make_wrapper(tup) for tup in cuda_wrappers]

    return load_inline(
        _make_name(*args),
        cpp_sources=cpp_sources,
        cuda_sources=cuda_sources,
        extra_cflags=DEFAULT_CFLAGS + extra_cflags,
        extra_cuda_cflags=DEFAULT_CUDA_CFLAGS + extra_cuda_cflags,
        extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
        extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
        build_directory=build_directory,
    )
