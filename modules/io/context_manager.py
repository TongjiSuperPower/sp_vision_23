from typing_extensions import Self


class ContextManager:
    '''继承该类可以使用with语法'''

    def __init__(self) -> None:
        raise NotImplementedError('该函数需子类实现')

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> bool:
        # 按ctrl+c所引发的KeyboardInterrupt，判断为手动退出，不打印报错信息
        ignore_error = (exc_type == KeyboardInterrupt)
        self._close()
        return ignore_error

    def _close() -> None:
        raise NotImplementedError('该函数需子类实现')
