"""
utils/progress.py — tqdm wrapper với fallback tự implement.
"""

import time

try:
    from tqdm import tqdm as _tqdm

    class tqdm(_tqdm):
        """Wrapper mỏng — giữ interface gốc."""
        pass

    HAS_TQDM = True

except ImportError:
    HAS_TQDM = False

    class tqdm:  # type: ignore[no-redef]
        """
        Fallback tqdm-like class khi thư viện chưa được cài.
        Hỗ trợ context manager, iterator và update().
        """

        def __init__(
            self,
            iterable=None,
            total: int | None = None,
            desc: str = "",
            unit: str = "it",
            leave: bool = True,
            ncols: int | None = None,
            **kwargs,
        ):
            self._iter = iterable
            self._total = total or (
                len(iterable) if iterable is not None and hasattr(iterable, "__len__") else None
            )
            self._desc = desc
            self._unit = unit
            self._n = 0
            self._start = time.time()

        def __iter__(self):
            for item in self._iter:
                yield item
                self.update(1)
            self.close()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def update(self, n: int = 1) -> None:
            self._n += n
            elapsed = time.time() - self._start
            rate = self._n / elapsed if elapsed > 0 else 0
            total_str = f"/{self._total}" if self._total else ""
            eta_str = ""
            if self._total and rate > 0:
                eta = (self._total - self._n) / rate
                eta_str = f" ETA {eta:.0f}s"
            print(
                f"\r  {self._desc}: {self._n}{total_str} {self._unit}"
                f"  [{elapsed:.0f}s{eta_str}  {rate:.1f}{self._unit}/s]",
                end="",
                flush=True,
            )

        def set_postfix_str(self, s: str = "", **kwargs) -> None:
            pass

        def close(self) -> None:
            print()