"""流式输出回调"""

from typing import Callable, Optional


class StreamingCallback:
    """流式输出回调收集器。

    用法：
        cb = StreamingCallback(on_token=lambda t: print(t, end="", flush=True))
        for chunk in llm.stream(messages):
            cb(chunk.content)
        print(cb.full_text)
    """

    def __init__(self, on_token: Optional[Callable[[str], None]] = None):
        self._tokens: list[str] = []
        self.on_token = on_token

    def __call__(self, token: str):
        self._tokens.append(token)
        if self.on_token:
            self.on_token(token)

    @property
    def full_text(self) -> str:
        return "".join(self._tokens)
