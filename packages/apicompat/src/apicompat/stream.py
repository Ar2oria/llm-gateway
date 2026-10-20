"""Stateful SSE stream conversion.

The converter works on complete JSON events; real streams arrive as arbitrary
byte chunks that split events mid-line. :class:`StreamCodec` bridges the two:
feed it chunks, get back converted bytes ready to write to the client.

Three public entry points build on it:

* :func:`convert_stream` — synchronous, accepts any ``Iterable[bytes]``.
* :func:`astream` — asynchronous, accepts an ``AsyncIterable[bytes]``.
* :func:`convert_event` — one already-complete SSE event (advanced users who
  need to drive the timing themselves).

All of them share :class:`StreamCodec`, so the semantics are identical.
"""

from __future__ import annotations

from typing import AsyncIterable, AsyncIterator, Iterable, Iterator

from .converters import _DECODER_CLASSES, _ENCODER_CLASSES
from .exceptions import ConversionError
from .ir import IRStreamEvent, StreamEventType
from .protocol import Protocol

#: Converted events are re-emitted in small pieces so the client sees
#: streaming start immediately instead of waiting for a whole upstream chunk.
DEFAULT_CHUNK_SIZE = 64

#: Decoder options for every converted event.
ENCODE_OPTIONS = {
    "output_format": "sse",
    # Chat targets need the usage chunk for token accounting; the upstream
    # contract (sub2api's IncludeUsage) puts it on the final chunk.
    "include_usage": True,
}


def _as_protocol(value: Protocol | str) -> Protocol:
    if isinstance(value, Protocol):
        return value
    try:
        return Protocol(value)
    except ValueError as exc:
        valid = ", ".join(p.value for p in Protocol)
        raise ConversionError(f"unknown protocol {value!r} (expected one of {valid})") from exc


class StreamCodec:
    """Convert an SSE byte stream from one protocol to another.

    Usage::

        codec = StreamCodec("openai_responses", "anthropic_messages")
        for chunk in upstream_chunks:
            for out in codec.feed(chunk):
                write(out)
        for out in codec.finalize():
            write(out)

    ``source`` is the protocol the chunks currently speak (usually the
    upstream's), ``target`` is the protocol the caller wants.
    """

    def __init__(
        self,
        source: Protocol | str,
        target: Protocol | str,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        self.source = _as_protocol(source)
        self.target = _as_protocol(target)
        self.chunk_size = chunk_size

        # Fresh decoder/encoder per stream: both carry per-stream state
        # (block indices, sequence numbers, accumulated items).
        self._decoder = _DECODER_CLASSES[self.source]()
        self._encoder = _ENCODER_CLASSES[self.target]()

        self._buf = b""
        self._pending_event_lines: list[str] = []
        self._saw_termination = False
        self._finished = False

    # -- public API ----------------------------------------------------

    def feed(self, chunk: bytes | str) -> list[bytes]:
        """Consume one upstream chunk and return the bytes to forward.

        Chunks may split events at any byte offset; incomplete lines are held
        until the next call.
        """
        if self._finished:
            return []
        if isinstance(chunk, str):
            chunk = chunk.encode()
        self._buf += chunk
        out: list[bytes] = []
        while b"\n" in self._buf:
            line_bytes, self._buf = self._buf.split(b"\n", 1)
            self._consume_line(line_bytes.decode("utf-8", errors="replace"), out)
        return out

    def finalize(self) -> list[bytes]:
        """Flush the tail of the stream.

        Emits any trailing unterminated line and, when the upstream never sent
        a termination event, synthesizes the target protocol's closing
        sequence (close open blocks, final usage, ``message_stop`` /
        ``response.completed`` / ``[DONE]``). Idempotent.
        """
        if self._finished:
            return []
        self._finished = True
        out: list[bytes] = []

        # Flush an unterminated trailing event. Invalid input is skipped
        # rather than aborting the stream (matches the gateway's behaviour).
        for line in self._pending_event_lines:
            if line.startswith("data: "):
                try:
                    self._emit_ir_events(self._decoder.decode_stream_event(line), out)
                except Exception:
                    pass
        self._pending_event_lines = []

        # Synthesize a clean termination when the upstream just stopped.
        if not self._saw_termination:
            finalize = getattr(self._decoder, "finalize", None)
            if finalize is not None:
                try:
                    self._emit_ir_events(finalize(), out)
                except Exception:  # pragma: no cover - defensive, mirrors gateway
                    pass
        return out

    def convert_event(self, event_text: str) -> list[bytes]:
        """Convert one complete SSE event (``event:``/``data:`` lines).

        Unlike :meth:`feed` this does not need a trailing newline.
        """
        out: list[bytes] = []
        for line in event_text.split("\n"):
            self._consume_line(line, out)
        return out

    # -- internals -----------------------------------------------------

    def _consume_line(self, line: str, out: list[bytes]) -> None:
        if not line.strip():
            return
        if line.startswith("event:"):
            # The data line follows; keep the type for the decoder.
            self._pending_event_lines.append(line)
            return
        if not line.startswith("data: "):
            return

        data = line[6:]
        if data == "[DONE]":
            # Chat clients terminate with [DONE] themselves; every other
            # target needs the decoder to produce its terminal event.
            if self.target == Protocol.OPENAI_CHAT:
                self._saw_termination = True
                out.append(b"data: [DONE]\n\n")
            else:
                self._decode_and_emit("data: [DONE]", out)
            self._pending_event_lines = []
            return

        event_text = "\n".join(self._pending_event_lines + [line])
        self._pending_event_lines = []
        self._decode_and_emit(event_text, out)

    def _decode_and_emit(self, event_text: str, out: list[bytes]) -> None:
        try:
            ir_events = self._decoder.decode_stream_event(event_text)
        except Exception:
            # A malformed event is skipped rather than aborting the stream.
            return
        self._emit_ir_events(ir_events, out)

    def _emit_ir_events(self, ir_events: list[IRStreamEvent], out: list[bytes]) -> None:
        for ir_event in ir_events:
            if ir_event.type == StreamEventType.DONE:
                self._saw_termination = True
            for encoded in self._encoder.encode_stream_event(ir_event, options=ENCODE_OPTIONS):
                out.extend(_split(encoded, self.chunk_size))


def _split(encoded: str, chunk_size: int) -> list[bytes]:
    data = encoded.encode()
    if chunk_size <= 0:
        return [data]
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def convert_stream(
    source: Protocol | str,
    target: Protocol | str,
    chunks: Iterable[bytes],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[bytes]:
    """Convert a whole SSE stream, synchronously.

    ``source`` is the protocol the chunks speak, ``target`` the one the caller
    wants. The returned iterator yields bytes ready to write to the client.

    Example::

        for chunk in convert_stream("openai_responses", "anthropic_messages", upstream):
            response_stream.write(chunk)
    """
    codec = StreamCodec(source, target, chunk_size=chunk_size)
    for chunk in chunks:
        yield from codec.feed(chunk)
    yield from codec.finalize()


async def astream(
    source: Protocol | str,
    target: Protocol | str,
    chunks: AsyncIterable[bytes],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> AsyncIterator[bytes]:
    """Convert an SSE stream that arrives asynchronously.

    The conversion itself is pure CPU work, so it runs inline on the event
    loop; only the upstream read is awaited.
    """
    codec = StreamCodec(source, target, chunk_size=chunk_size)
    async for chunk in chunks:
        for out in codec.feed(chunk):
            yield out
    for out in codec.finalize():
        yield out


def encode_stream_terminator(protocol: Protocol | str) -> bytes:
    """Encode just the protocol's stream-terminating event.

    Useful when an upstream connection dies mid-body: the missing events
    cannot be recovered, but the client still needs a terminator
    (``message_stop`` / ``response.completed`` / ``[DONE]``) so its parser
    closes instead of hanging on a dangling block.
    """
    target = _as_protocol(protocol)
    encoder = _ENCODER_CLASSES[target]()
    events = encoder.encode_stream_event(
        IRStreamEvent(type=StreamEventType.DONE), options={"output_format": "sse"}
    )
    # The gateway has always joined these with newlines; keep that shape so
    # downstream byte comparisons stay identical.
    return "\n".join(events).encode()
