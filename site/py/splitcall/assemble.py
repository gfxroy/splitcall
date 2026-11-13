"""Turn parser deltas into one normalized tool-call result."""

from __future__ import annotations

from splitcall.types import Delta, NormCall, Normalized


def assemble(
    deltas: list[Delta],
    *,
    finish_reason: str | None,
    assigns_id: bool,
    assigns_finish: bool,
    assigns_type: bool,
    error: str | None = None,
    reasoning: str = "",
) -> Normalized:
    """Group streaming deltas by index in first-seen order.

    Later deltas with id None do not clear an id already seen. Argument
    strings are concatenated. A repeated identical name is ignored.
    """
    if error:
        return Normalized(
            error=error,
            finish_reason=finish_reason,
            assigns_id=assigns_id,
            assigns_finish=assigns_finish,
            assigns_type=assigns_type,
            reasoning=reasoning,
            deltas=deltas,
        )

    content_parts: list[str] = []
    order: list[int] = []
    bucket: dict[int, NormCall] = {}
    unindexed: list[NormCall] = []

    for delta in deltas:
        if delta.content:
            content_parts.append(delta.content)
        toolish = any(
            value is not None
            for value in (delta.name, delta.arguments, delta.id, delta.type)
        )
        if not toolish:
            continue
        if delta.index is None:
            unindexed.append(
                NormCall(
                    index=None,
                    name=delta.name,
                    arguments=delta.arguments or "",
                    id=delta.id,
                    type=delta.type,
                )
            )
            continue
        call = bucket.get(delta.index)
        if call is None:
            call = NormCall(index=delta.index, name=None, arguments="", id=None, type=None)
            bucket[delta.index] = call
            order.append(delta.index)
        if delta.name:
            if call.name is None:
                call.name = delta.name
            elif delta.name != call.name:
                if delta.name.startswith(call.name):
                    call.name = delta.name
                elif not call.name.startswith(delta.name):
                    call.name += delta.name
        if delta.arguments:
            call.arguments += delta.arguments
        if delta.id:
            if call.id is None:
                call.id = delta.id
            elif delta.id != call.id:
                call.id_changes += 1
                call.id = delta.id
        if delta.type and not call.type:
            call.type = delta.type

    calls = [bucket[index] for index in order]
    calls.extend(unindexed)
    return Normalized(
        content="".join(content_parts),
        calls=calls,
        finish_reason=finish_reason,
        assigns_id=assigns_id,
        assigns_finish=assigns_finish,
        assigns_type=assigns_type,
        reasoning=reasoning,
        deltas=deltas,
    )


def assemble_calls(
    calls: list[NormCall],
    content: str | None,
    *,
    finish_reason: str | None,
    assigns_id: bool,
    assigns_finish: bool,
    assigns_type: bool,
    reasoning: str = "",
    error: str | None = None,
) -> Normalized:
    """Keep batch results in list order. Do not merge calls that share an index."""
    return Normalized(
        content=content or "",
        calls=list(calls),
        finish_reason=finish_reason,
        assigns_id=assigns_id,
        assigns_finish=assigns_finish,
        assigns_type=assigns_type,
        reasoning=reasoning,
        error=error,
    )
