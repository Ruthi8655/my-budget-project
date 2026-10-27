"""
loop.py — drives the model ↔ tool loop for one turn.

Runs while finish_reason == "tool_calls", appending one role:"tool" message
per tool_call_id.  Stops at MAX_ITERATIONS to prevent runaway loops.
"""
import os
import time
import openai
from API.harness.tools    import dispatch, get_openai_tools
from API.harness.caching  import ordered_messages, cached_tokens_from
from API.harness.tokens   import TurnUsage

# Pause before every model call to avoid rate-limit bursts (identical to SDK version)
MODEL_CALL_DELAY: float = float(os.getenv("MODEL_CALL_DELAY", "4"))

MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "10"))


def run_loop(
    client,
    model: str,
    system_msg: dict,
    history_window: list[dict],
    usage: TurnUsage,
) -> tuple[str, list[dict]]:
    """
    Execute the agentic loop for one user turn.

    Returns:
        answer        — final assistant text
        new_messages  — all messages produced this turn (to be persisted)
    """
    messages     = ordered_messages(system_msg, list(history_window))
    tools        = get_openai_tools()
    new_messages: list[dict] = []

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"[loop.py] model call #{iteration}")
        time.sleep(MODEL_CALL_DELAY)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
        except openai.BadRequestError as e:
            raw = getattr(e, "response", None)
            body = getattr(raw, "text", "") if raw is not None else ""
            print(f"[loop.py] BadRequestError — response body: {body[:600]}")
            raise
        except openai.RateLimitError as e:  # 429 — log and re-raise for caller/retry logic
            raw = getattr(e, "response", None)
            body = getattr(raw, "text", "") if raw is not None else ""
            print(f"[loop.py] RateLimitError — response body: {body[:600]}")
            raise
        usage.add(response.usage)
        cached_tokens_from(response)

        choice  = response.choices[0]
        msg     = choice.message
        finish  = choice.finish_reason

        # Serialise assistant message for history — use model_dump so that
        # extra fields (e.g. Gemini's thought_signature inside tool_calls)
        # are preserved verbatim and sent back on the next request.
        assistant_record: dict = msg.model_dump(exclude_none=True)
        # Normalise: ensure content is always present (some models omit it)
        assistant_record.setdefault("content", "")
        messages.append(assistant_record)
        new_messages.append(assistant_record)

        if finish != "tool_calls":
            print(f"[loop.py] finish_reason={finish} after {iteration} iteration(s)")
            return msg.content or "", new_messages

        # Dispatch every tool call and append tool messages
        for tc in msg.tool_calls:
            result = dispatch(tc.function.name, tc.function.arguments)
            tool_msg = {
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            }
            messages.append(tool_msg)
            new_messages.append(tool_msg)

    print(f"[loop.py] MAX_ITERATIONS={MAX_ITERATIONS} reached, returning last content")
    return msg.content or "", new_messages
