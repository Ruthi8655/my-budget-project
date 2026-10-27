import requests

_BASE = "https://api.frankfurter.app"
_TIMEOUT = 8


def convert_currency(amount: float, from_cur: str, to_cur: str) -> float:
    from_cur, to_cur = from_cur.upper(), to_cur.upper()
    if from_cur == to_cur:
        return round(float(amount), 4)
    try:
        r = requests.get(
            f"{_BASE}/latest",
            params={"amount": amount, "from": from_cur, "to": to_cur},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        return round(r.json()["rates"][to_cur], 4)
    except requests.RequestException as e:
        raise ValueError(f"Currency conversion failed: {e}") from e
    except (KeyError, TypeError) as e:
        raise ValueError(f"Unexpected frankfurter response: {e}") from e
