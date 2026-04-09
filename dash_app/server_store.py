"""
Server-side in-memory store for DataFrames and models.
Avoids serializing large objects through dcc.Store (browser).
"""

_store = {}


def set_df(key: str, value) -> None:
    _store[key] = value


def get_df(key: str):
    return _store.get(key)


def delete(key: str) -> None:
    _store.pop(key, None)


def clear() -> None:
    _store.clear()


def keys() -> list:
    return list(_store.keys())


# Convenience aliases
def set_val(key: str, value) -> None:
    _store[key] = value


def get_val(key: str, default=None):
    return _store.get(key, default)
