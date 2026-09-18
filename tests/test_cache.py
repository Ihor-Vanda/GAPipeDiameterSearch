"""LRUCache: витіснення найстарішого та оновлення давності при читанні."""

from pipeopt.solver.cache import LRUCache


def test_get_missing_returns_none():
    assert LRUCache(4).get("nope") is None


def test_set_and_get():
    c = LRUCache(4)
    c.set("a", 1)
    assert c.get("a") == 1


def test_evicts_least_recently_used():
    c = LRUCache(2)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)

    assert c.get("a") is None, "найстаріший ключ мав бути витіснений"
    assert c.get("b") == 2
    assert c.get("c") == 3


def test_read_refreshes_recency():
    c = LRUCache(2)
    c.set("a", 1)
    c.set("b", 2)
    c.get("a")          # 'a' знову найсвіжіший
    c.set("c", 3)       # витісняється 'b', не 'a'

    assert c.get("a") == 1
    assert c.get("b") is None


def test_overwrite_does_not_grow_cache():
    c = LRUCache(2)
    c.set("a", 1)
    c.set("a", 2)
    c.set("b", 3)

    assert c.get("a") == 2
    assert c.get("b") == 3
    assert len(c.cache) == 2
