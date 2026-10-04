"""Small async Redis double with atomic pop and controllable key expiration."""

import asyncio
import time
from unittest.mock import AsyncMock


class InMemoryRedis:
    def __init__(self):
        self.entries = {}
        self.now = int(time.time())
        self.set = AsyncMock(side_effect=self.store)
        self.get = AsyncMock(side_effect=self.fetch)
        self.incr = AsyncMock(side_effect=self.increment)
        self.getdel = AsyncMock(side_effect=self.consume)
        self.delete = AsyncMock(side_effect=self.remove)
        self.exists = AsyncMock(side_effect=self.check_exists)
        self.eval = AsyncMock(side_effect=self.fixed_window)
        self.ping = AsyncMock(return_value=True)
        self.aclose = AsyncMock()

    def expire_key(self, key):
        if key in self.entries:
            expires_at = self.entries[key][1]
            if expires_at is not None and expires_at <= self.now:
                del self.entries[key]

    async def store(self, key, value, *, ex=None, exat=None, nx=None):
        await asyncio.sleep(0)
        self.expire_key(key)
        if nx and key in self.entries:
            return None
        if exat is None and ex is not None:
            exat = self.now + ex
        self.entries[key] = (value, exat)
        return True

    async def fetch(self, key):
        await asyncio.sleep(0)
        self.expire_key(key)
        entry = self.entries.get(key)
        return entry[0] if entry else None

    async def increment(self, key):
        await asyncio.sleep(0)
        self.expire_key(key)
        current = self.entries.get(key, (0, None))[0]
        updated = int(current) + 1
        self.entries[key] = (updated, None)
        return updated

    async def consume(self, key):
        # Let concurrent callers reach the command, then consume without yielding.
        await asyncio.sleep(0)
        self.expire_key(key)
        entry = self.entries.pop(key, None)
        return entry[0] if entry else None

    async def remove(self, key):
        await asyncio.sleep(0)
        self.expire_key(key)
        return int(self.entries.pop(key, None) is not None)

    async def check_exists(self, key):
        await asyncio.sleep(0)
        self.expire_key(key)
        return int(key in self.entries)

    async def fixed_window(self, script, numkeys, *args):
        """Emulate the rate-limit Lua script: INCR, expire-on-first-write,
        and PTTL, in one step. Refresh-session entries are untouched."""
        await asyncio.sleep(0)
        key, window_ms = args[0], args[1]
        self.expire_key(key)
        count = int(self.entries.get(key, (0, 0))[0]) + 1
        if count == 1:
            self.entries[key] = (count, self.now + window_ms / 1000)
        else:
            _, exat = self.entries[key]
            self.entries[key] = (count, exat)
        ttl_ms = int((self.entries[key][1] - self.now) * 1000)
        return [count, max(ttl_ms, 0)]
