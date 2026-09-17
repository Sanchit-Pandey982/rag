"""Small async Redis double with atomic pop and controllable key expiration."""

import asyncio
import time
from unittest.mock import AsyncMock


class InMemoryRedis:
    def __init__(self):
        self.entries = {}
        self.now = int(time.time())
        self.set = AsyncMock(side_effect=self.store)
        self.getdel = AsyncMock(side_effect=self.consume)
        self.delete = AsyncMock(side_effect=self.remove)
        self.ping = AsyncMock(return_value=True)
        self.aclose = AsyncMock()

    def expire_key(self, key):
        if key in self.entries and self.entries[key][1] <= self.now:
            del self.entries[key]

    async def store(self, key, value, *, exat, nx):
        await asyncio.sleep(0)
        self.expire_key(key)
        if nx and key in self.entries:
            return None
        self.entries[key] = (value, exat)
        return True

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
