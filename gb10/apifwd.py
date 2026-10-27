#!/usr/bin/env python3
"""apifwd.py: relay 127.0.0.1:$FWD_PORT on this node to the lane's API at $FWD_TARGET:$FWD_PORT.

When rank 0 and the OpenAI API run on the other node (API on its fabric address), this keeps the endpoint at
http://127.0.0.1:8000 on this node. Plain TCP relay: bytes in, bytes out (streaming included), nothing logged or parsed.
Run by gb10/beam-serve.sh as the user unit beam-fwd on $API_FWD_HOST.
"""
import asyncio
import os

LISTEN = os.environ.get("FWD_LISTEN", "127.0.0.1")
PORT = int(os.environ.get("FWD_PORT", "8000"))
TARGET = os.environ["FWD_TARGET"]


async def pipe(reader, writer):
    try:
        while data := await reader.read(1 << 16):
            writer.write(data)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()
    except (ConnectionError, OSError):
        pass


async def handle(client_r, client_w):
    try:
        up_r, up_w = await asyncio.open_connection(TARGET, PORT)
    except OSError:
        client_w.close()
        return
    try:
        await asyncio.gather(pipe(client_r, up_w), pipe(up_r, client_w))
    finally:
        up_w.close()
        client_w.close()


async def main():
    server = await asyncio.start_server(handle, LISTEN, PORT, reuse_address=True)
    async with server:
        await server.serve_forever()


asyncio.run(main())
