"""Base functionality for async communication.

Distributed under the GNU General Public License v2
Copyright (C) 2024 Alex Ruddick
Copyright (C) 2023 NuMat Technologies
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

import serial
import serial_asyncio_fast  # type: ignore

logger = logging.getLogger('alicat')


class Client(ABC):
    """Serial or TCP client."""

    address: str
    open: bool
    timeout: float
    timeouts: int
    max_timeouts: int
    reconnecting: bool
    eol: bytes
    lock: asyncio.Lock

    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter

    def __init__(self, timeout: float):
        """Initialize common attributes."""
        self.address = ''
        self.open = False
        self.timeout = timeout
        self.timeouts = 0
        self.max_timeouts = 10
        self.reconnecting = False
        self.eol = b'\r'
        self.lock = asyncio.Lock()

    async def _read(self, length: int) -> str:
        """Read a fixed number of bytes from the device."""
        response = await self.reader.read(length)
        return response.decode().strip()

    async def _readline(self) -> str:
        """Read until a line terminator."""
        response = await self.reader.readuntil(self.eol)
        return response.decode().strip().replace('\x00', '')

    async def _write(self, message: str) -> None:
        """Write a command and do not expect a response."""
        self.writer.write(message.encode() + self.eol)

    async def write_and_read(self, command: str) -> str | None:
        """Write a command and read a response.

        As industrial devices are commonly unplugged, this has been expanded to
        handle recovering from disconnects.
        """
        await self._handle_connection()
        async with self.lock:
            if self.open:
                try:
                    response = await self._handle_communication(command)
                    return response
                except asyncio.IncompleteReadError:
                    logger.error('IncompleteReadError.  Are there multiple connections?')
                    return None
            else:
                return None

    async def clear(self) -> None:
        """Clear the reader stream when it has been corrupted from multiple connections."""
        logger.warning("Multiple connections detected; clearing reader stream.")
        try:
            junk = await asyncio.wait_for(self._read(100), timeout=self.timeout)
            logger.warning(junk)
        except TimeoutError:
            pass

    async def _handle_communication(self, command: str) -> str | None:
        """Manage communication, including timeouts and logging."""
        try:
            await self._write(command)
            future = self._readline()
            result = await asyncio.wait_for(future, timeout=self.timeout)
            self.timeouts = 0
            return result
        except (asyncio.TimeoutError, TypeError, OSError):
            self.timeouts += 1
            if self.timeouts == self.max_timeouts:
                logger.error(f'Reading from {self.address} timed out '
                             f'{self.timeouts} times.')
                await self.close()
            return None

    async def _handle_connection(self) -> None:
        """Automatically maintain connection."""
        if self.open:
            return
        async with self.lock:
            try:
                self.reader, self.writer = await asyncio.wait_for(self._connect(), timeout=self.timeout)
                self.reconnecting = False
            except (asyncio.TimeoutError, OSError):
                if not self.reconnecting:
                    logger.error(f'Connecting to {self.address} timed out.')
                self.reconnecting = True

    async def close(self) -> None:
        """Close the connection."""
        if self.open:
            self.writer.close()
            await self.writer.wait_closed()
        self.open = False

    @abstractmethod
    async def _connect(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        ...


class TcpClient(Client):
    """A generic reconnecting asyncio TCP client.

    This base functionality can be used by any industrial control device
    communicating over TCP.
    """

    port: str

    def __init__(self, address: str, timeout: float = 1.0):
        """Communicator using a TCP/IP<=>serial gateway."""
        super().__init__(timeout)
        try:
            self.address, self.port = address.split(':')
        except ValueError as e:
            raise ValueError('address must be hostname:port') from e

    async def _connect(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Asynchronously open a TCP connection with the server."""
        await self.close()
        reader, writer = await asyncio.open_connection(self.address, self.port)
        self.open = True
        return reader, writer

class SerialClient(Client):
    """Client using a directly-connected RS232 serial device."""

    baudrate: int
    bytesize: int
    stopbits: int
    parity: str

    def __init__(self,
                 address: str,
                 baudrate: int = 19200,
                 timeout: float = 10.0,
                 bytesize: int = serial.EIGHTBITS,
                 stopbits: int = serial.STOPBITS_ONE,
                 parity: str = serial.PARITY_NONE):

        super().__init__(timeout)

        self.address = address
        assert isinstance(self.address, str)

        self.baudrate = baudrate
        self.bytesize = bytesize
        self.stopbits = stopbits
        self.parity = parity

async def _connect(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Asynchronously open a TCP connection with the server."""
        await self.close()

        reader, writer = await serial_asyncio_fast.open_serial_connection(
            url = self.address,
            baudrate = self.baudrate,
            bytesize = self.bytesize,
            stopbits = self.stopbits,
            parity = self.parity,
            timeout = self.timeout
        )

        self.open = True

        return reader, writer

def _is_float(msg: Any) -> bool:
    try:
        float(msg)
        return True
    except ValueError:
        return False
