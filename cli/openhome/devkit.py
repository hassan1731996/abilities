"""DevKit onboarding over Bluetooth, and DevKit status from OpenHome.

Nothing here prints: functions return values, raise DevKitError subclasses,
and report progress through an optional ``on_progress`` callback.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from bleak import BleakClient, BleakScanner
from bleak import exc as bleak_exc
from bleak.exc import BleakDeviceNotFoundError, BleakError

from .config import Config
from .endpoints import devkit_socket
from .errors import (
    ApiKeyRejected,
    ConnectionLost,
    DeviceNotFound,
    DeviceOffline,
    DevKitError,
    NotAuthenticatedError,
    ScanFailed,
    WifiFailed,
)

# Only newer bleak releases raise this.
_BluetoothUnavailable = getattr(bleak_exc, "BleakBluetoothNotAvailableError", None)

# ── GATT contract (fixed by the DevKit firmware; payloads are UTF-8 JSON) ──
SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
WIFI_SCAN_UUID = "12345678-1234-5678-1234-56789abcdef1"
WIFI_CONNECT_UUID = "12345678-1234-5678-1234-56789abcdef2"
WIFI_STATUS_UUID = "12345678-1234-5678-1234-56789abcdef3"
API_KEY_UUID = "12345678-1234-5678-1234-56789abcdef4"
HEARTBEAT_UUID = "12345678-1234-5678-1234-56789abcdef5"
# Never written: a successful write restarts the DevKit's Bluetooth service.
ENV_CONFIG_UUID = "12345678-1234-5678-1234-56789abcdef6"

# DevKits advertise as openhome_<last 6 characters of the MAC address>.
DEVICE_PREFIX = "openhome"

SCAN_TIMEOUT = 10.0
CONNECT_TIMEOUT = 20.0
CONNECT_ATTEMPTS = 5
NETWORK_SCAN_TIMEOUT = 45.0
# A rejected join, including the fallback to the previous network, can take
# close to a minute.
WIFI_JOIN_TIMEOUT = 90.0
API_KEY_TIMEOUT = 20.0
CLOUD_TIMEOUT = 20.0
# bleak does not time out reads and writes on its own.
READ_TIMEOUT = 10.0
WRITE_TIMEOUT = 15.0
OP_TIMEOUT = 10.0  # subscribe, unsubscribe, scanner start/stop, disconnect

_OPEN_SECURITY = {"", "open", "none", "-"}

Progress = Callable[[str], None]


def _is_devkit(name: str) -> bool:
    return name.strip().lower().startswith(DEVICE_PREFIX)


# ── value types ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Device:
    name: str
    address: str
    rssi: int | None = None


@dataclass(frozen=True)
class Network:
    ssid: str
    signal: int = 0
    security: str = ""
    bssid: str = ""

    @property
    def is_open(self) -> bool:
        return self.security.strip().lower() in _OPEN_SECURITY


@dataclass(frozen=True)
class WifiStatus:
    state: str = "unknown"
    message: str = ""
    reason: str = ""          # "invalid_password" | "failed"
    reverted: bool = False    # firmware rolled back to the previous network
    reverted_ssid: str = ""

    @property
    def is_connected(self) -> bool:
        return self.state.lower() == "connected"

    @property
    def is_final(self) -> bool:
        return self.state.lower() in ("connected", "error", "failed")

    def explain(self, ssid: str) -> str:
        # Our own wording: the firmware's messages are for its logs, not users.
        if self.reason == "invalid_password":
            text = f"Wrong password for {ssid!r}."
        else:
            # The firmware often can't tell a bad password from other join
            # failures, and a bad password is by far the likeliest cause.
            text = f"The DevKit couldn't connect to {ssid!r}. Check the password and try again."
        if self.reverted:
            where = f" {self.reverted_ssid!r}" if self.reverted_ssid else " its previous network"
            text += f" It's still connected to{where}."
        return text


@dataclass(frozen=True)
class ApiKeyStatus:
    state: str = ""
    configured: bool = False
    message: str = ""
    key_prefix: str = ""

    @property
    def is_verified(self) -> bool:
        # `configured` can be left over from an earlier key, so it is never
        # sufficient on its own — only a fresh `success` proves the current key.
        return self.state.lower() == "success"


@dataclass(frozen=True)
class Metric:
    used: float = 0.0
    total: float = 0.0
    unit: str = ""

    @property
    def percent(self) -> int | None:
        return round(self.used / self.total * 100) if self.total else None


@dataclass
class CloudStatus:
    online: bool = False
    detail: str = ""
    firmware: str = ""
    ip_address: str = ""
    agent_connected: bool = False
    mqtt_running: bool = False
    local_mode: bool = False
    timestamp: str = ""
    metrics: dict[str, Metric] = field(default_factory=dict)


# ── error translation ───────────────────────────────────────────────────
def _detail(what: str, exc: BaseException | None = None) -> str:
    """Technical detail, shown only when OPENHOME_DEBUG is set."""
    if not os.environ.get("OPENHOME_DEBUG"):
        return ""
    return f" [{what}: {type(exc).__name__}: {exc}]" if exc else f" [{what}]"


@contextlib.contextmanager
def _ble_errors(what: str):
    """Translate bleak and OS errors into DevKitError subclasses."""
    try:
        yield
    except DevKitError:
        raise
    except BleakDeviceNotFoundError as exc:
        raise DeviceNotFound(
            "The DevKit is out of range or switched off." + _detail(what, exc)
        ) from exc
    except BleakError as exc:
        if _BluetoothUnavailable and isinstance(exc, _BluetoothUnavailable):
            raise DevKitError(
                "Bluetooth is off or unavailable on this computer." + _detail(what, exc)
            ) from exc
        raise ConnectionLost("Lost the connection to the DevKit." + _detail(what, exc)) from exc
    except asyncio.TimeoutError as exc:
        raise DevKitError("The DevKit didn't respond in time." + _detail(what, exc)) from exc
    except OSError as exc:
        raise DevKitError(
            "Bluetooth is off or unavailable on this computer." + _detail(what, exc)
        ) from exc


def _decode(data: bytes | bytearray, what: str) -> dict[str, Any]:
    try:
        payload = json.loads(bytes(data).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DevKitError(
            "The DevKit sent a response we couldn't read." + _detail(what, exc)
        ) from exc
    if not isinstance(payload, dict):
        raise DevKitError("The DevKit sent a response we couldn't read." + _detail(what))
    return payload


# ── pairing ─────────────────────────────────────────────────────────────
# The DevKit requests pairing when a client connects. Without an agent to
# answer, BlueZ refuses and the link drops about a second later.
_AGENT_PATH = "/org/openhome/devkit/agent"


def _make_agent_class():
    from dbus_fast.service import ServiceInterface, method

    # Annotations are D-Bus type signatures ("o" object path, "s" string,
    # "u" uint32, "q" uint16) that dbus-fast reads to build the interface.
    class _PairingAgent(ServiceInterface):
        def __init__(self):
            super().__init__("org.bluez.Agent1")

        @method()
        def Release(self):
            pass

        @method()
        def RequestAuthorization(self, device: "o"):  # noqa: F821
            pass

        @method()
        def AuthorizeService(self, device: "o", uuid: "s"):  # noqa: F821
            pass

        @method()
        def RequestConfirmation(self, device: "o", passkey: "u"):  # noqa: F821
            pass

        @method()
        def RequestPinCode(self, device: "o") -> "s":  # noqa: F821
            return "0000"

        @method()
        def RequestPasskey(self, device: "o") -> "u":  # noqa: F821
            return 0

        @method()
        def DisplayPinCode(self, device: "o", pincode: "s"):  # noqa: F821
            pass

        @method()
        def DisplayPasskey(self, device: "o", passkey: "u", entered: "q"):  # noqa: F821
            pass

        @method()
        def Cancel(self):
            pass

    return _PairingAgent


@contextlib.asynccontextmanager
async def pairing_agent():
    """Register a Just Works pairing agent for the duration of a block.

    The DevKit asks to pair when a client connects, and on Linux nothing
    answers unless the client registers an agent. No-op on other platforms.
    """
    if not sys.platform.startswith("linux"):
        yield False
        return
    try:
        from dbus_fast import BusType
        from dbus_fast.aio import MessageBus
    except ImportError:
        yield False
        return

    bus = manager = None
    try:
        bus = await MessageBus(bus_type=BusType.SYSTEM).connect()
        bus.export(_AGENT_PATH, _make_agent_class()())
        introspection = await bus.introspect("org.bluez", "/org/bluez")
        manager = bus.get_proxy_object(
            "org.bluez", "/org/bluez", introspection
        ).get_interface("org.bluez.AgentManager1")
        await manager.call_register_agent(_AGENT_PATH, "NoInputNoOutput")
        await manager.call_request_default_agent(_AGENT_PATH)
    except Exception:  # any D-Bus failure: carry on without an agent
        if bus is not None:
            with contextlib.suppress(Exception):
                bus.disconnect()
        yield False
        return

    try:
        yield True
    finally:
        with contextlib.suppress(Exception):
            await manager.call_unregister_agent(_AGENT_PATH)
        with contextlib.suppress(Exception):
            bus.disconnect()


# ── discovery ───────────────────────────────────────────────────────────
async def scan_devices(
    timeout: float = SCAN_TIMEOUT,
    *,
    strict: bool = True,
    on_progress: Progress | None = None,
) -> list[Device]:
    """Scan for DevKits, stopping shortly after the first one appears."""
    note = on_progress or (lambda _m: None)
    found: dict[str, Device] = {}
    settle_deadline: float | None = None
    loop = asyncio.get_running_loop()

    def seen(device, adv) -> None:
        nonlocal settle_deadline
        name = (adv.local_name or device.name or "").strip()
        if device.address in found:
            return
        if strict and not _is_devkit(name):
            return
        found[device.address] = Device(name or "(unnamed)", device.address, adv.rssi)
        note(f"found {name or 'a DevKit'}")
        if strict and settle_deadline is None:
            # Give any sibling DevKits a moment to show up before we stop.
            settle_deadline = loop.time() + 2.0

    with _ble_errors("scanning for DevKits"):
        scanner = BleakScanner(detection_callback=seen)
        await asyncio.wait_for(scanner.start(), timeout=OP_TIMEOUT)
        try:
            deadline = loop.time() + timeout
            while loop.time() < deadline:
                if settle_deadline is not None and loop.time() >= settle_deadline:
                    break
                await asyncio.sleep(0.2)
        finally:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(scanner.stop(), timeout=OP_TIMEOUT)

    return sorted(found.values(), key=lambda d: d.rssi or -999, reverse=True)


def forget_device(address: str) -> None:
    """Clear BlueZ's cached record for a device (Linux only, best effort)."""
    if not sys.platform.startswith("linux") or not address:
        return
    try:
        subprocess.run(
            ["bluetoothctl", "remove", address],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        pass  # bluetoothctl missing or wedged: nothing we can do, and not fatal


# ── the device ──────────────────────────────────────────────────────────
class DevKit:
    """A connected DevKit. Use as an async context manager."""

    def __init__(self, target: Device | str, *, on_progress: Progress | None = None):
        self.address = target.address if isinstance(target, Device) else target
        self.name = target.name if isinstance(target, Device) else self.address
        self._note = on_progress or (lambda _m: None)
        self._client: Any = None
        self._agent = contextlib.AsyncExitStack()

    # -- lifecycle -------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return bool(self._client and self._client.is_connected)

    async def connect(self, attempts: int = CONNECT_ATTEMPTS) -> None:
        last: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                client = BleakClient(self.address, timeout=CONNECT_TIMEOUT)
                with _ble_errors("connecting"):
                    # bleak honours its own timeout; this is a backstop.
                    await asyncio.wait_for(client.connect(), timeout=CONNECT_TIMEOUT + 5)
                self._client = client
                if not self._has_service():
                    # BlueZ can report "connected" over a stale record or the
                    # wrong bearer, with no usable GATT. Treat that as failure
                    # rather than handing the caller a dead client.
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(client.disconnect(), timeout=OP_TIMEOUT)
                    self._client = None
                    raise ConnectionLost("Couldn't set up the connection to the DevKit.")
                await self._subscribe_heartbeat()
                return
            except (ConnectionLost, DeviceNotFound, DevKitError) as exc:
                last = exc
                if attempt < attempts:
                    delay = min(4.0, 1.0 * 1.6 ** (attempt - 1))
                    self._note(f"still trying to connect ({attempt + 1}/{attempts})…")
                    forget_device(self.address)
                    await asyncio.sleep(delay)
        raise DeviceNotFound(
            "Couldn't connect to the DevKit. Make sure it's switched on and "
            "close by, then try again." + _detail("connect", last)
        )

    async def reconnect(self) -> None:
        await self.disconnect()
        await self.connect()

    async def disconnect(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(client.disconnect(), timeout=OP_TIMEOUT)

    async def __aenter__(self) -> "DevKit":
        # The agent must be live before the first connect: the DevKit asks to
        # pair immediately, and an unanswered request kills the link.
        await self._agent.__aenter__()
        await self._agent.enter_async_context(pairing_agent())
        try:
            await self.connect()
        except BaseException:
            await self._agent.aclose()
            raise
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.disconnect()
        with contextlib.suppress(Exception):
            await self._agent.aclose()

    async def _write(self, client: Any, uuid: str, payload: bytes, what: str) -> None:
        """Write a characteristic; a write that never completes counts as a lost link."""
        with _ble_errors(what):
            try:
                await asyncio.wait_for(client.write_gatt_char(uuid, payload), timeout=WRITE_TIMEOUT)
            except asyncio.TimeoutError as exc:
                raise ConnectionLost(
                    "The DevKit didn't respond in time." + _detail(what, exc)
                ) from exc

    def _has_service(self) -> bool:
        try:
            services = list(self._client.services or [])
        except Exception:
            return False
        want = SERVICE_UUID.lower()
        return any(str(svc.uuid).lower() == want for svc in services)

    @contextlib.contextmanager
    def muted(self):
        """Silence progress notes for a block, e.g. while a spinner is shown."""
        note, self._note = self._note, (lambda _m: None)
        try:
            yield
        finally:
            self._note = note

    def _need_link(self) -> Any:
        if not self.is_connected:
            raise ConnectionLost("Lost the connection to the DevKit.")
        return self._client

    async def _subscribe_heartbeat(self) -> None:
        # Non-fatal: some firmware returns UNLIKELY_ERROR here while the link
        # is perfectly healthy.
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                self._client.start_notify(HEARTBEAT_UUID, lambda *_a: None), timeout=OP_TIMEOUT
            )

    @contextlib.asynccontextmanager
    async def _notifications(self, uuid: str, handler):
        """Subscribe for the duration of a block. Unsubscribing never raises."""
        client = self._need_link()
        subscribed = False
        try:
            with _ble_errors("subscribing to updates"):
                await asyncio.wait_for(client.start_notify(uuid, handler), timeout=OP_TIMEOUT)
            subscribed = True
        except DevKitError:
            pass  # subscribing can fail on its own; callers decide what that means
        try:
            yield subscribed
        finally:
            if subscribed:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(client.stop_notify(uuid), timeout=OP_TIMEOUT)

    # -- wifi scan -------------------------------------------------------
    async def scan_networks(self, timeout: float = NETWORK_SCAN_TIMEOUT) -> list[Network]:
        """Ask the DevKit to scan for WiFi; results arrive one network per notification."""
        client = self._need_link()
        networks: list[Network] = []
        done = asyncio.Event()
        failure: list[str] = []

        def handler(_sender, data) -> None:
            try:
                msg = _decode(data, "scanning for networks")
            except DevKitError:
                return  # a malformed frame should not abort the whole scan
            status = msg.get("status")
            if status in ("initiated", "scanning"):
                networks.clear()  # the device restarted the list
            elif status == "network":
                net = msg.get("network") or {}
                networks.append(
                    Network(
                        ssid=str(net.get("ssid") or ""),
                        signal=int(net.get("signal") or 0),
                        security=str(net.get("security") or ""),
                        bssid=str(net.get("bssid") or ""),
                    )
                )
                total = msg.get("total") or 0
                self._note(f"  {len(networks)}/{total} {networks[-1].ssid}")
            elif status == "complete":
                done.set()
            elif status == "error":
                failure.append(str(msg.get("message") or "the device did not say why"))
                done.set()

        async with self._notifications(WIFI_SCAN_UUID, handler) as subscribed:
            if not subscribed:
                # Results only ever arrive as notifications; without them there
                # is nothing to wait for, so fail now rather than after 45s.
                raise ScanFailed("The DevKit couldn't scan for WiFi networks. Please try again.")
            await self._write(client, WIFI_SCAN_UUID, b"\x01", "starting the network scan")
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(done.wait(), timeout=timeout)

        if failure:
            raise ScanFailed(
                "The DevKit couldn't scan for WiFi networks. Please try again."
                + _detail("scan", RuntimeError(failure[0]))
            )
        if not done.is_set() and not networks:
            raise ScanFailed("The DevKit didn't find any WiFi networks. Please try again.")
        return _tidy(networks)

    # -- wifi status / join ----------------------------------------------
    async def _read_json(self, uuid: str, what: str) -> dict[str, Any]:
        client = self._need_link()
        with _ble_errors(what):
            raw = await asyncio.wait_for(
                client.read_gatt_char(uuid), timeout=READ_TIMEOUT
            )
        return _decode(raw, what)

    async def wifi_status(self) -> WifiStatus:
        return _wifi_status(await self._read_json(WIFI_STATUS_UUID, "reading WiFi status"))

    async def join_wifi(
        self, ssid: str, password: str = "", timeout: float = WIFI_JOIN_TIMEOUT
    ) -> WifiStatus | None:
        """Send WiFi credentials and return the DevKit's verdict on this attempt.

        Raises WifiFailed if the DevKit reports a failure, and returns None if no
        verdict arrives within ``timeout``.
        """
        client = self._need_link()
        done = asyncio.Event()
        latest: list[WifiStatus] = []
        loop = asyncio.get_running_loop()
        written_at = 0.0
        armed = False  # seen this attempt's "connecting" state

        def belongs_here(status: WifiStatus) -> bool:
            # The DevKit keeps its last verdict between attempts, so a result
            # counts only if it names this network and follows this attempt's
            # "connecting" (which the firmware reports first, within seconds).
            if status.message and ssid.lower() not in status.message.lower():
                return False
            return armed or (loop.time() - written_at) >= 3.0

        def consider(status: WifiStatus) -> bool:
            nonlocal armed
            if not status.is_final:
                if status.state.lower() == "connecting":
                    armed = True
                return False
            if belongs_here(status):
                latest.append(status)
                return True
            return False

        def handler(_sender, data) -> None:
            try:
                status = _wifi_status(_decode(data, "joining WiFi"))
            except DevKitError:
                return
            if consider(status):
                done.set()

        payload = json.dumps({"ssid": ssid, "password": password or ""}).encode("utf-8")
        async with self._notifications(WIFI_STATUS_UUID, handler):
            # A failure here means the write never landed, which is a real error.
            await self._write(client, WIFI_CONNECT_UUID, payload, "sending WiFi credentials")
            written_at = loop.time()

            deadline = written_at + timeout
            while not done.is_set() and loop.time() < deadline:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(done.wait(), timeout=2.0)
                if done.is_set():
                    break
                if not self.is_connected:
                    # Joining WiFi can drop the link; the verdict survives it.
                    try:
                        await self.reconnect()
                    except DevKitError:
                        continue  # keep trying until the deadline
                try:
                    polled = await self.wifi_status()
                except DevKitError:
                    continue
                if consider(polled):
                    break

        status = latest[-1] if latest else None
        if status is None:
            return None
        if status.is_connected:
            return status
        raise WifiFailed(status.explain(ssid))

    # -- api key ---------------------------------------------------------
    async def api_key_status(self) -> ApiKeyStatus:
        return _api_key_status(
            await self._read_json(API_KEY_UUID, "reading API key status")
        )

    async def set_api_key(
        self, api_key: str, timeout: float = API_KEY_TIMEOUT
    ) -> ApiKeyStatus | None:
        """Send the API key and return the DevKit's verdict.

        Raises ApiKeyRejected if the DevKit refuses it, and returns None if no
        verdict arrives within ``timeout``.
        """
        client = self._need_link()
        done = asyncio.Event()
        latest: list[ApiKeyStatus] = []

        def handler(_sender, data) -> None:
            try:
                status = _api_key_status(_decode(data, "verifying the API key"))
            except DevKitError:
                return
            latest.append(status)
            if status.state.lower() in ("success", "invalid", "error"):
                done.set()

        payload = json.dumps({"api_key": api_key}).encode("utf-8")
        async with self._notifications(API_KEY_UUID, handler):
            await self._write(client, API_KEY_UUID, payload, "sending the API key")
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(done.wait(), timeout=timeout)

        status = latest[-1] if latest else None
        # A final notification is the device's answer - don't second-guess it.
        # Only when none arrived do we re-read, and then trust just `state`:
        # `configured` can be left over from an earlier key.
        if status is None:
            if not self.is_connected:
                return None
            await asyncio.sleep(0.5)
            try:
                status = await self.api_key_status()
            except DevKitError:
                return None
        if status.is_verified:
            return status
        raise ApiKeyRejected(
            "That API key wasn't accepted. Check it and try again."
            + _detail("api key", RuntimeError(status.message) if status.message else None)
        )


def _wifi_status(msg: dict[str, Any]) -> WifiStatus:
    return WifiStatus(
        state=str(msg.get("state") or "unknown"),
        message=str(msg.get("message") or ""),
        reason=str(msg.get("reason") or ""),
        reverted=bool(msg.get("reverted")),
        reverted_ssid=str(msg.get("reverted_ssid") or ""),
    )


def _api_key_status(msg: dict[str, Any]) -> ApiKeyStatus:
    return ApiKeyStatus(
        state=str(msg.get("state") or ""),
        configured=bool(msg.get("configured")),
        message=str(msg.get("message") or ""),
        key_prefix=str(msg.get("key_prefix") or ""),
    )


def _tidy(networks: list[Network]) -> list[Network]:
    """Drop hidden SSIDs, keep the strongest BSSID per SSID, strongest first."""
    best: dict[str, Network] = {}
    for net in networks:
        if not net.ssid.strip():
            continue
        current = best.get(net.ssid)
        if current is None or net.signal > current.signal:
            best[net.ssid] = net
    return sorted(best.values(), key=lambda n: n.signal, reverse=True)


# ── cloud health ────────────────────────────────────────────────────────
async def cloud_status(
    config: Config | None = None, *, timeout: float = CLOUD_TIMEOUT
) -> CloudStatus:
    """Fetch the account's DevKit status from OpenHome."""
    cfg = config or Config.from_env()
    if not cfg.api_key:
        raise NotAuthenticatedError("You're not signed in.")

    try:
        import websockets
    except ImportError as exc:  # pragma: no cover
        raise DevKitError(
            "DevKit status isn't available in this installation. Reinstall the OpenHome CLI."
        ) from exc

    url = f"{cfg.ws_base}{devkit_socket(cfg.api_key)}"
    status = CloudStatus(detail="Your DevKit isn't online right now.")
    try:
        async with websockets.connect(url, open_timeout=timeout) as ws:
            await ws.send("frontend")  # identify as the dashboard, not the device
            await ws.send(json.dumps({"command": "device_stats"}))
            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout
            while loop.time() < deadline:
                remaining = deadline - loop.time()
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                try:
                    msg = json.loads(raw)
                except (TypeError, ValueError):
                    continue  # non-JSON keepalives
                if not isinstance(msg, dict):
                    continue
                if isinstance(msg.get("device_status"), str):
                    state = msg["device_status"]
                    status.online = state == "connected"
                    status.detail = (
                        "DevKit connected." if status.online
                        else "Your DevKit isn't online right now."
                    )
                    if not status.online:
                        break
                if msg.get("response") == "device_stats" and isinstance(msg.get("data"), dict):
                    return _cloud_stats(msg["data"])
    except OSError as exc:
        raise DevKitError("Couldn't reach OpenHome. Check your internet connection.") from exc
    except Exception as exc:  # websockets raises a wide family of its own
        if isinstance(exc, DevKitError):
            raise
        text = str(exc)
        # A rejected upgrade carries the HTTP status; 5xx is the service being
        # down, which is not something the user can fix by retrying harder.
        code = next((c for c in ("500", "502", "503", "504") if c in text), "")
        if code:
            raise DevKitError(
                "OpenHome is temporarily unavailable. Please try again shortly."
                + _detail("status", exc)
            ) from exc
        if "401" in text or "403" in text:
            raise DevKitError(
                "Your API key wasn't accepted. Run `openhome login` to sign in again."
            ) from exc
        raise DevKitError(
            "Couldn't reach OpenHome. Please try again." + _detail("status", exc)
        ) from exc

    if not status.online:
        raise DeviceOffline(status.detail)
    return status


async def wait_until_online(
    config: Config | None = None,
    *,
    timeout: float = 90.0,
    on_progress: Progress | None = None,
) -> CloudStatus:
    """Poll OpenHome until the DevKit reports in, or give up."""
    note = on_progress or (lambda _m: None)
    cfg = config or Config.from_env()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    attempt = 0
    unreachable: DevKitError | None = None
    strikes = 0
    while loop.time() < deadline:
        attempt += 1
        try:
            return await cloud_status(cfg, timeout=min(15.0, max(5.0, deadline - loop.time())))
        except DeviceOffline:
            strikes = 0
            note(f"still starting up (check {attempt})…")
        except DevKitError as exc:
            # The cloud itself is failing, not the device. Retrying a service
            # that is down just makes the user wait, so give up after a few.
            # Say nothing here: the caller reports this once, when we give up.
            unreachable = exc
            strikes += 1
            if strikes >= 3:
                raise exc
        await asyncio.sleep(5.0)
    if unreachable is not None and strikes:
        raise unreachable
    raise DeviceOffline("The DevKit hasn't come online yet.")


def _cloud_stats(data: dict[str, Any]) -> CloudStatus:
    hw = data.get("hardware_stats") or {}
    metrics = {}
    for key in ("cpu", "ram", "disk"):
        raw = hw.get(key)
        if isinstance(raw, dict) and isinstance(raw.get("total"), (int, float)):
            metrics[key] = Metric(
                used=float(raw.get("used") or 0),
                total=float(raw["total"]),
                unit=str(raw.get("unit") or ""),
            )
    return CloudStatus(
        online=True,
        detail="DevKit connected.",
        firmware=str(data.get("firmware_version") or ""),
        ip_address=str(data.get("ip_address") or ""),
        agent_connected=bool((data.get("agent_stats") or {}).get("connected")),
        mqtt_running=bool((data.get("mqtt") or {}).get("running")),
        local_mode=bool(data.get("is_local_mode")),
        timestamp=str(data.get("timestamp") or ""),
        metrics=metrics,
    )
