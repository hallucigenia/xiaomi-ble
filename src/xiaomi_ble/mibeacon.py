"""Frame ordering for standard-auth MiBeacon devices."""

from sensor_state_data import DeviceKey

from .const import SERVICE_MIBEACON

MIBEACON_UUID = SERVICE_MIBEACON

_VERSION_AUTH_MESH_MASK = 0xFC80
_STANDARD_AUTH_V5 = 0x5800
_ENCRYPTED_OBJECT = 0x48


def standard_mibeacon_product_id(packet: bytes) -> int | None:
    """Read the product ID of a non-Mesh, standard-auth v5 frame."""
    if (
        len(packet) >= 5
        and int.from_bytes(packet[:2], "little") & _VERSION_AUTH_MESH_MASK
        == _STANDARD_AUTH_V5
    ):
        return int.from_bytes(packet[2:4], "little")
    return None


def standard_mibeacon_frame_counter(packet: bytes) -> int | None:
    """Read the counter of an encrypted object frame before authentication."""
    if (
        len(packet) >= 14
        and standard_mibeacon_product_id(packet) is not None
        and packet[0] & _ENCRYPTED_OBJECT == _ENCRYPTED_OBJECT
    ):
        return int.from_bytes(packet[4:5] + packet[-7:-4], "little")
    return None


def is_newer_frame(counter: int, previous: int | None) -> bool:
    """Compare the device's wrapping 32-bit counter."""
    return previous is None or 0 < (counter - previous) % (1 << 32) < (1 << 31)


class MiBeaconStateOrder:
    """Track authenticated frames separately for each state field."""

    def __init__(self) -> None:
        self._counters: dict[tuple[str, DeviceKey], int] = {}

    def accept(self, field: tuple[str, DeviceKey], counter: int) -> bool:
        """Record a newer authenticated frame for a field."""
        if not is_newer_frame(counter, self._counters.get(field)):
            return False
        self._counters[field] = counter
        return True
