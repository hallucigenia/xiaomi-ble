"""Tests for authenticated MiBeacon frames and their state order."""

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESCCM
from home_assistant_bluetooth import BluetoothServiceInfo
from sensor_state_data import DeviceKey

from xiaomi_ble import EncryptionScheme, XiaomiBluetoothDeviceData
from xiaomi_ble.mibeacon import (
    MIBEACON_UUID,
    MiBeaconStateOrder,
    standard_mibeacon_frame_counter,
    standard_mibeacon_product_id,
)

_BINDKEY = bytes(range(16))
_ADDRESS = "12:34:56:78:9A:BC"
_WIRE_ADDRESS = bytes.fromhex("bc9a78563412")
_COUNTER = 0x123456FF
_NIGHTLIGHT = 0x07F6
_MOTION_SENSOR = 0x0A8D
_OCCUPANCY_SENSOR = 0x4683

_MOTION = DeviceKey(key="motion", device_id=None)
_LIGHT = DeviceKey(key="light", device_id=None)
_BATTERY = DeviceKey(key="battery", device_id=None)
_BUTTON = DeviceKey(key="button", device_id=None)
_RSSI = DeviceKey(key="signal_strength", device_id=None)

_MOTION_IN_DARK = bytes.fromhex("0f 00 03 01 00 00")
_MOTION_CLEAR = bytes.fromhex("17 10 04 78 00 00 00")
_LIGHT_PRESENT = bytes.fromhex("18 10 01 01")
_BATTERY_80 = bytes.fromhex("0a 10 01 50")
_BUTTON_PRESS = bytes.fromhex("01 10 03 00 00 00")


def _frame(
    counter: int = _COUNTER,
    objects: bytes = _MOTION_CLEAR,
    *,
    product_id: int = _NIGHTLIGHT,
    extra_flags: int = 0,
    extension: bytes = b"",
) -> bytes:
    sequence = counter.to_bytes(4, "little")
    header = (
        (0x5948 | extra_flags).to_bytes(2, "little")
        + product_id.to_bytes(2, "little")
        + sequence[:1]
    )
    encrypted = AESCCM(_BINDKEY, tag_length=4).encrypt(
        _WIRE_ADDRESS + header[2:5] + sequence[1:], objects, b"\x11"
    )
    return header + extension + encrypted[:-4] + sequence[1:] + encrypted[-4:]


def _advertisement(packet: bytes) -> BluetoothServiceInfo:
    return BluetoothServiceInfo(
        name="Test",
        address=_ADDRESS,
        rssi=-60,
        manufacturer_data={},
        service_data={MIBEACON_UUID: packet},
        service_uuids=[MIBEACON_UUID],
        source="test",
    )


@pytest.mark.parametrize(
    ("product_id", "extra_flags", "extension"),
    [
        pytest.param(_NIGHTLIGHT, 0, b"", id="nightlight-without-optional-fields"),
        pytest.param(_MOTION_SENSOR, 0x10, _WIRE_ADDRESS, id="motion-with-address"),
        pytest.param(_OCCUPANCY_SENSOR, 0x20, b"\x01", id="occupancy-with-capability"),
        pytest.param(
            _NIGHTLIGHT,
            0x30,
            _WIRE_ADDRESS + b"\x20\x00",
            id="address-and-io-capability",
        ),
    ],
)
def test_authentic_standard_frame(
    product_id: int, extra_flags: int, extension: bytes
) -> None:
    """Decode across supported products without updating authentication state."""
    packet = _frame(product_id=product_id, extra_flags=extra_flags, extension=extension)
    device = XiaomiBluetoothDeviceData(_BINDKEY, mibeacon_order=MiBeaconStateOrder())

    assert standard_mibeacon_product_id(packet) == product_id
    assert standard_mibeacon_frame_counter(packet) == _COUNTER
    update = device.update_mibeacon(_ADDRESS, packet)
    assert update is not None
    assert update.binary_entity_values[_MOTION].native_value is False
    assert device.encryption_scheme is EncryptionScheme.NONE
    assert not device.bindkey_verified
    assert device.decryption_failed
    assert device.pending
    assert device.last_service_info is None


@pytest.mark.parametrize(
    "header",
    [
        pytest.param(bytes.fromhex("4859f607"), id="truncated"),
        pytest.param(bytes.fromhex("c859f607ff"), id="mesh"),
        pytest.param(bytes.fromhex("4849f607ff"), id="version-four"),
        pytest.param(bytes.fromhex("4855f607ff"), id="enhanced-auth"),
        pytest.param(bytes.fromhex("4851f607ff"), id="old-auth"),
    ],
)
def test_other_headers(header: bytes) -> None:
    """Only standard-auth v5 frames have a reliable extended counter."""
    assert standard_mibeacon_product_id(header) is None
    assert standard_mibeacon_frame_counter(header) is None


@pytest.mark.parametrize(
    "packet",
    [
        pytest.param(_frame()[:13], id="short-frame"),
        pytest.param(bytes.fromhex("1059f607ff") + _WIRE_ADDRESS, id="discovery"),
        pytest.param(bytes.fromhex("4059f607ff") + bytes(14), id="unsealed-frame"),
    ],
)
def test_only_encrypted_objects_have_counters(packet: bytes) -> None:
    """Exclude packets without an authenticated object from ordering."""
    assert standard_mibeacon_product_id(packet) == _NIGHTLIGHT
    assert standard_mibeacon_frame_counter(packet) is None


@pytest.mark.parametrize(
    ("packet", "address"),
    [
        pytest.param(_frame()[:-1] + bytes([_frame()[-1] ^ 1]), _ADDRESS, id="bad-tag"),
        pytest.param(
            _frame()[:-7] + b"\x00" + _frame()[-6:], _ADDRESS, id="bad-counter"
        ),
        pytest.param(_frame(), "12:34:56", id="short-address"),
        pytest.param(_frame(), "not a MAC address", id="invalid-address-format"),
        pytest.param(_frame(), "00:00:00:00:00:00", id="bad-address"),
        pytest.param(
            _frame(extra_flags=0x10, extension=_WIRE_ADDRESS),
            "00:00:00:00:00:00",
            id="bad-included-address",
        ),
        pytest.param(
            _frame(extra_flags=0x10, extension=_WIRE_ADDRESS)[:16],
            _ADDRESS,
            id="truncated-after-address",
        ),
        pytest.param(
            _frame(extra_flags=0x30, extension=_WIRE_ADDRESS)[:18],
            _ADDRESS,
            id="missing-capability",
        ),
        pytest.param(
            _frame(extra_flags=0x20, extension=b"\x20")[:14],
            _ADDRESS,
            id="missing-io-capability",
        ),
    ],
)
def test_invalid_frame(packet: bytes, address: str) -> None:
    """Reject malformed frames and frames with an invalid tag."""
    device = XiaomiBluetoothDeviceData(_BINDKEY, mibeacon_order=MiBeaconStateOrder())
    assert device.update_mibeacon(address, packet) is None
    assert device.title is None
    assert device.pending


@pytest.mark.parametrize(
    "bindkey",
    [
        pytest.param(None, id="removed-key"),
        pytest.param(bytes(12), id="legacy-key"),
        pytest.param(bytes(16), id="different-key"),
    ],
)
def test_authentication_uses_current_key(
    bindkey: bytes | None, ordered_device: XiaomiBluetoothDeviceData
) -> None:
    """Discard the previous cipher when replacing the bindkey."""
    assert ordered_device.update_mibeacon(_ADDRESS, _frame(100)) is not None
    ordered_device.set_bindkey(bindkey)
    assert ordered_device.update_mibeacon(_ADDRESS, _frame(101)) is None


@pytest.fixture
def ordered_device() -> XiaomiBluetoothDeviceData:
    """Enable tracking for one synthetic device."""
    device = XiaomiBluetoothDeviceData(_BINDKEY)
    device.mibeacon_order = MiBeaconStateOrder()
    return device


def test_delayed_clear_after_newer_telemetry(
    ordered_device: XiaomiBluetoothDeviceData,
) -> None:
    """A newer battery measurement leaves a delayed motion update valid."""
    on = ordered_device.update(_advertisement(_frame(100, _MOTION_IN_DARK)))
    assert on.binary_entity_values[_MOTION].native_value is True
    battery = ordered_device.update(_advertisement(_frame(102, _BATTERY_80)))
    assert battery.entity_values[_BATTERY].native_value == 80
    clear_packet = _frame(101, _MOTION_CLEAR)
    clear = ordered_device.update_mibeacon(_ADDRESS, clear_packet)
    assert clear is not None
    assert clear.binary_entity_values[_MOTION].native_value is False
    assert set(clear.binary_entity_values) == {_MOTION}
    assert not clear.entity_values
    assert ordered_device.update_mibeacon(_ADDRESS, clear_packet) is None
    old_motion = ordered_device.update(_advertisement(_frame(99, _MOTION_IN_DARK)))
    assert old_motion.binary_entity_values[_MOTION].native_value is False
    assert old_motion.entity_values[_RSSI].native_value == -60


def test_shared_state_in_different_objects(
    ordered_device: XiaomiBluetoothDeviceData,
) -> None:
    """Accept the fresh motion from one object while retaining newer light."""
    ordered_device.update(_advertisement(_frame(100, _MOTION_CLEAR)))
    light = ordered_device.update(_advertisement(_frame(103, _LIGHT_PRESENT)))
    assert light.binary_entity_values[_LIGHT].native_value is True
    combined = ordered_device.update(_advertisement(_frame(102, _MOTION_IN_DARK)))
    assert combined.binary_entity_values[_MOTION].native_value is True
    assert combined.binary_entity_values[_LIGHT].native_value is True


@pytest.mark.parametrize(
    ("objects", "motion"),
    [
        pytest.param(_MOTION_IN_DARK + _MOTION_CLEAR, False, id="clear-last"),
        pytest.param(_MOTION_CLEAR + _MOTION_IN_DARK, True, id="motion-last"),
    ],
)
def test_state_order_within_frame(
    ordered_device: XiaomiBluetoothDeviceData, objects: bytes, motion: bool
) -> None:
    """The last object writing a field wins within one frame."""
    combined = ordered_device.update(_advertisement(_frame(100, objects)))
    assert combined.binary_entity_values[_MOTION].native_value is motion
    assert combined.binary_entity_values[_LIGHT].native_value is False


@pytest.mark.parametrize(
    ("previous", "counter"),
    [
        pytest.param(0x123456FF, 0x12345700, id="low-byte-rollover"),
        pytest.param(0xFFFFFFFF, 0, id="full-rollover"),
        pytest.param(31, 512, id="restart-counter-jump"),
    ],
)
def test_counter_advances(
    ordered_device: XiaomiBluetoothDeviceData, previous: int, counter: int
) -> None:
    """Accept a newer frame and suppress a late repeat of the old frame."""
    old = _advertisement(_frame(previous, _MOTION_IN_DARK))
    ordered_device.update(old)
    clear = ordered_device.update(_advertisement(_frame(counter, _MOTION_CLEAR)))
    assert clear.binary_entity_values[_MOTION].native_value is False
    assert (
        ordered_device.update(old).binary_entity_values[_MOTION].native_value is False
    )


def test_seed_stores_order_without_values(
    ordered_device: XiaomiBluetoothDeviceData,
) -> None:
    """Priming the order with cached battery data leaves motion independent."""
    cached = _frame(102, _BATTERY_80)
    assert ordered_device.update_mibeacon(_ADDRESS, cached, seed=True) is None
    assert ordered_device.title is None

    stale = ordered_device.update(_advertisement(cached))
    assert _BATTERY not in stale.entity_values
    clear = ordered_device.update_mibeacon(_ADDRESS, _frame(101, _MOTION_CLEAR))
    assert clear is not None
    assert clear.binary_entity_values[_MOTION].native_value is False


def test_ambiguous_half_range_does_not_overwrite(
    ordered_device: XiaomiBluetoothDeviceData,
) -> None:
    """Do not order counters at half the serial range."""
    ordered_device.update(_advertisement(_frame(0, _MOTION_IN_DARK)))
    ambiguous = ordered_device.update(_advertisement(_frame(0x80000000, _MOTION_CLEAR)))
    assert ambiguous.binary_entity_values[_MOTION].native_value is True


def test_invalid_supplement_does_not_advance_order_or_change_authentication(
    ordered_device: XiaomiBluetoothDeviceData,
) -> None:
    """Selected advertisements still own the authentication state."""
    selected = _advertisement(_frame(100, _MOTION_IN_DARK))
    ordered_device.update(selected)
    corrupt = _frame(1000, _MOTION_CLEAR)
    corrupt = corrupt[:-1] + bytes([corrupt[-1] ^ 1])
    assert ordered_device.update_mibeacon(_ADDRESS, corrupt) is None
    assert ordered_device.update_mibeacon(_ADDRESS, corrupt) is None
    assert not ordered_device.decryption_failed
    assert ordered_device.bindkey_verified
    assert ordered_device.last_service_info is selected

    clear = ordered_device.update_mibeacon(_ADDRESS, _frame(101, _MOTION_CLEAR))
    assert clear is not None
    assert clear.binary_entity_values[_MOTION].native_value is False


def test_supplement_leaves_events_to_regular_updates(
    ordered_device: XiaomiBluetoothDeviceData,
) -> None:
    """A supplemental frame does not consume a regular update's event."""
    packet = _frame(100, _BUTTON_PRESS + _BATTERY_80, product_id=_MOTION_SENSOR)
    supplemental = ordered_device.update_mibeacon(_ADDRESS, packet)
    assert supplemental is not None
    assert supplemental.entity_values[_BATTERY].native_value == 80
    assert not supplemental.events

    selected = _advertisement(packet)
    assert _BUTTON in ordered_device.update(selected).events
    assert _BUTTON in ordered_device.update(selected).events


def test_default_decoder_keeps_existing_behavior() -> None:
    """Without optional ordering the regular decode path is unchanged."""
    device = XiaomiBluetoothDeviceData(_BINDKEY)
    assert device.update_mibeacon(_ADDRESS, _frame()) is None
    device.update(_advertisement(_frame(101, _MOTION_CLEAR)))
    late = device.update(_advertisement(_frame(100, _MOTION_IN_DARK)))
    assert late.binary_entity_values[_MOTION].native_value is True
