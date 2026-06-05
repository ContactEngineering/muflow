"""Tests for I/O utilities."""

import math
import tempfile
from datetime import date, datetime

import numpy as np
import xarray as xr

import json

import pytest

from muflow.io.json import (
    ExtendedJSONEncoder,
    dumps_json,
    loads_json,
)
from muflow.io.xarray import (
    load_xarray_from_bytes,
    load_xarray_from_file,
    save_xarray_to_bytes,
    save_xarray_to_file,
)


class TestExtendedJSONEncoder:
    """Tests for ExtendedJSONEncoder."""

    def test_nan(self):
        """Should encode NaN as string."""
        result = dumps_json({"value": float("nan")})
        assert '"NaN"' in result

    def test_infinity(self):
        """Should encode Infinity as string."""
        result = dumps_json({"value": float("inf")})
        assert '"Infinity"' in result

    def test_negative_infinity(self):
        """Should encode -Infinity as string."""
        result = dumps_json({"value": float("-inf")})
        assert '"-Infinity"' in result

    def test_numpy_array(self):
        """Should encode numpy arrays as lists."""
        arr = np.array([1, 2, 3])
        result = dumps_json({"arr": arr})
        assert "[1, 2, 3]" in result

    def test_numpy_int(self):
        """Should encode numpy integers."""
        result = dumps_json({"value": np.int64(42)})
        assert "42" in result

    def test_numpy_float(self):
        """Should encode numpy floats."""
        result = dumps_json({"value": np.float64(3.14)})
        assert "3.14" in result

    def test_numpy_bool(self):
        """Should encode numpy bools."""
        result = dumps_json({"value": np.bool_(True)})
        assert "true" in result

    def test_numpy_nan(self):
        """Should encode numpy NaN as string."""
        result = dumps_json({"value": np.float64("nan")})
        assert '"NaN"' in result

    def test_datetime(self):
        """Should encode datetime as ISO string."""
        dt = datetime(2023, 6, 15, 12, 30, 45)
        result = dumps_json({"dt": dt})
        assert "2023-06-15T12:30:45" in result

    def test_date(self):
        """Should encode date as ISO string."""
        d = date(2023, 6, 15)
        result = dumps_json({"date": d})
        assert "2023-06-15" in result


class TestEncoderDirectBranches:
    """Cover the ExtendedJSONEncoder methods that the normal dumps_json path
    bypasses.

    ``iterencode`` pre-converts dict/list payloads via ``_convert_floats``, so
    ``default()`` only fires for values that survive that pass — e.g. numpy
    scalars nested inside a *tuple* (which ``_convert_floats`` does not
    recurse into). And because ``np.float64`` subclasses ``float``, the
    ``np.floating`` branch (and ``_encode_np_floating``) is reached only by
    ``np.float32``.
    """

    def test_np_float32_regular_and_special(self):
        # np.float32 is np.floating but NOT a Python float, so it exercises
        # the _convert_floats -> _encode_np_floating path.
        assert "1.5" in dumps_json({"v": np.float32(1.5)})
        assert '"NaN"' in dumps_json({"v": np.float32("nan")})
        assert '"Infinity"' in dumps_json({"v": np.float32("inf")})

    def test_default_fires_for_numpy_inside_tuple(self):
        # _convert_floats leaves tuples untouched, so json serialises the tuple
        # as an array and calls default() for each numpy/​special element.
        payload = {
            "t": (
                np.int64(7),
                np.float32(2.5),
                np.bool_(True),
                np.array([1, 2]),
                float("nan"),
            )
        }
        loaded = loads_json(dumps_json(payload))
        seq = loaded["t"]
        assert seq[0] == 7
        assert seq[1] == pytest.approx(2.5)
        assert seq[2] is True
        assert seq[3] == [1, 2]
        assert math.isnan(seq[4])

    def test_encode_top_level_special_float(self):
        # encode() handles a bare top-level float (not wrapped in a container).
        assert dumps_json(float("nan")) == '"NaN"'
        assert dumps_json(float("inf")) == '"Infinity"'
        assert dumps_json(float("-inf")) == '"-Infinity"'

    def test_default_raises_for_unsupported_type(self):
        class Unserialisable:
            pass

        # Wrapped in a tuple so it reaches default(); the fallthrough to
        # super().default() raises TypeError as the base encoder would.
        with pytest.raises(TypeError):
            json.dumps({"x": (Unserialisable(),)}, cls=ExtendedJSONEncoder)


class TestLoadsJson:
    """Tests for loads_json()."""

    def test_nan_string_to_float(self):
        """Should decode 'NaN' string back to float NaN."""
        result = loads_json('{"value": "NaN"}')
        assert math.isnan(result["value"])

    def test_infinity_string_to_float(self):
        """Should decode 'Infinity' string back to float inf."""
        result = loads_json('{"value": "Infinity"}')
        assert math.isinf(result["value"])
        assert result["value"] > 0

    def test_negative_infinity_string_to_float(self):
        """Should decode '-Infinity' string back to float -inf."""
        result = loads_json('{"value": "-Infinity"}')
        assert math.isinf(result["value"])
        assert result["value"] < 0

    def test_nested_nan(self):
        """Should decode nested NaN values."""
        result = loads_json('{"outer": {"inner": "NaN"}}')
        assert math.isnan(result["outer"]["inner"])

    def test_nan_in_list(self):
        """Should decode NaN in lists."""
        result = loads_json('{"values": [1, "NaN", 3]}')
        assert result["values"][0] == 1
        assert math.isnan(result["values"][1])
        assert result["values"][2] == 3

    def test_normal_strings_unchanged(self):
        """Normal strings should not be converted."""
        result = loads_json('{"text": "Hello", "other": "world"}')
        assert result["text"] == "Hello"
        assert result["other"] == "world"


class TestRoundTrip:
    """Tests for JSON round-trip."""

    def test_roundtrip_nan(self):
        """NaN should round-trip correctly."""
        original = {"value": float("nan")}
        json_str = dumps_json(original)
        loaded = loads_json(json_str)
        assert math.isnan(loaded["value"])

    def test_roundtrip_complex_structure(self):
        """Complex structures should round-trip correctly."""
        original = {
            "normal": 42,
            "nan": float("nan"),
            "inf": float("inf"),
            "nested": {
                "array": [1, float("nan"), 3],
                "text": "hello",
            },
        }
        json_str = dumps_json(original)
        loaded = loads_json(json_str)

        assert loaded["normal"] == 42
        assert math.isnan(loaded["nan"])
        assert math.isinf(loaded["inf"])
        assert loaded["nested"]["array"][0] == 1
        assert math.isnan(loaded["nested"]["array"][1])
        assert loaded["nested"]["text"] == "hello"


class TestXarrayIO:
    """Tests for xarray I/O utilities."""

    def test_save_and_load_bytes(self):
        """Should round-trip xarray Dataset through bytes."""
        ds = xr.Dataset({
            "temperature": (["x", "y"], np.random.rand(3, 4)),
            "pressure": (["x", "y"], np.random.rand(3, 4)),
        })

        data = save_xarray_to_bytes(ds)
        assert isinstance(data, bytes)
        assert len(data) > 0

        loaded = load_xarray_from_bytes(data)
        assert "temperature" in loaded
        assert "pressure" in loaded
        np.testing.assert_array_almost_equal(
            loaded["temperature"].values,
            ds["temperature"].values,
        )

    def test_save_and_load_file(self):
        """Should round-trip xarray Dataset through file."""
        ds = xr.Dataset({
            "data": (["x"], np.array([1.0, 2.0, 3.0])),
        })

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            path = f.name

        try:
            save_xarray_to_file(ds, path)
            loaded = load_xarray_from_file(path)
            np.testing.assert_array_equal(
                loaded["data"].values,
                ds["data"].values,
            )
        finally:
            import os
            if os.path.exists(path):
                os.remove(path)

    def test_preserves_attrs(self):
        """Should preserve dataset attributes."""
        ds = xr.Dataset(
            {"data": (["x"], [1, 2, 3])},
            attrs={"description": "test dataset"},
        )

        data = save_xarray_to_bytes(ds)
        loaded = load_xarray_from_bytes(data)

        assert loaded.attrs["description"] == "test dataset"

    def test_preserves_coords(self):
        """Should preserve coordinates."""
        ds = xr.Dataset(
            {"data": (["x"], [1, 2, 3])},
            coords={"x": [10, 20, 30]},
        )

        data = save_xarray_to_bytes(ds)
        loaded = load_xarray_from_bytes(data)

        np.testing.assert_array_equal(loaded.coords["x"].values, [10, 20, 30])
