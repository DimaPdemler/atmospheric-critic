import unittest

import numpy as np
import xarray as xr

from FeatureMetric.scripts.convert_swift_zarr_to_netcdf import (
    SURFACE_VARIABLES,
    lead_hours,
    prepare_swift_forecasts,
    sample_oriented_netcdf_encoding,
)


def swift_fixture():
    values = np.zeros((2, 1, 3, 2, 3), dtype=np.float32)
    coordinates = {
        "init_time": np.asarray(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"),
        "member": [0],
        "prediction_timedelta": np.asarray([6, 12, 24], dtype="timedelta64[h]"),
        "latitude": [-1.5, 1.5],
        "longitude": [0.0, 1.5, 3.0],
    }
    return xr.Dataset({name: (("init_time", "member", "prediction_timedelta", "latitude", "longitude"), values)
                       for name in SURFACE_VARIABLES}, coords=coordinates)


class SwiftConverterTests(unittest.TestCase):
    def test_preparation_squeezes_member_and_preserves_forecast_coordinates(self):
        converted = prepare_swift_forecasts(swift_fixture(), lead_hour_values=[6, 24])
        self.assertEqual(converted["2m_temperature"].dims, (
            "time", "prediction_timedelta", "latitude", "longitude",
        ))
        self.assertNotIn("member", converted.dims)
        self.assertNotIn("_ARRAY_DIMENSIONS", converted["2m_temperature"].attrs)
        np.testing.assert_array_equal(lead_hours(converted.prediction_timedelta.values), [6, 24])
        np.testing.assert_array_equal(
            converted.time.values, np.asarray(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"),
        )

    def test_preparation_rejects_a_missing_lead(self):
        with self.assertRaisesRegex(ValueError, "unavailable"):
            prepare_swift_forecasts(swift_fixture(), lead_hour_values=[48])

    def test_netcdf_chunks_match_one_complete_forecast_sample(self):
        converted = prepare_swift_forecasts(swift_fixture())
        encoding = sample_oriented_netcdf_encoding(converted, compression_level=3)

        for variable in SURFACE_VARIABLES:
            self.assertEqual(encoding[variable]["chunksizes"], (1, 1, 2, 3))
            self.assertTrue(encoding[variable]["zlib"])
            self.assertEqual(encoding[variable]["complevel"], 3)


if __name__ == "__main__":
    unittest.main()


class ForecastEncodingTests(unittest.TestCase):
    """The layout choice that made SWIFT ~49x slower to train against."""

    def _dataset(self):
        import numpy as np
        import xarray as xr
        shape = (4, 2, 3, 5)
        return xr.Dataset(
            {"2m_temperature": (("time", "prediction_timedelta", "latitude", "longitude"),
                                np.zeros(shape, dtype="float32"))},
            coords={"time": np.arange(shape[0]), "prediction_timedelta": np.arange(shape[1]),
                    "latitude": np.arange(shape[2]), "longitude": np.arange(shape[3])},
        )

    def test_default_is_uncompressed_and_contiguous(self):
        from FeatureMetric.scripts.convert_swift_zarr_to_netcdf import forecast_encoding
        encoding = forecast_encoding(self._dataset(), 0)["2m_temperature"]
        self.assertFalse(encoding["zlib"])
        self.assertTrue(encoding["contiguous"])

    def test_compression_chunks_one_timestep_and_lead(self):
        from FeatureMetric.scripts.convert_swift_zarr_to_netcdf import forecast_encoding
        encoding = forecast_encoding(self._dataset(), 4)["2m_temperature"]
        self.assertTrue(encoding["zlib"])
        self.assertEqual(encoding["complevel"], 4)
        # (time, lead, lat, lon) -> one field per chunk, never a span of timesteps.
        self.assertEqual(encoding["chunksizes"], (1, 1, 3, 5))
