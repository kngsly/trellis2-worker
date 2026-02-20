import unittest

from worker import normalize_generation_request


class RequestNormalizationTests(unittest.TestCase):
    def test_hd_1536_native_4096_kept(self):
        normalized, adjustments = normalize_generation_request(
            {
                "mesh_profile": "hd",
                "geometry_resolution": 1536,
                "texture_generation_mode": "native_1024",
                "texture_output_size": 4096,
                "decimation_target": 1800000,
                "steps": 16,
            }
        )
        self.assertEqual(normalized["geometry_resolution"], 1536)
        self.assertEqual(normalized["texture_generation_mode"], "native_1024")
        self.assertEqual(normalized["texture_output_size"], 4096)
        self.assertEqual(normalized["pipeline_type"], "1024")
        self.assertEqual(adjustments, [])

    def test_fast_512_caps_texture_to_2048(self):
        normalized, adjustments = normalize_generation_request(
            {
                "mesh_profile": "hd",
                "geometry_resolution": 1024,
                "texture_generation_mode": "fast_512",
                "texture_output_size": 4096,
                "decimation_target": 900000,
                "steps": 12,
            }
        )
        self.assertEqual(normalized["texture_generation_mode"], "fast_512")
        self.assertEqual(normalized["texture_output_size"], 2048)
        self.assertTrue(any("fast_512" in note for note in adjustments))

    def test_512_geo_with_4096_downgrades(self):
        normalized, adjustments = normalize_generation_request(
            {
                "mesh_profile": "hd",
                "geometry_resolution": 512,
                "texture_generation_mode": "native_1024",
                "texture_output_size": 4096,
                "decimation_target": 1200000,
                "steps": 12,
            }
        )
        self.assertEqual(normalized["geometry_resolution"], 512)
        self.assertEqual(normalized["texture_output_size"], 2048)
        self.assertTrue(any("geometry_resolution < 1024" in note for note in adjustments))


if __name__ == "__main__":
    unittest.main()
