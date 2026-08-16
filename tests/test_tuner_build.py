from __future__ import annotations

import unittest

from tuner.build import BUILD_VERSION, modules_to_rebuild


class BuildPlanningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.campaign = {
            "champion": {
                "modules": {
                    "gate": {"bits": 4, "group_size": 64},
                    "up": {"bits": 3, "group_size": 64},
                    "fc": {"bits": 6, "group_size": 64},
                }
            }
        }

    def test_build_version_is_v2_or_later(self) -> None:
        self.assertGreaterEqual(BUILD_VERSION, 2)

    def test_only_mutated_module_is_rebuilt(self) -> None:
        candidate = {
            "modules": {
                "gate": {"bits": 4, "group_size": 64},
                "up": {"bits": 3, "group_size": 64},
                "fc": {"bits": 5, "group_size": 64},
            }
        }
        self.assertEqual(modules_to_rebuild(self.campaign, candidate), ["fc"])

    def test_group_mutation_is_detected(self) -> None:
        candidate = {
            "modules": {
                "gate": {"bits": 4, "group_size": 32},
                "up": {"bits": 3, "group_size": 64},
                "fc": {"bits": 6, "group_size": 64},
            }
        }
        self.assertEqual(modules_to_rebuild(self.campaign, candidate), ["gate"])


if __name__ == "__main__":
    unittest.main()
