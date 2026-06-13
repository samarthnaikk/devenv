import unittest


class MemoryImportsTest(unittest.TestCase):
    def test_memory_engine_is_importable(self) -> None:
        from core.memory import MemoryEngine

        self.assertIsNotNone(MemoryEngine)

