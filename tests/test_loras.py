import unittest

from z_image.loras import clean_example_prompt, catalog_prompts


class CleanExamplePromptTests(unittest.TestCase):
    def test_strips_trigger_prefix_with_comma(self) -> None:
        raw = "Studio Ghibli Dark Fairytale, a lonely child in a forest"
        self.assertEqual(
            clean_example_prompt(raw, "Studio Ghibli Dark Fairytale"),
            "a lonely child in a forest",
        )

    def test_strips_trigger_without_comma(self) -> None:
        raw = "G0thicL1nes mythic warrior on a cliff"
        self.assertEqual(
            clean_example_prompt(raw, "G0thicL1nes"),
            "mythic warrior on a cliff",
        )

    def test_strips_leading_punctuation(self) -> None:
        self.assertEqual(
            clean_example_prompt(",; candid street portrait", ""),
            "candid street portrait",
        )

    def test_strips_trailing_double_punctuation(self) -> None:
        self.assertEqual(
            clean_example_prompt("neon alley at night,,", ""),
            "neon alley at night",
        )
        self.assertEqual(
            clean_example_prompt("moody portrait..", ""),
            "moody portrait",
        )

    def test_preserves_internal_punctuation(self) -> None:
        self.assertEqual(
            clean_example_prompt("woman laughing, urban background", ""),
            "woman laughing, urban background",
        )


class CatalogPromptsTests(unittest.TestCase):
    def test_dedupes_after_cleaning(self) -> None:
        entry = {
            "trigger": "TinyDaal",
            "prompts": [
                "TinyDaal tiny robot on a ledge",
                "TinyDaal, tiny robot on a ledge",
            ],
        }
        self.assertEqual(catalog_prompts(entry), ["tiny robot on a ledge"])


if __name__ == "__main__":
    unittest.main()
