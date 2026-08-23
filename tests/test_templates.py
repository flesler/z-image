import unittest

from z_image.templates import (
    ResolvedPrompt,
    all_variants,
    combo_count,
    has_placeholders,
    parse_groups,
    resolve_prompt,
)


EXAMPLE = "photo of a {man|woman|dog} eating {an apple|a slice of pizza}"


class TemplateTests(unittest.TestCase):
    def test_has_placeholders(self) -> None:
        self.assertTrue(has_placeholders(EXAMPLE))
        self.assertFalse(has_placeholders("plain prompt"))
        self.assertFalse(has_placeholders("literal {braces} without pipes"))
        self.assertFalse(has_placeholders("{only one option}"))

    def test_parse_groups(self) -> None:
        self.assertEqual(parse_groups(EXAMPLE), [["man", "woman", "dog"], ["an apple", "a slice of pizza"]])
        self.assertEqual(parse_groups("no groups here"), [])
        self.assertEqual(parse_groups("{red|green} light"), [["red", "green"]])

    def test_combo_count(self) -> None:
        self.assertEqual(combo_count(parse_groups(EXAMPLE)), 6)
        self.assertEqual(combo_count([["a", "b", "c"]]), 3)
        self.assertEqual(combo_count([]), 1)

    def test_cycles_all_combinations(self) -> None:
        seen: set[tuple[str, str]] = set()
        for seed in range(100, 106):
            resolved = resolve_prompt(EXAMPLE, seed, base_seed=100)
            self.assertIsNotNone(resolved.template)
            man_match = resolved.prompt.split(" eating ", 1)
            self.assertEqual(len(man_match), 2)
            seen.add((man_match[0], man_match[1]))

        self.assertEqual(
            seen,
            {
                ("photo of a man", "an apple"),
                ("photo of a woman", "an apple"),
                ("photo of a dog", "an apple"),
                ("photo of a man", "a slice of pizza"),
                ("photo of a woman", "a slice of pizza"),
                ("photo of a dog", "a slice of pizza"),
            },
        )

    def test_incrementing_seeds_cycle_without_base(self) -> None:
        prompts = [resolve_prompt(EXAMPLE, seed).prompt for seed in range(1049160638, 1049160644)]
        self.assertEqual(len(set(prompts)), 6)

    def test_all_variants(self) -> None:
        variants = all_variants(EXAMPLE)
        self.assertIsNotNone(variants)
        assert variants is not None
        self.assertEqual(len(variants), 6)
        self.assertEqual(variants[0], "photo of a man eating an apple")
        self.assertEqual(variants[1], "photo of a woman eating an apple")
        self.assertEqual(all_variants("plain prompt"), None)

    def test_leftmost_group_changes_fastest(self) -> None:
        resolved = resolve_prompt(EXAMPLE, 100, base_seed=100)
        self.assertEqual(resolved.prompt, "photo of a man eating an apple")

        resolved = resolve_prompt(EXAMPLE, 101, base_seed=100)
        self.assertEqual(resolved.prompt, "photo of a woman eating an apple")

        resolved = resolve_prompt(EXAMPLE, 103, base_seed=100)
        self.assertEqual(resolved.prompt, "photo of a man eating a slice of pizza")

    def test_wraps_after_full_cycle(self) -> None:
        first = resolve_prompt(EXAMPLE, 100, base_seed=100)
        again = resolve_prompt(EXAMPLE, 106, base_seed=100)
        self.assertEqual(first.prompt, again.prompt)

    def test_multi_word_options_keep_spaces(self) -> None:
        text = "a {tall dark stranger|short cheerful friend} in {New York City|Los Angeles}"
        resolved = resolve_prompt(text, 0, base_seed=0)
        self.assertEqual(resolved.prompt, "a tall dark stranger in New York City")

        resolved = resolve_prompt(text, 1, base_seed=0)
        self.assertEqual(resolved.prompt, "a short cheerful friend in New York City")

        resolved = resolve_prompt(text, 2, base_seed=0)
        self.assertEqual(resolved.prompt, "a tall dark stranger in Los Angeles")

    def test_spaces_inside_braces_are_trimmed(self) -> None:
        text = "pick { red | green | blue }"
        resolved = resolve_prompt(text, 0, base_seed=0)
        self.assertEqual(resolved.prompt, "pick red")

    def test_no_placeholders_returns_plain_prompt(self) -> None:
        resolved = resolve_prompt("sunset over water", 42)
        self.assertEqual(resolved, ResolvedPrompt(template=None, prompt="sunset over water"))

    def test_literal_braces_without_pipes_are_kept(self) -> None:
        text = "use {json} with {yes|no}"
        resolved = resolve_prompt(text, 0, base_seed=0)
        self.assertEqual(resolved.prompt, "use {json} with yes")

        resolved = resolve_prompt(text, 1, base_seed=0)
        self.assertEqual(resolved.prompt, "use {json} with no")

    def test_three_groups(self) -> None:
        text = "{a|b}-{c|d}-{e|f}"
        seen = {resolve_prompt(text, seed, base_seed=0).prompt for seed in range(8)}
        self.assertEqual(len(seen), 8)
        self.assertIn("a-c-e", seen)
        self.assertIn("b-d-f", seen)

if __name__ == "__main__":
    unittest.main()
