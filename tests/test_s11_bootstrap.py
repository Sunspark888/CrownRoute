import unittest

import numpy as np

from student.bootstrap_sources import paired_intervals


def example_rows():
    rows = []
    for index, method in enumerate(("first", "second")):
        for run, seed in enumerate((42, 43, 44)):
            for source in range(9):
                for patch in range(1 + source % 3):
                    tp = (source + run + patch + index * (1 + source % 2)) % 3
                    fp = (source // 2 + patch + run * index) % 3
                    fn = 2 - tp
                    rows.append({
                        "method": method, "seed": seed, "threshold": 0.5,
                        "image_id": f"{source}-{patch}", "source": str(source),
                        "tp": tp, "fp": fp, "fn": fn, "count_abs_error": abs(fp - fn),
                    })
    return rows


class PairedBootstrapTests(unittest.TestCase):
    def test_cluster_draws_and_seed_means(self):
        comparisons = [("first", "second"), ("second", "first")]
        rows = example_rows()
        result = paired_intervals(reversed(rows), comparisons)
        # Independent patch-by-patch enumeration of the 2,000 fixed source draws.
        expected = [[-0.25490196078431393, 0.18333333333333324],
                    [0.6055126308290895, 6.848908802284016]]
        endpoints = [[row["ci_low"], row["ci_high"]] for row in result]
        np.testing.assert_allclose(endpoints[:2], expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(endpoints[2:], -np.fliplr(expected), atol=1e-12)
        self.assertTrue(all(row["source_clusters"] == 9 for row in result))
        self.assertTrue(all(row["bootstrap_draws"] == 2000 for row in result))

    def test_unpaired_or_duplicate_patches_are_rejected(self):
        rows = example_rows()
        comparison = [("first", "second")]
        with self.assertRaisesRegex(ValueError, "must match"):
            paired_intervals(rows[:-1], comparison)
        with self.assertRaisesRegex(ValueError, "must match"):
            paired_intervals([*rows[:-1], {**rows[-1], "source": "different"}], comparison)
        with self.assertRaisesRegex(ValueError, "Duplicate patch"):
            paired_intervals([*rows, rows[0]], comparison)


if __name__ == "__main__":
    unittest.main()
