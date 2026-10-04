// Package consistency holds the tests that check this Zarr implementation
// against zarr-python. It has no code of its own; see README.md.
//
// The tests drive zarr-python through oracle.py and apply the same
// operations with both implementations, then require that:
//
//   - the metadata documents are byte-identical;
//   - the stores hold the same keys, and chunks are byte-identical
//     wherever the codecs are deterministic (otherwise each decodes the
//     other's chunks to the same values);
//   - Go and Python read the same values from both stores, for whole
//     arrays and for random selections of every kind.
//
// Without a Python interpreter (ZARR_PYTHON unset) only the tests against
// the committed fixtures in testdata run.
package consistency
