# Third-party notices

## 5D Chess JS 1.2.1

- Authors: Shaun Wu; contributors include Shad Amethyst, SnowmanFactory, and Neathp.
- Source: https://gitlab.com/5d-chess/5d-chess-js
- License: AGPL-3.0-or-later. A copy is in `LICENSE` and in the installed package.
- Used for piece geometry, board transitions, notation, and built-in starting positions. The package is unmodified; local wrappers replace its eager complete-action search and timeout-based terminal detection.

Its installed dependency tree also includes `blueimp-md5` (MIT), `module-alias` (MIT), and `present` (MIT). Their license files are distributed with their npm packages. `package-lock.json` pins the resolved dependency tree.

No game artwork or proprietary Steam game files are included. The interface renders pieces with system Unicode glyphs.

## LCZero model transfer

The optional LCZero importer and PyTorch backbone are independently written using
the public format and inference equations from the LCZero project:

- Source: https://github.com/LeelaChessZero/lc0
- Reference files: `proto/net.proto`, `src/neural/loader.cc`, and the BLAS backend.
- Upstream code license: GPL-3.0-or-later; no upstream source is vendored here.
- The user-supplied model is read locally and is not included in this repository.
  Its protobuf license field is empty; this project makes no claim about its
  redistribution license. Generated checkpoints remain local git-ignored artifacts.

The importer records the source SHA256, tensor mapping, and omitted orthodox
heads in each trained checkpoint. See `docs/lc0-transfer.md` for architecture
and numerical-parity limits.
