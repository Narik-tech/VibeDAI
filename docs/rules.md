# Rules, sources, and compatibility

Vibe-D AI uses the piece geometry, board-history representation, and notation parser from **5d-chess-js 1.2.1**, with its own legal-action generator and search. This is an independent local analysis tool, not the official game's AI. Playing strength has not been established by an external rating or tournament comparison.

## What a search turn means

A move moves one piece. An **action** is the ordered sequence of moves a player submits together. Search depth counts complete submitted actions, rather than alternating players after every individual piece move.

Pieces move in file, rank, time, and timeline coordinates. Temporal movement steps between boards of the same player color, so one unit in the time dimension is two internal half-turn indices. Rooks use one axis, bishops exactly two, unicorns exactly three, and dragons all four. A queen uses any nonempty combination of the four axes, with equal distance along each changed axis. A king uses the same 80 directions for a single step. A knight changes one axis by two and a different axis by one.

Pawns move forward along rank or toward the opponent's side of the timeline axis. Their ordinary rank/file captures and their timeline/time captures are separate extensions: ordinary pawns do not acquire every possible combination of a forward and sideways axis. Unmoved pawns may also double-step across timelines when the intermediate square is clear. Promotion, en passant, and castling use the underlying library's rules.

Historical boards are immutable. A temporal move advances the source board and either advances a latest destination board or creates a new timeline from a historical destination. Each side's first extra timeline is active; additional timelines become active according to the opponent's number of created timelines. Inactive boards remain legal move sources and destinations, but do not hold the present back.

Submission requires that the present has passed to the opponent and that no opponent move can capture one of the submitting player's royal pieces. Search permits temporary self-check between component moves and continues considering optional moves after a legal submission becomes possible. Different orders reaching exactly the same historical state are deduplicated. A king or royal queen may be threatened but cannot actually be captured in a played action.

Checkmate or stalemate is established only by exhausting the legal action tree. A stopped search is not evidence of either. The check used to distinguish these terminal states is the library's forced-pass check; it is separate from checking actual opponent captures at submission.

Action generation prunes a partial turn when an actual opponent move can already capture a royal piece. This attack cannot be repaired by the remaining component moves: existing opponent-color boards, including the attack's source, target, and ray, remain immutable throughout the turn. Forced-pass (phantom) checks are treated differently and are still allowed between component moves. The source geometry can also be cached for a whole action because moves consume mover-color latest boards and add only opponent-color boards. Tests compare these optimizations against exhaustive generation, including arrivals that turn a formerly latest destination into history.

Royal safety tests attacks directly against every recorded royal square, using the pinned library's movement vectors and pawn/brawn capture rules. It includes historical and inactive boards, blockers, even timelines, both royal types, and promotion availability. Piece and royal locations are cached only within one immutable action traversal; public `canSubmit` and `inCheck` calls observe edits to caller-owned boards.

When a spatial component fails royal safety, the generator also checks that component against the original turn position. If it is unsafe there too, the same spatial outcome is excluded from later combinations. An attack that needs another component does not justify excluding either move alone. Temporal moves retain their full ordering because an earlier component can change an arrival from a merge to a branch. `cacheUnsafeMoves: false` disables the learned spatial exclusions; `pruneUnsafe: false` disables both kinds of safety pruning for exhaustive comparisons. Budget interruptions continue to propagate without a terminal claim.

During nonchecked capture quiescence, the engine separately proves that at least one legal turn exists, then generates only turns containing a capture or promotion. It skips a quiet partial path when no remaining source has a capture or promotion: the remaining mover-color snapshots cannot change to create one. This filtering is not applied to normal-depth search or check evasions.

Quiescence search is a finite tactical extension, not a proof that a position is tactically quiet. At its emergency check-sequence limit it evaluates a legal evasion instead of continuing indefinitely. The resulting position is still checked for a legal-action witness: a mating or stalemating evasion receives its terminal score. Longer tactics may be missed. Heuristic evaluation is kept outside the search's reserved mate-score range.

Preferred turns are replayed in their exact component order before general enumeration. Replay checks move geometry and submission legality, and duplicate suppression removes only the repeated result, preserving optional continuations. Quiet-move ordering history omits absolute half-turn numbers so successful moves can guide later plies. Tactical transposition entries use separate keys for each remaining horizon, normalize mate distances, and classify bounds against the actual searched window. They share the normal table's entry limit and never substitute for a full-turn search result.

## Primary sources consulted

- [The developer's official game description](https://www.5dchesswithmultiversetimetravel.com/presskit.html) describes historical branching, travel between timelines, and protecting kings in the past as well as the present.
- [5D Chess JS upstream project](https://gitlab.com/5d-chess/5d-chess-js) and [API documentation source](https://gitlab.com/5d-chess/5d-chess-js/-/blob/master/docs/README.md) define the board and move APIs, supported variants, and notation compatibility.
- [Piece implementation](https://gitlab.com/5d-chess/5d-chess-js/-/blob/master/src/piece.js) and [board implementation](https://gitlab.com/5d-chess/5d-chess-js/-/blob/master/src/board.js) are the concrete primary sources for movement vectors, pawn extensions, branching, active timelines, and the present used by this implementation.
- [The original 5DPGN/5DFEN specification revision used by the library](https://github.com/adri326/5dchess-notation/tree/92c68f0c19183b4ff4459b4a32107b0bf6e069a2) is the import/export compatibility target. This is not a promise of compatibility with every later notation revision.
- [5dchess-tools](https://github.com/adri326/5dchess-tools) describes an independent Rust analysis engine and its use of lazy action generation. Its own documentation cautions that complete checkmate proofs can be extremely expensive.
- [cwmtt's FastCheckmate implementation](https://github.com/penteract/cwmtt/blob/master/Game/Chess/TimeTravel/FastCheckmate.lhs) inspired reusing unsafe board outcomes across move combinations. This engine retains its lazy complete-turn traversal and learns only the independently verified spatial exclusions described above.

## Upstream behavior deliberately avoided

The released 1.2.1 bundle and upstream source were inspected directly. Vibe-D AI does not call the library's eager `actions()` generator or high-level `inCheckmate` / `inStalemate` getters:

- In [action.js](https://gitlab.com/5d-chess/5d-chess-js/-/blob/master/src/action.js), the promotion argument is passed in the `spatialOnly` parameter slot. Its defaults also omit optional inactive or future-board moves. It materializes the full action tree before returning results.
- In [mate.js](https://gitlab.com/5d-chess/5d-chess-js/-/blob/master/src/mate.js), a timeout returns `[true, true]`; the high-level getters in [index.js](https://gitlab.com/5d-chess/5d-chess-js/-/blob/master/src/index.js) expose the first component as a positive mate/stalemate result. The slow checkmate traversal also passes a node wrapper where a board array is expected.
- High-level imports reset to standard when no Board header is supplied and can discard malformed token suffixes. Vibe-D AI parses complete action and move tokens, matches moves against generated legal geometry, and rejects incomplete final turns. It preserves the selected variant when a Board header is absent, or infers Custom when FEN boards are present.

These are targeted replacements, not a proof that every upstream variant or unusual position matches the commercial game. The tests cover temporal branching, all four movement axes, full-turn check resolution, optional/inactive-board moves, exact-state deduplication, castling, promotion, en passant, and a known temporal checkmate.

## Import boundaries

Import accepts fully submitted 5DPGN games and custom 5DFEN positions with a board and at least one royal piece of each color. Custom positions are analysis setups; the importer does not prove they are reachable from an official starting position. Empty PGN input at the HTTP import endpoint is rejected. `createPosition()` without a game creates the selected built-in initial position; Custom requires FEN setup boards.

The mover is inferred from the earliest active frontier, including positions without timeline zero. Imported action numbers and player separators must agree with that mover. Malformed headers, duplicate Board/Size/Mode/Promotions headers, inconsistent FEN/variant combinations, and unconsumed move text are rejected before the live session changes. Brace and semicolon comments, result markers, ordinary move annotations, and exported temporal coordinate/branch tokens are supported. An upstream duplicated-file pawn capture such as `eexf3` is accepted only when it exactly matches a generated legal move's export. Session state and the notation exporter keep independent board containers so play and undo preserve historical positions.

To prevent malformed input from causing enormous sparse allocations inside the upstream parser, import rejects text over 500,000 characters, boards over 16×16, timeline coordinates outside −64…64, and turn coordinates outside 0…2048. These are explicit input errors; search does not silently discard actions beyond a timeline count or move-list limit. The upstream parser's treatment of special turn-zero and even-timeline custom setups remains a compatibility limitation; verify unusual custom numbering against the exported notation.

The dependency declares **AGPL-3.0-or-later** in its [package metadata](https://gitlab.com/5d-chess/5d-chess-js/-/blob/master/package.json). This project uses the same license and keeps the upstream dependency and attribution visible.
