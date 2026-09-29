const stepKey = (l, t, r, f) => ((l + 2) * 5 + t + 2) * 25 + (r + 2) * 5 + f + 2;
const rayKey = (l, t, r, f) => ((l + 1) * 3 + t + 1) * 9 + (r + 1) * 3 + f + 1;
const royal = piece => piece === 11 || piece === 12 || piece === 19 || piece === 20;
const lineCoordinate = (index, even) => index % 2 ? -(index + 1) / 2 : index / 2 - (even ? 1 : 0);
const movementCache = new WeakMap();

/** Exact royal-capture detection using the pinned rules library's geometry. */
export function createRoyalSafety(raw, { cacheBoards = false } = {}) {
  let movement = movementCache.get(raw);
  if (!movement) movementCache.set(raw, movement = Array.from({ length: 13 }, (_, type) => {
    const steps = new Uint8Array(625), rays = new Uint8Array(81);
    for (const v of raw.pieceFuncs.movePos(type * 2)) steps[stepKey(...v)] = 1;
    for (const v of raw.pieceFuncs.moveVecs(type * 2)) rays[rayKey(...v)] = 1;
    return { steps, rays };
  }));
  // Opt in only for an immutable traversal. Public positions can be edited,
  // so callers of canSubmit/inCheck must not retain these cached snapshots.
  const boardCache = cacheBoards ? new WeakMap() : null;

  function scanBoard(squares) {
    let entry = boardCache?.get(squares);
    if (entry) return entry;
    entry = { royals: [], pieces: [] };
    for (let r = 0; r < squares.length; r++) for (let f = 0; squares[r] && f < squares[r].length; f++) {
      const piece = Math.abs(squares[r][f]);
      if (piece) entry.pieces.push(r, f, piece);
      if (royal(piece)) entry.royals.push(r, f, piece % 2);
    }
    boardCache?.set(squares, entry);
    return entry;
  }

  function appendTargets(targets, squares, l, line, t, color) {
    if (boardCache) {
      const entries = scanBoard(squares).royals;
      for (let i = 0; i < entries.length; i += 3) {
        if (entries[i + 2] !== color) targets.push(l, line, t, entries[i], entries[i + 1]);
      }
    } else {
      for (let r = 0; r < squares.length; r++) for (let f = 0; squares[r] && f < squares[r].length; f++) {
        const piece = Math.abs(squares[r][f]);
        if (royal(piece) && piece % 2 !== color) targets.push(l, line, t, r, f);
      }
    }
  }

  function find(position, describe) {
    const { board } = position, color = (position.action + 1) % 2;
    // A complete single-timeline turn has no latest board for the opponent
    // until its first component is applied. Avoid scanning the entire history
    // in that common case; historical boards can only be capture targets.
    let firstSource = 0;
    for (; firstSource < board.length; firstSource++) {
      const timeline = board[firstSource], turn = timeline?.length - 1;
      if (turn % 2 === color && timeline[turn]) break;
    }
    if (firstSource === board.length) return describe ? null : false;
    const even = raw.boardFuncs.isEvenTimeline(board), targets = [];
    // Captures can target every recorded board of the source's half-turn
    // color, including historical royals on inactive timelines.
    for (let l = 0; l < board.length; l++) {
      const timeline = board[l];
      if (!timeline) continue;
      const line = lineCoordinate(l, even);
      for (let t = color; t < timeline.length; t += 2) {
        const squares = timeline[t];
        if (squares) appendTargets(targets, squares, l, line, t, color);
      }
    }
    if (!targets.length) return describe ? null : false;
    let promotion;
    function promotionPiece() {
      if (promotion !== undefined) return promotion;
      const choices = position.promotions?.length ? position.promotions : raw.pieceFuncs.availablePromotionPieces(board);
      return promotion = choices.find(piece => piece % 2 === color) ?? null;
    }
    for (let l = firstSource; l < board.length; l++) {
      const timeline = board[l];
      if (!timeline || (timeline.length - 1) % 2 !== color) continue;
      const t = timeline.length - 1, squares = timeline[t], line = lineCoordinate(l, even);
      if (!squares) continue;
      const sources = scanBoard(squares).pieces;
      for (let source = 0; source < sources.length; source += 3) {
        const r = sources[source], f = sources[source + 1], piece = sources[source + 2];
        if (piece % 2 !== color) continue;
        const type = Math.ceil(piece / 2), profile = movement[type];
        for (let i = 0; i < targets.length; i += 5) {
          const targetLine = targets[i], targetTime = targets[i + 2];
          const targetRank = targets[i + 3], targetFile = targets[i + 4];
          const dl = targets[i + 1] - line, dt = (targetTime - t) / 2;
          const dr = targetRank - r, df = targetFile - f;
          let promotes = false;
          if (type === 1 || type === 8) {
            const forward = color === 0 ? 1 : -1;
            const spatial = dl === 0 && dt === 0 && dr === forward && Math.abs(df) === 1;
            const temporal = dl === -forward && Math.abs(dt) === 1 && dr === 0 && df === 0;
            const brawn = type === 8 && (
              (dl === -forward && dt === 0 && ((dr === 0 && Math.abs(df) === 1) || (dr === forward && df === 0))) ||
              (dl === 0 && dt === -1 && dr === forward && df === 0)
            );
            if (!spatial && !temporal && !brawn) continue;
            // Upstream's ordinary timeline/time pawn capture never promotes.
            // Spatial and additional brawn captures use the source's height.
            promotes = !temporal && targetRank === (color === 0 ? squares.length - 1 : 0);
            if (promotes && promotionPiece() === null) continue;
          } else {
            const al = Math.abs(dl), at = Math.abs(dt), ar = Math.abs(dr), af = Math.abs(df);
            const distance = Math.max(al, at, ar, af);
            if (!(distance <= 2 && profile.steps[stepKey(dl, dt, dr, df)])) {
              if (!distance || (al && al !== distance) || (at && at !== distance) ||
                  (ar && ar !== distance) || (af && af !== distance)) continue;
              const sl = Math.sign(dl), st = Math.sign(dt), sr = Math.sign(dr), sf = Math.sign(df);
              if (!profile.rays[rayKey(sl, st, sr, sf)]) continue;
              let blocked = false;
              for (let offset = 1; offset < distance; offset++) {
                const intermediateLine = raw.pieceFuncs.timelineMove(l, sl * offset, even);
                if (board[intermediateLine]?.[t + st * offset * 2]?.[r + sr * offset]?.[f + sf * offset] !== 0) {
                  blocked = true;
                  break;
                }
              }
              if (blocked) continue;
            }
          }
          if (!describe) return true;
          const destination = [targetLine, targetTime, targetRank, targetFile];
          if (promotes) destination.push(promotionPiece());
          return [[l, t, r, f], destination];
        }
      }
    }
    return describe ? null : false;
  }

  return {
    attackedByNextPlayer: position => find(position, false),
    findRoyalAttack: position => find(position, true),
    createCached: () => createRoyalSafety(raw, { cacheBoards: true }),
  };
}
