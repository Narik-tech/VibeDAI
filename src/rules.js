import Chess from '5d-chess-js';
import { createRoyalSafety } from './royal-safety.js';

// Use the upstream geometry, history representation and notation. Its eager
// action generator and timeout-as-mate API are deliberately not used here.
export const raw = new Chess().raw;
const royalSafety = createRoyalSafety(raw);
const { attackedByNextPlayer } = royalSafety;
const royal = piece => [11, 12, 19, 20].includes(Math.abs(piece));
const pieceAt = (board, square) => board[square[0]]?.[square[1]]?.[square[2]]?.[square[3]];
const equalMove = (a, b) => raw.validateFuncs.compareMove(a, b) === 0;

const PIECE_TOKEN = '(?:RQ|CK|[PBNRQKSWCYUD])';
const SQUARE_TOKEN = '[a-p](?:1[0-6]|[1-9])';
const BOARD_TOKEN = '\\(L?[+-]?\\d+T[+]?\\d+\\)';
const MOVE_SUFFIX = '[+#~*!?=]*';
const SPATIAL_MOVE = new RegExp(`^(?<board>${BOARD_TOKEN})?(?<piece>${PIECE_TOKEN})?(?<source>[a-p]?(?:1[0-6]|[1-9])?)x?(?<end>${SQUARE_TOKEN})(?:=(?<promotion>${PIECE_TOKEN}))?${MOVE_SUFFIX}$`);
const TEMPORAL_MOVE = new RegExp(`^(?<board>${BOARD_TOKEN})(?<piece>${PIECE_TOKEN})?(?<source>${SQUARE_TOKEN})(?<jump>>{1,2})x?(?<destination>${BOARD_TOKEN})(?<end>${SQUARE_TOKEN})(?:=(?<promotion>${PIECE_TOKEN}))?${MOVE_SUFFIX}$`);
const CASTLING_MOVE = new RegExp(`^(?<board>${BOARD_TOKEN})?(?<castle>O-O(?:-O)?)${MOVE_SUFFIX}$`);
const CRITICAL_HEADERS = new Set(['board', 'size', 'mode', 'promotions']);

function stripComments(text) {
  let clean = '', bracket = false, brace = false, line = false;
  for (const char of text) {
    if (line) { if (char === '\n' || char === '\r') { line = false; clean += ' '; } continue; }
    if (brace) {
      if (char === '{') throw new Error('Malformed PGN comment.');
      if (char === '}') { brace = false; clean += ' '; }
      continue;
    }
    if (!bracket && char === ';') { line = true; continue; }
    if (!bracket && char === '{') { brace = true; continue; }
    if (!bracket && char === '}') throw new Error('Malformed PGN comment.');
    if (char === '[') bracket = true;
    if (char === ']') bracket = false;
    clean += char;
  }
  if (brace) throw new Error('Malformed PGN comment.');
  return clean;
}

function readDocument(text) {
  const headers = [], fens = [], seen = new Set();
  const body = stripComments(text).replace(/\[[^\[\]]*\]/g, block => {
    const header = /^\[([A-Za-z][A-Za-z0-9_.-]*)\s+"([^"\r\n]*)"\]$/.exec(block);
    if (header) {
      const key = header[1].toLowerCase();
      if (CRITICAL_HEADERS.has(key) && seen.has(key)) throw new Error(`Duplicate ${header[1]} header.`);
      seen.add(key);
      headers.push({ key, value: header[2], text: block });
    } else if (/^\[[^\]\r\n"]+:[^:\]]+:[^:\]]+:[^:\]]+\]$/.test(block)) fens.push(block);
    else throw new Error('Malformed PGN header or FEN board.');
    return ' ';
  }).trim();
  if (/[\[\]]/.test(body)) throw new Error('Malformed PGN header or FEN board.');
  return { headers, fens, body };
}

/** Guard the upstream parser's sparse-array allocations before parsing input. */
export function normalizePGN(pgn, variant = 'standard') {
  if (typeof pgn !== 'string') throw new Error('PGN must be a string.');
  if (pgn.length > 500_000) throw new Error('PGN import is limited to 500,000 characters.');
  const variants = raw.metadataFuncs.variantDict;
  const variantName = value => variants.find(([name, key]) => name.toLowerCase() === value.toLowerCase() || key === value.toLowerCase());
  if (typeof variant !== 'string' || !variantName(variant)) throw new Error('Unknown starting variant.');
  const document = readDocument(pgn);
  const header = name => document.headers.find(item => item.key === name);
  if (header('mode') && header('mode').value.toLowerCase() !== '5d') throw new Error('Only 5D mode is supported.');
  const boardHeader = header('board');
  if (boardHeader && !variantName(boardHeader.value)) throw new Error('Unknown Board header.');
  if (document.fens.length && boardHeader && variantName(boardHeader.value)[1] !== 'custom') throw new Error('FEN boards require the Custom variant.');
  if (header('promotions') && !header('promotions').value.split(',').every(value => new RegExp(`^${PIECE_TOKEN}$`).test(value))) throw new Error('Invalid Promotions header.');
  for (const match of pgn.matchAll(/\[size\s+"([^"]*)"\]/gi)) {
    const size = /^(\d+)x(\d+)$/.exec(match[1]);
    if (!size || size.slice(1).some(value => +value < 1 || +value > 16)) throw new Error('Board dimensions must be between 1 and 16.');
  }
  const checkCoordinate = (line, turn) => {
    if (!/^[+-]?\d+$/.test(line) || !/^\d+$/.test(turn) || !Number.isSafeInteger(+line) || Math.abs(+line) > 64 || !Number.isSafeInteger(+turn) || +turn < 0 || +turn > 2048) {
      throw new Error('Import coordinates exceed the supported limits: timeline ±64, turn 0–2048.');
    }
  };
  for (const match of pgn.matchAll(/\(L?([+-]?\d+)T[+]?(\d+)\)/g)) checkCoordinate(match[1], match[2]);
  const fenCoordinates = new Set();
  for (const match of pgn.matchAll(/\[([^\]\r\n"]*):([^:\]]*):([^:\]]*):([^:\]]*)\]/g)) {
    checkCoordinate(match[2], match[3]);
    if (!['w', 'b'].includes(match[4])) throw new Error('Invalid FEN board player.');
    const coordinate = `${match[2]}:${match[3]}:${match[4]}`;
    if (fenCoordinates.has(coordinate)) throw new Error('Duplicate FEN board coordinate.');
    fenCoordinates.add(coordinate);
    for (const run of match[1].matchAll(/\d+/g)) {
      if (+run[0] < 1 || +run[0] > 16) throw new Error('FEN empty-square runs must be between 1 and 16.');
    }
  }
  if (pgn.trim() && !document.headers.length && !document.fens.length && !document.body) throw new Error('PGN contains no game or position.');
  const prefix = boardHeader ? [] : [`[Board "${document.fens.length ? 'Custom' : variantName(variant)[0]}"]`];
  return [...prefix, ...document.headers.map(item => item.text), ...document.fens, document.body].filter(Boolean).join('\n');
}

function assertUsableBoard(board) {
  let hasBoard = false;
  const kings = [false, false];
  for (const timeline of board) {
    if (!timeline) continue;
    for (const turn of timeline) {
      if (!turn) continue;
      hasBoard = true;
      if (!Array.isArray(turn) || turn.length < 1 || turn.length > 16 || !Array.isArray(turn[0]) || turn[0].length < 1 || turn[0].length > 16) {
        throw new Error('Invalid board dimensions.');
      }
      for (const rank of turn) {
        if (!Array.isArray(rank) || rank.length !== turn[0].length || rank.some(piece => !Number.isInteger(piece) || Math.abs(piece) > 24)) throw new Error('Malformed board pieces or ranks.');
        for (const piece of rank) if (royal(piece)) kings[Math.abs(piece) % 2] = true;
      }
    }
  }
  if (!hasBoard || !kings.every(Boolean)) throw new Error('A position must contain a board and at least one royal piece of each color.');
}

export function createValidatedGame({ variant = 'standard', pgn } = {}) {
  const normalized = normalizePGN(pgn ?? '', variant);
  const document = readDocument(normalized);
  const metadata = raw.metadataFuncs.strToObj(normalized);
  const game = new Chess(undefined, metadata.board);
  game.skipDetection = true;
  Object.assign(game.metadata, metadata);
  if (metadata.board === 'custom') {
    if (!document.fens.length) throw new Error('Custom positions require FEN boards with royal pieces of both colors.');
    game.fen(normalized);
  }
  assertUsableBoard(game.rawBoard);
  const frontier = raw.boardFuncs.active(game.rawBoard).map(line => game.rawBoard[line].length - 1);
  if (!frontier.length) throw new Error('The position has no active frontier.');
  game.rawAction = Math.min(...frontier) % 2;
  game.rawStartingAction = game.rawAction;
  game.rawBoardHistory = [raw.boardFuncs.copy(game.rawBoard)];
  game.rawPromotionPieces = metadata.promotions
    ? metadata.promotions.split(',').flatMap(piece => [raw.pieceFuncs.fromChar(piece, 0), raw.pieceFuncs.fromChar(piece, 1)])
    : raw.pieceFuncs.availablePromotionPieces(game.rawBoard);
  let position = { board: game.rawBoard, action: game.rawAction, promotions: game.rawPromotionPieces.slice() };
  let moves = [], hasMarker = false, finished = false;
  const commit = () => {
    if (!moves.length) throw new Error('PGN action is empty.');
    if (!canSubmit(position)) throw new Error('PGN ends with an incomplete or illegal turn. Import fully submitted turns.');
    position = submitPosition(position);
    game.rawBoard = position.board;
    game.rawBoardHistory.push(raw.boardFuncs.copy(position.board));
    game.rawActionHistory.push(moves);
    game.rawAction = position.action;
    moves = [];
  };
  let rest = document.body;
  while (rest.trim()) {
    rest = rest.trimStart();
    if (finished) throw new Error('Unexpected text after the game result.');
    const result = /^(1\/2-1\/2|1-0|0-1|\*)(?=\s|$)/.exec(rest);
    if (result) {
      if (moves.length) commit();
      finished = true;
      rest = rest.slice(result[0].length);
      continue;
    }
    const marker = /^(\d+)(\.\.\.|\.)/.exec(rest);
    if (marker || rest[0] === '/') {
      if (moves.length) { commit(); hasMarker = false; }
      if (hasMarker) throw new Error('PGN action is empty.');
      const side = marker ? (marker[2] === '...' ? 1 : 0) : 1;
      if (side !== position.action % 2 || (marker && +marker[1] !== Math.floor(position.action / 2) + 1)) throw new Error('PGN action number or player does not match the position.');
      hasMarker = true;
      rest = rest.slice(marker ? marker[0].length : 1);
      continue;
    }
    const annotation = /^\((?:>L|~T)[+-]?\d+\)/.exec(rest);
    if (annotation) {
      if (!moves.length) throw new Error('Annotation must follow a move.');
      rest = rest.slice(annotation[0].length);
      continue;
    }
    if (!hasMarker) throw new Error('Expected a numbered PGN action or a Black action separator.');
    const token = /^[^\s/]+/.exec(rest)?.[0];
    if (!token) throw new Error('Malformed PGN move token.');
    const move = parseMove(position, token);
    position = applyMove(position, move);
    moves.push(move);
    rest = rest.slice(token.length);
  }
  if (moves.length) commit();
  else if (hasMarker && !finished) throw new Error('PGN action is empty.');
  game.rawBoard = position.board;
  game.rawMoveBuffer = [];
  return { chess: game, position: { ...position, board: raw.boardFuncs.copy(position.board) } };
}

export function createPosition(options = {}) {
  return createValidatedGame(options).position;
}

/** Pseudo-legal individual moves. Royal captures are threats, never played. */
export function pseudoMoves(position) {
  return raw.boardFuncs.moves(position.board, position.action, false, false, false, position.promotions)
    .filter(move => !royal(pieceAt(position.board, move[1])));
}

// Keep the pinned library's direction order: tied search moves must retain
// their order. Pawns, brawns and unmoved kings use its special-move handling.
const SEARCH_MOVEMENT = Array.from({ length: 13 }, (_, type) => ({
  steps: raw.pieceFuncs.movePos(type * 2), rays: raw.pieceFuncs.moveVecs(type * 2),
}));
const SEARCH_SINGLE_TIMELINE_MOVEMENT = SEARCH_MOVEMENT.map(({ steps, rays }) => ({
  steps: steps.filter(vector => vector[0] === 0), rays: rays.filter(vector => vector[0] === 0),
}));
const SEARCH_SPATIAL_MOVEMENT = SEARCH_SINGLE_TIMELINE_MOVEMENT.map(({ steps, rays }) => ({
  steps: steps.filter(vector => vector[1] === 0), rays: rays.filter(vector => vector[1] === 0),
}));

function searchPseudoMoves(position) {
  const { board } = position, color = position.action % 2;
  const even = raw.boardFuncs.isEvenTimeline(board), moves = [];
  for (let l = 0; l < board.length; l++) {
    const timeline = board[l], t = timeline?.length - 1;
    if (t % 2 !== color) continue;
    const squares = timeline[t];
    // A lone timeline has no destination for timeline-changing directions.
    // Before its first same-color historical board, only spatial directions
    // can land anywhere. Filtering the fixed tables retains move tie order.
    const movement = board.length === 1
      ? (t < 2 ? SEARCH_SPATIAL_MOVEMENT : SEARCH_SINGLE_TIMELINE_MOVEMENT) : SEARCH_MOVEMENT;
    for (let r = 0; squares && r < squares.length; r++) {
      for (let f = 0; squares[r] && f < squares[r].length; f++) {
        const piece = squares[r][f], absolute = Math.abs(piece);
        if (!absolute || absolute % 2 !== color) continue;
        const from = [l, t, r, f], type = Math.ceil(absolute / 2);
        if (type === 1 || type === 8 || piece === -11 || piece === -12) {
          // Ordinary pawns need the adjacent timeline for every temporal
          // move, including the first step of a double push. Brawns can also
          // capture into their own past, so keep their full geometry.
          const spatialOnly = type === 1 && !board[raw.pieceFuncs.timelineMove(l, color ? 1 : -1, even)];
          for (const move of raw.pieceFuncs.moves(board, from, spatialOnly, position.promotions)) {
            if (!royal(pieceAt(board, move[1]))) moves.push(move);
          }
          continue;
        }
        const { steps, rays } = movement[type];
        for (let index = 0; index < steps.length; index++) {
          const vector = steps[index], dl = vector[0], dt = vector[1], dr = vector[2], df = vector[3];
          const line = raw.pieceFuncs.timelineMove(l, dl, even), turn = t + dt * 2;
          const rank = r + dr, file = f + df, row = board[line]?.[turn]?.[rank];
          if (!row || file < 0 || file >= row.length) continue;
          const target = row[file];
          if ((target === 0 || Math.abs(target) % 2 !== color) && !royal(target)) {
            moves.push([from, [line, turn, rank, file]]);
          }
        }
        for (let index = 0; index < rays.length; index++) {
          const vector = rays[index], dl = vector[0], dt = vector[1], dr = vector[2], df = vector[3];
          let line = raw.pieceFuncs.timelineMove(l, dl, even), turn = t + dt * 2;
          let rank = r + dr, file = f + df;
          for (;;) {
            const row = board[line]?.[turn]?.[rank];
            if (!row || file < 0 || file >= row.length) break;
            const target = row[file];
            if (target !== 0) {
              if (Math.abs(target) % 2 !== color && !royal(target)) moves.push([from, [line, turn, rank, file]]);
              break;
            }
            moves.push([from, [line, turn, rank, file]]);
            line = raw.pieceFuncs.timelineMove(line, dl, even);
            turn += dt * 2; rank += dr; file += df;
          }
        }
      }
    }
  }
  return moves;
}

/** Reuse geometry while search histories and promotion lists stay immutable. */
export function createSearchMoveGenerator() {
  const histories = new WeakMap();
  return position => {
    let entries = histories.get(position.board);
    if (!entries) histories.set(position.board, entries = []);
    const color = position.action % 2, previous = entries[color];
    if (previous && previous.promotions === position.promotions) return previous.moves;
    // Probe geometry with scalar coordinates, allocating arrays only for
    // actual destinations instead of every off-board direction and ray step.
    const moves = searchPseudoMoves(position);
    entries[color] = { promotions: position.promotions, moves };
    return moves;
  };
}

export function isTacticalMove(position, move) {
  return Boolean(pieceAt(position.board, move[1])) || move.length === 3 || move[1].length > 4;
}

/** Apply a generated move. Call parseMove first when accepting untrusted input. */
export function applyMove(position, move) {
  // Upstream copies changed single boards and creates branch containers. Only
  // the source and an arrival onto another latest board append to an existing
  // timeline; keep every unaffected history container shared among siblings.
  const board = position.board.slice();
  const source = move[0][0], destination = move[1][0];
  board[source] = board[source]?.slice() ?? board[source];
  if (destination !== source && board[destination]?.length - 1 === move[1][1]) {
    board[destination] = board[destination].slice();
  }
  const [from, to] = move;
  if (move.length === 2 && source === destination && from[1] === to[1]
      && raw.boardFuncs.positionIsLatest(board, from) && pieceAt(board, from)) {
    // An ordinary spatial move edits at most two ranks. Keep untouched ranks
    // shared just like historical boards, while preserving the upstream piece
    // flags, promotion handling and sparse-container normalization.
    const previous = board[source][from[1]], next = previous.slice();
    next[from[2]] = previous[from[2]].slice();
    if (to[2] !== from[2]) next[to[2]] = previous[to[2]].slice();
    next[from[2]][from[3]] = 0;
    next[to[2]][to[3]] = to[4] || Math.abs(previous[from[2]][from[3]]);
    raw.boardFuncs.setTurn(board, source, from[1] + 1, next);
  } else raw.boardFuncs.move(board, move);
  return { ...position, board };
}

/** The upstream present rule, scanning from the frontier instead of turn zero. */
export function presentTimelines(position) {
  const active = raw.boardFuncs.active(position.board);
  const latest = [], color = position.action % 2;
  let lowest = -1;
  for (const line of active) {
    const timeline = position.board[line];
    let end = timeline.length - 1;
    while (end >= 0 && !timeline[end]) end--;
    // Upstream treats an empty active timeline's latest turn as zero.
    latest.push(Math.max(0, end));
    let turn = end;
    if (turn % 2 !== color) turn--;
    while (turn >= 0 && !timeline[turn]) turn -= 2;
    if (turn >= 0 && (lowest < 0 || turn < lowest)) lowest = turn;
  }
  const present = [];
  if (lowest < 0) return present;
  for (let index = 0; index < active.length; index++) {
    if (latest[index] < lowest) return [];
    if (latest[index] === lowest) present.push(active[index]);
  }
  return present;
}

export function canSubmit(position) {
  return presentTimelines(position).length === 0 && !attackedByNextPlayer(position);
}

export function submitPosition(position) {
  if (!canSubmit(position)) throw new Error('Turn cannot be submitted: advance the present and protect every royal piece.');
  return { ...position, action: position.action + 1 };
}

/** Forced-pass check, for classifying an exhausted turn tree as mate/stalemate. */
function checkAfterPass(position, attackedByNextPlayer) {
  const present = presentTimelines(position);
  if (!present.length) return attackedByNextPlayer(position);
  const board = position.board.slice();
  for (const line of present) {
    const timeline = board[line], turn = timeline.length - 1;
    if (turn % 2 !== position.action % 2) continue;
    board[line] = timeline.slice();
    let passed = timeline[turn];
    // A forced pass only reads the repeated board. Sharing it avoids copying
    // every square and lets the search reuse its cached royal/piece scan.
    // Match turnFuncs.copy's normalization for sparse or empty custom boards.
    if (!passed?.length) passed = null;
    else for (let rank = 0; rank < passed.length; rank++) if (!passed[rank]) {
      passed = passed.filter(Boolean);
      if (!passed.length) passed = null;
      break;
    }
    board[line].push(passed);
  }
  return attackedByNextPlayer({ ...position, board });
}

export function inCheck(position) {
  return checkAfterPass(position, attackedByNextPlayer);
}

/** Reuse immutable board scans within one search; never retain across edits. */
export function createSearchRoyalSafety() {
  const safety = royalSafety.createCached();
  return {
    attackedByNextPlayer: safety.attackedByNextPlayer,
    inCheck: position => checkAfterPass(position, safety.attackedByNextPlayer),
  };
}

// History matters: a past royal or empty square can determine a temporal move.
// Keep the complete serialized key rather than trusting a short hash collision.
export function positionKey(position) {
  return JSON.stringify([position.action % 2, position.promotions, position.board]);
}

/**
 * Exact history keys for one immutable search. Public positions can be edited,
 * so positionKey deliberately does not share this cache between calls/searches.
 * Single boards and unchanged timelines share encodings across sibling
 * histories. Whole histories are weakly cached by their immutable outer
 * container, so submissions and repeated windows reuse their body without
 * retaining dead positions. Compact mode encodes signed pieces as individual
 * characters for internal search keys; the default keeps the public JSON form.
 */
export function createPositionKeyCache({ compact = false } = {}) {
  const boards = new WeakMap(), timelineKeys = new WeakMap(), histories = new WeakMap();
  const codes = [];
  function encodeBoard(board) {
    // All supported signed piece codes fit in one Latin-1 character. Array
    // boundaries remain explicit, so board dimensions and empty rows cannot
    // collide. These characters never overlap the '[' / ']' delimiters.
    // Retain exact JSON semantics for unusual caller-supplied cell values.
    if (!Array.isArray(board)) return `j${JSON.stringify(board)}`;
    // Reuse the scratch array and build a flat string in one allocation.
    // Character-by-character concatenation creates ropes that need flattening
    // before hashing; clearing this array would also discard its capacity.
    let length = 2;
    codes[0] = 99; codes[1] = 91;
    for (const row of board) {
      if (!Array.isArray(row)) return `j${JSON.stringify(board)}`;
      if (length + row.length + 2 > 4096) return `j${JSON.stringify(board)}`;
      codes[length++] = 91;
      for (const piece of row) {
        if (!Number.isInteger(piece) || piece < -32 || piece > 32) return `j${JSON.stringify(board)}`;
        codes[length++] = piece + 128;
      }
      codes[length++] = 93;
    }
    codes[length++] = 93;
    codes.length = length;
    return String.fromCharCode(...codes);
  }
  function boardKey(board) {
    if (!board) return 'null';
    let serialized = boards.get(board);
    if (serialized === undefined) {
      serialized = compact ? encodeBoard(board) : JSON.stringify(board);
      boards.set(board, serialized);
    }
    return serialized;
  }
  const keyPosition = position => {
    let history = histories.get(position.board);
    if (history === undefined) {
      const timelines = [];
      for (const timeline of position.board) {
        if (!timeline) { timelines.push('null'); continue; }
        let timelineKey = timelineKeys.get(timeline);
        if (timelineKey === undefined) {
          const turns = [];
          for (const board of timeline) {
            turns.push(boardKey(board));
          }
          timelineKey = `[${turns.join(',')}]`;
          timelineKeys.set(timeline, timelineKey);
        }
        timelines.push(timelineKey);
      }
      history = `[${timelines.join(',')}]`;
      histories.set(position.board, history);
    }
    const prefix = JSON.stringify([position.action % 2, position.promotions]);
    return `${prefix.slice(0, -1)},${history}]`;
  };
  // Inside one generated turn, moves only append boards. Every partial state
  // shares the starting history, side, and promotions. Deduplicate by the
  // exact appended boards and their timeline indices, avoiding repeated
  // hashing and retention of that common history. Search-table keys above
  // still contain the complete history; these smaller keys never leave the
  // traversal for which they were created.
  keyPosition.forAction = position => current => {
    let changes = '';
    for (let line = 0; line < current.board.length; line++) {
      const timeline = current.board[line];
      if (timeline === position.board[line] || !timeline) continue;
      const start = position.board[line]?.length ?? 0;
      if (timeline.length === start) continue;
      changes += `${line}:[`;
      for (let turn = start; turn < timeline.length; turn++) {
        if (turn !== start) changes += ',';
        changes += boardKey(timeline[turn]);
      }
      changes += '];';
    }
    return changes;
  };
  return keyPosition;
}

export function formatMove(position, move) {
  return raw.pgnFuncs.fromMove(move, position.board, position.action, '', true, true, true);
}

export function formatAction(position, moves) {
  return raw.pgnFuncs.fromAction(moves, position.board, position.action, '', true, true, true);
}

export function parseMove(position, input) {
  if (typeof input === 'string') {
    const token = stripComments(input).trim().replace(/(?:\s*\((?:>L|~T)[+-]?\d+\))+$/, '').trim();
    const match = TEMPORAL_MOVE.exec(token) || CASTLING_MOVE.exec(token) || SPATIAL_MOVE.exec(token);
    if (!match) {
      // The upstream exporter occasionally repeats a pawn's source file
      // (e.g. "bbxc6"). Accept only an exact generated representation of a
      // legal move; never restore its permissive prefix-only token parser.
      const exact = pseudoMoves(position).filter(move => [false, true].some(explicitBoard => {
        const notation = raw.pgnFuncs.fromMove(move, position.board, position.action, '', false, false, explicitBoard);
        return notation === token;
      }));
      if (exact.length === 1) return exact[0];
      throw new Error('Malformed move notation: unexpected or incomplete token.');
    }
    const fields = match.groups;
    const coordinates = value => {
      const parts = /^\(L?([+-]?\d+)T[+]?(\d+)\)$/.exec(value);
      let line = +parts[1];
      if (Math.abs(line) > 64 || +parts[2] > 2048) throw new Error('Move coordinates exceed supported import limits.');
      if (raw.boardFuncs.isEvenTimeline(position.board)) {
        if (parts[1] === '-0') line = -1;
        else if (parts[1] === '+0') line = 1;
        else if (line < 0) line--;
        else if (line > 0) line++;
      }
      const timeline = line < 0 ? -line * 2 - 1 : line * 2;
      const turn = (+parts[2] - 1) * 2 + position.action % 2 + (raw.boardFuncs.isTurnZero(position.board) ? 2 : 0);
      return [timeline, turn];
    };
    let startBoard;
    if (fields.board) startBoard = coordinates(fields.board);
    else if (position.board[0]) startBoard = [0, position.board[0].length - 1];
    else {
      const sources = position.board.flatMap((line, index) => line && (line.length - 1) % 2 === position.action % 2 ? [[index, line.length - 1]] : []);
      if (sources.length !== 1) throw new Error('Specify the source timeline and turn.');
      startBoard = sources[0];
    }
    const destination = fields.destination ? coordinates(fields.destination) : startBoard;
    const expectedPiece = raw.pieceFuncs.fromChar(fields.piece || (fields.castle ? 'K' : 'P'), position.action);
    const squareMatches = (square, text) => {
      if (!text) return true;
      const coordinate = /^([a-p]?)(\d*)$/.exec(text);
      return (!coordinate[1] || square[3] === coordinate[1].charCodeAt(0) - 97) && (!coordinate[2] || square[2] === +coordinate[2] - 1);
    };
    const candidates = pseudoMoves(position).filter(move => {
      if (move[0][0] !== startBoard[0] || move[0][1] !== startBoard[1] || move[1][0] !== destination[0] || move[1][1] !== destination[1]) return false;
      if (Math.abs(pieceAt(position.board, move[0])) !== expectedPiece) return false;
      if (fields.castle) return move.length === 4 && (fields.castle === 'O-O' ? move[1][3] > move[0][3] : move[1][3] < move[0][3]);
      if (!squareMatches(move[0], fields.source) || !squareMatches(move[1], fields.end)) return false;
      if (fields.promotion) return move[1][4] === raw.pieceFuncs.fromChar(fields.promotion, position.action);
      return move[1].length < 5;
    });
    if (candidates.length !== 1) throw new Error(candidates.length ? 'Ambiguous move: specify the source square.' : 'Illegal piece move in this position.');
    return candidates[0];
  }
  const move = raw.convertFuncs.move(input, position.board, position.action, position.promotions);
  const generated = pseudoMoves(position).find(candidate => equalMove(candidate, move));
  if (!generated) throw new Error('Illegal piece move in this position.');
  return generated;
}

export function validateAction(position, moves) {
  if (!Array.isArray(moves)) throw new Error('An action must be an ordered array of moves.');
  let next = position;
  for (const move of moves) next = applyMove(next, parseMove(next, move));
  return submitPosition(next);
}

/**
 * Exhaustive, lazy legal action generation. A yield is a legal submission, not
 * a leaf: the player may still move on optional boards. Individual moves may
 * expose a king temporarily; only the resulting submitted action must be safe.
 * Each move consumes at least one playable latest board, so a turn is finite.
 * Search can omit ordinary moves on optional boards; default rule enumeration
 * remains exhaustive. Recompute the present after every component move because
 * time travel can change which timelines are active and required.
 * orderMoves(current, moves, prefix) receives a copy of the component prefix
 * leading from the initial position to current, without submitting the turn.
 * firstOnly proves legal-turn existence, stopping after the first submission.
 */
export function* generateActions(position, options = {}) {
  const steps = generateActionSteps(position, options);
  const orderMoves = options.orderMoves, firstOnly = options.firstOnly;
  try {
    let step = steps.next();
    while (!step.done) {
      if (step.value.candidate) {
        yield step.value.candidate;
        if (firstOnly) return;
        step = steps.next();
      } else {
        const { current, moves, prefix } = step.value;
        step = steps.next(orderMoves ? orderMoves(current, moves, prefix.slice()) : moves);
      }
    }
  } finally { steps.return?.(); }
}

/**
 * The same legal traversal, with an awaitable orderMoves hook. The hook may
 * also return an async iterable of move batches: later batches are requested
 * only after earlier branches have been visited, keeping inference resumable.
 */
export async function* generateActionsAsync(position, options = {}) {
  const steps = generateActionSteps(position, options);
  const orderMoves = options.orderMoves, firstOnly = options.firstOnly;
  const batches = new Set();
  async function nextBatch(iterator) {
    const next = await iterator.next();
    if (next.done) batches.delete(iterator);
    return { moves: next.done ? [] : next.value, more: next.done ? null : iterator };
  }
  try {
    let step = steps.next();
    while (!step.done) {
      if (step.value.candidate) {
        yield step.value.candidate;
        if (firstOnly) return;
        step = steps.next();
      } else if (step.value.nextBatch) {
        step = steps.next(await nextBatch(step.value.nextBatch));
      } else {
        const { current, moves, prefix } = step.value;
        const ordered = orderMoves ? await orderMoves(current, moves, prefix.slice()) : moves;
        if (ordered?.[Symbol.asyncIterator]) {
          const iterator = ordered[Symbol.asyncIterator]();
          batches.add(iterator);
          step = steps.next(await nextBatch(iterator));
        } else step = steps.next(ordered);
      }
    }
  } finally {
    steps.return?.();
    // Close every nested ordering generator even when inference or a search
    // budget throws while a parent batch is suspended.
    await Promise.all([...batches].map(iterator => iterator.return?.()));
  }
}

// Both drivers share every legality, deduplication and pruning decision. The
// traversal pauses only to request move ordering or expose a legal submission.
function* generateActionSteps(position, { tick = () => {}, preferredAction = null, pruneUnsafe = true, tacticalOnly = false, firstOnly = false, cacheMoves = true, cacheUnsafeMoves = true, keyPosition = positionKey, generateMoves = pseudoMoves, skipOptionalSpatial = false, onSkipOptionalSpatial, royalSafety: searchRoyalSafety } = {}) {
  const { attackedByNextPlayer } = searchRoyalSafety ?? royalSafety.createCached();
  const timeline = position.board.length === 1 ? position.board[0] : null;
  if (firstOnly && !tacticalOnly && !Array.isArray(preferredAction)
      && timeline?.at(-1) && (timeline.length - 1) % 2 === position.action % 2) {
    // There is exactly one playable source. Every move consumes it and only
    // creates opponent-color frontiers, even when it branches into the past.
    // Thus royal safety alone proves submission after a move. An existence
    // probe needs no partial-state traversal or history serialization here.
    tick();
    const moves = generateMoves(position);
    let ordered = yield { current: position, moves, prefix: [] };
    const unsafe = new Set();
    while (ordered) {
      const batch = Object.hasOwn(ordered, 'more') ? ordered.moves : ordered;
      for (const move of batch) {
        const spatial = move[0][0] === move[1][0] && move[0][1] === move[1][1];
        if (pruneUnsafe && spatial && unsafe.size && unsafe.has(JSON.stringify(move))) continue;
        tick();
        const current = applyMove(position, move);
        tick();
        if (!attackedByNextPlayer(current)) {
          yield { candidate: { moves: [move], position: { ...current, action: current.action + 1 } } };
          return;
        }
        if (pruneUnsafe && cacheUnsafeMoves && spatial) unsafe.add(JSON.stringify(move));
      }
      if (!ordered.more) break;
      ordered = yield { nextBatch: ordered.more };
    }
    return;
  }
  const keyState = keyPosition.forAction?.(position) ?? keyPosition;
  const visited = new Set();
  const path = [];
  const unsafeSpatialMoves = new Set(), testedSpatialMoves = new Set(), moveKeys = new WeakMap();
  const spatial = move => move[0][0] === move[1][0] && move[0][1] === move[1][1];
  const moveKey = move => {
    if (!moveKeys.has(move)) moveKeys.set(move, JSON.stringify(move));
    return moveKeys.get(move);
  };
  const knownUnsafe = move => unsafeSpatialMoves.size > 0 && spatial(move) && unsafeSpatialMoves.has(moveKey(move));
  function learnUnsafeMove() {
    const move = path.at(-1);
    if (!cacheUnsafeMoves || !move || !spatial(move)) return;
    const key = moveKey(move);
    if (testedSpatialMoves.has(key)) return;
    testedSpatialMoves.add(key);
    // Test the move against the original history, with all other components
    // removed. If it already exposes a royal there, every combination that
    // includes this spatial board outcome is unsafe. Later components only
    // append opponent-color boards and cannot erase the existing attack.
    // Temporal arrivals can merge or branch depending on earlier components,
    // so their outcomes must never be rejected using this spatial-move cache.
    if (path.length === 1) unsafeSpatialMoves.add(key);
    else {
      tick();
      if (attackedByNextPlayer(applyMove(position, move))) unsafeSpatialMoves.add(key);
    }
  }
  let sourceMoves, preferredKey;
  const availableMoves = (current, restrict = skipOptionalSpatial, present) => {
    if (restrict) present ??= presentTimelines(current);
    if (!cacheMoves) {
      const moves = generateMoves(current);
      if (!restrict) return moves;
      const allowed = moves.filter(move => !spatial(move) || present.includes(move[0][0]));
      if (allowed.length !== moves.length) onSkipOptionalSpatial?.();
      return allowed;
    }
    // Upstream emits moves in source-timeline order. Index that fixed geometry
    // once, then test availability once per source instead of once per move.
    // A consumed board often has hundreds of moves and contributes none to
    // the rest of the action. Optional boards keep only their temporal moves.
    if (!sourceMoves) {
      const initialMoves = generateMoves(position);
      sourceMoves = [];
      if (position.board.length === 1) {
        if (initialMoves.length) sourceMoves.push({ line: 0, turn: initialMoves[0][0][1], moves: initialMoves });
      } else {
        for (const move of initialMoves) {
          const [line, turn] = move[0];
          let source = sourceMoves.at(-1);
          if (!source || source.line !== line) sourceMoves.push(source = { line, turn, moves: [] });
          source.moves.push(move);
        }
      }
    }
    const moves = [];
    let skipped = false;
    for (const source of sourceMoves) {
      if (current.board[source.line].length - 1 !== source.turn) continue;
      const allowed = restrict && !present.includes(source.line)
        ? (source.temporal ??= source.moves.filter(move => !spatial(move))) : source.moves;
      skipped ||= allowed.length !== source.moves.length;
      for (const move of allowed) moves.push(move);
    }
    if (skipped) onSkipOptionalSpatial?.();
    return moves;
  };
  // A preferred turn is an ordered sequence, not a set of favorite component
  // moves. Replay it before enumerating shorter legal prefixes or alternative
  // orders (which can create different branches). Validate it against current
  // geometry so a stale/illegal hint never becomes a playable action.
  if (Array.isArray(preferredAction)) {
    tick();
    let current = position, legal = true, tactical = false;
    const moves = [];
    for (const preferred of preferredAction) {
      tick();
      const move = availableMoves(current).find(candidate => equalMove(candidate, preferred));
      if (!move) { legal = false; break; }
      tactical ||= isTacticalMove(current, move);
      moves.push(move);
      current = applyMove(current, move);
      tick();
    }
    if (legal && (!tacticalOnly || tactical) && presentTimelines(current).length === 0 && !attackedByNextPlayer(current)) {
      if (!firstOnly) preferredKey = keyState(current);
      yield { candidate: { moves, position: { ...current, action: current.action + 1 } } };
    }
  }
  function* visit(current, hasTacticalMove = false) {
    tick();
    // Moves in this action originate and land on mover-color boards. An attack
    // from an opponent-color latest board onto an opponent-color royal square
    // therefore survives every remaining move: its source, target and entire
    // ray are immutable. This is not the forced-pass check, which other
    // component moves can still resolve.
    // Check before serializing history: dead partial turns need no state key.
    const unsafe = attackedByNextPlayer(current);
    if (pruneUnsafe && unsafe) { learnUnsafeMove(); return; }
    const stateKey = keyState(current);
    const key = (tacticalOnly && hasTacticalMove ? 't:' : '') + stateKey;
    if (visited.has(key)) return;
    visited.add(key);
    const canYield = stateKey !== preferredKey && (!tacticalOnly || hasTacticalMove) && !unsafe;
    const present = canYield || skipOptionalSpatial ? presentTimelines(current) : null;
    if (canYield && present.length === 0) {
      yield { candidate: { moves: path.slice(), position: { ...current, action: current.action + 1 } } };
    }
    // Every move consumes existing mover-color sources and creates only
    // opponent-color boards. Geometry, unmoved flags, and en-passant history
    // on every remaining source therefore stay fixed throughout this action.
    // A destination becoming historical changes branching, not its geometry.
    const moves = availableMoves(current, skipOptionalSpatial, present);
    // No mover-color board is added or changed within an action. Remaining
    // source pieces and capture targets are unchanged; consuming other sources
    // cannot create a capture or promotion that is absent from this move set.
    // This skips purely quiet combinations only in capture quiescence.
    if (tacticalOnly && !hasTacticalMove && !moves.some(move => isTacticalMove(current, move))) {
      if (!skipOptionalSpatial) return;
      // Time travel may activate a previously inactive capture source. Already
      // active future boards cannot become present within this action: existing
      // mover-color history stays fixed, so the present can only move earlier.
      const active = raw.boardFuncs.active(current.board);
      if (!availableMoves(current, false).some(move => !active.includes(move[0][0]) && isTacticalMove(current, move))) return;
    }
    let ordered = yield { current, moves, prefix: path };
    while (ordered) {
      const batch = Object.hasOwn(ordered, 'more') ? ordered.moves : ordered;
      for (const move of batch) {
        if (pruneUnsafe && knownUnsafe(move)) continue;
        tick();
        path.push(move);
        yield* visit(applyMove(current, move), tacticalOnly && (hasTacticalMove || isTacticalMove(current, move)));
        path.pop();
      }
      if (!ordered.more) break;
      ordered = yield { nextBatch: ordered.more };
    }
  }
  yield* visit(position);
}
