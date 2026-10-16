//! The native rules layer retains the JavaScript engine's sparse history and
//! complete-turn semantics. Old boards are shared and never edited in place.
use std::collections::HashSet;
use std::sync::{Arc, OnceLock};

use crate::model::{Action, Board, BoardHistory, Move, Position};

pub fn royal(piece: i8) -> bool {
    matches!(piece.abs(), 11 | 12 | 19 | 20)
}

pub fn board_at(history: &BoardHistory, line: i32, turn: i32) -> Option<&Board> {
    if line < 0 || turn < 0 {
        return None;
    }
    history
        .get(line as usize)?
        .as_ref()?
        .get(turn as usize)?
        .as_deref()
}

pub fn piece_at(history: &BoardHistory, square: &[i32]) -> Option<i8> {
    if square.len() < 4 || square[2] < 0 || square[3] < 0 {
        return None;
    }
    board_at(history, square[0], square[1])?
        .get(square[2] as usize)?
        .get(square[3] as usize)
        .copied()
}

pub fn is_even_timeline(history: &BoardHistory) -> bool {
    history.first().and_then(Option::as_ref).is_none()
}

pub fn timeline_move(source: i32, delta: i32, even: bool) -> i32 {
    let result = if source % 2 == 0 {
        source + delta * 2
    } else {
        source - delta * 2
    };
    if !even && result < 0 {
        -result - 1
    } else if even && result <= 0 {
        -result + 1
    } else {
        result
    }
}

fn line_coordinate(index: usize, even: bool) -> i32 {
    if index % 2 == 1 {
        -((index as i32 + 1) / 2)
    } else {
        index as i32 / 2 - i32::from(even)
    }
}

pub fn active_timelines(history: &BoardHistory) -> Vec<usize> {
    let mut negative = 0;
    let mut positive = 0;
    for (line, timeline) in history.iter().enumerate() {
        if timeline.is_none() {
            continue;
        }
        if line % 2 == 1 {
            negative = negative.max(line);
        } else {
            positive = positive.max(line);
        }
    }
    history
        .iter()
        .enumerate()
        .filter_map(|(line, timeline)| {
            (timeline.is_some()
                && if line % 2 == 1 {
                    line <= positive + 1
                } else {
                    line <= negative + 3
                })
            .then_some(line)
        })
        .collect()
}

// Generate the fixed movement tables once. Mask and sign order deliberately
// match the pinned geometry library, keeping tied search moves deterministic.
struct Geometry {
    steps: Vec<[i32; 4]>,
    rays: Vec<[i32; 4]>,
}
static GEOMETRY: OnceLock<Vec<Geometry>> = OnceLock::new();

fn signed_vectors(mask: u32, magnitudes: [i32; 4]) -> Vec<[i32; 4]> {
    let axes: Vec<_> = (0..4)
        .filter(|axis| mask & (1 << (3 - axis)) != 0)
        .collect();
    (0..1usize << axes.len())
        .map(|signs| {
            let mut vector = [0; 4];
            for (index, &axis) in axes.iter().enumerate() {
                vector[axis] = magnitudes[axis]
                    * if signs & (1 << (axes.len() - 1 - index)) == 0 {
                        1
                    } else {
                        -1
                    };
            }
            vector
        })
        .collect()
}

fn geometry() -> &'static [Geometry] {
    GEOMETRY.get_or_init(|| {
        (0..13)
            .map(|kind| {
                let mut steps = Vec::new();
                let mut rays = Vec::new();
                for mask in 1u32..16 {
                    let dimensions = mask.count_ones();
                    if kind == 3 && dimensions == 2 {
                        let axes: Vec<_> = (0..4)
                            .filter(|axis| mask & (1 << (3 - axis)) != 0)
                            .collect();
                        for double_axis in [axes[1], axes[0]] {
                            let mut magnitudes = [1; 4];
                            magnitudes[double_axis] = 2;
                            steps.extend(signed_vectors(mask, magnitudes));
                        }
                    }
                    if kind == 6 || kind == 9 {
                        steps.extend(signed_vectors(mask, [1; 4]));
                    }
                    let ray = match kind {
                        2 => dimensions == 2,
                        4 => dimensions == 1,
                        5 | 10 => true,
                        7 => dimensions <= 2,
                        11 => dimensions == 3,
                        12 => dimensions == 4,
                        _ => false,
                    };
                    if ray {
                        let mut vectors = signed_vectors(mask, [1; 4]);
                        if matches!(kind, 5 | 10) && mask == 3 {
                            vectors.swap(1, 2);
                        }
                        rays.extend(vectors);
                    }
                }
                Geometry { steps, rays }
            })
            .collect()
    })
}

pub fn move_pos(piece: i8) -> &'static [[i32; 4]] {
    geometry()
        .get((piece.unsigned_abs() as usize).div_ceil(2))
        .map_or(&[], |entry| entry.steps.as_slice())
}

pub fn move_vecs(piece: i8) -> &'static [[i32; 4]] {
    geometry()
        .get((piece.unsigned_abs() as usize).div_ceil(2))
        .map_or(&[], |entry| entry.rays.as_slice())
}

fn available_promotions(position: &Position) -> Vec<i8> {
    if !position.promotions.is_empty() {
        return position.promotions.clone();
    }
    let mut pieces = HashSet::new();
    for timeline in position.board.iter().flatten() {
        for squares in timeline.iter().flatten() {
            for &piece in squares.iter().flatten() {
                let piece = piece.abs();
                if matches!(piece, 3..=10 | 13 | 14 | 17 | 18) {
                    pieces.insert(piece);
                    pieces.insert(if piece % 2 == 0 { piece - 1 } else { piece + 1 });
                }
            }
        }
    }
    let mut choices: Vec<_> = pieces.into_iter().collect();
    choices.sort_unstable_by(|a, b| b.cmp(a));
    choices
}

fn append_pawn_moves(
    position: &Position,
    from: &[i32],
    piece: i8,
    even: bool,
    moves: &mut Vec<Move>,
    promotions: &[i8],
) {
    let [line, turn, rank, file] = [from[0], from[1], from[2], from[3]];
    let color = piece.abs() % 2;
    let forward = if color == 0 { 1 } else { -1 };
    let next_rank = rank + forward;
    let promotion_rank = if color == 0 {
        board_at(&position.board, line, turn).unwrap().len() as i32 - 1
    } else {
        0
    };
    let append = |destination: [i32; 4], promotes: bool, moves: &mut Vec<Move>| {
        if promotes && destination[2] == promotion_rank {
            for &promotion in promotions
                .iter()
                .filter(|&&promotion| promotion % 2 == color)
            {
                let mut to = destination.to_vec();
                to.push(promotion as i32);
                moves.push(vec![from.to_vec(), to]);
            }
        } else {
            moves.push(vec![from.to_vec(), destination.to_vec()]);
        }
    };
    if piece_at(&position.board, &[line, turn, next_rank, file]) == Some(0) {
        append([line, turn, next_rank, file], true, moves);
        if piece < 0
            && piece_at(&position.board, &[line, turn, next_rank + forward, file]) == Some(0)
        {
            append([line, turn, next_rank + forward, file], true, moves);
        }
    }
    for destination_file in [file + 1, file - 1] {
        let Some(target) = piece_at(&position.board, &[line, turn, next_rank, destination_file])
        else {
            continue;
        };
        if target != 0 && target.abs() % 2 != color && !royal(target) {
            append([line, turn, next_rank, destination_file], true, moves);
        }
        let neighbor =
            piece_at(&position.board, &[line, turn, rank, destination_file]).map(|p| p - color);
        if !matches!(neighbor, Some(1 | 15)) {
            continue;
        }
        let previous_rank = next_rank + forward;
        if piece_at(
            &position.board,
            &[line, turn, previous_rank, destination_file],
        ) != Some(0)
            || piece_at(&position.board, &[line, turn - 2, rank, destination_file]) != Some(0)
        {
            continue;
        }
        let origin = piece_at(
            &position.board,
            &[line, turn - 2, previous_rank, destination_file],
        )
        .map(|p| p + color);
        if matches!(origin, Some(-1 | -15)) && !royal(target) {
            moves.push(vec![
                from.to_vec(),
                vec![line, turn, next_rank, destination_file],
                vec![line, turn, rank, destination_file],
            ]);
        }
    }
    let adjacent = timeline_move(line, -forward, even);
    if piece_at(&position.board, &[adjacent, turn, rank, file]) == Some(0) {
        append([adjacent, turn, rank, file], false, moves);
        let twice = timeline_move(line, -2 * forward, even);
        if piece < 0 && piece_at(&position.board, &[twice, turn, rank, file]) == Some(0) {
            append([twice, turn, rank, file], false, moves);
        }
    }
    for destination_turn in [turn + 2, turn - 2] {
        if let Some(target) = piece_at(&position.board, &[adjacent, destination_turn, rank, file]) {
            if target != 0 && target.abs() % 2 != color && !royal(target) {
                append([adjacent, destination_turn, rank, file], false, moves);
            }
        }
    }
    if piece.abs() < 15 {
        return;
    }
    for destination in [
        [adjacent, turn, rank, file + 1],
        [adjacent, turn, rank, file - 1],
        [adjacent, turn, next_rank, file],
        [line, turn - 2, next_rank, file],
    ] {
        if let Some(target) = piece_at(&position.board, &destination) {
            if target != 0 && target.abs() % 2 != color && !royal(target) {
                append(destination, true, moves);
            }
        }
    }
}

// Castling uses the pinned library's spatial attack predicate, including its
// special-piece conventions. Full submission safety is checked separately.
fn spatially_attacked(board: &Board, rank: i32, file: i32, player: i8) -> bool {
    let at = |r: i32, f: i32| -> Option<i8> {
        if r < 0 || f < 0 {
            None
        } else {
            board.get(r as usize)?.get(f as usize).copied().map(i8::abs)
        }
    };
    for step in move_pos(6).iter().take(8) {
        if at(rank + step[2], file + step[3]).is_some_and(|piece| piece != 0 && piece - player == 5)
        {
            return true;
        }
    }
    for (index, vector) in move_vecs(10).iter().take(8).enumerate() {
        let (dr, df) = (vector[2], vector[3]);
        let (mut r, mut f) = (rank + dr, file + df);
        let Some(piece) = at(r, f) else {
            continue;
        };
        if piece != 0 {
            let compared = piece - player;
            if piece % 2 == player {
                continue;
            }
            if index < 4 {
                if matches!(compared, 7..=13 | 17 | 19) {
                    return true;
                }
            } else {
                if matches!(compared, 5 | 7 | 21 | 23) {
                    continue;
                }
                return true;
            }
        }
        r += dr;
        f += df;
        while let Some(piece) = at(r, f) {
            if piece != 0 {
                if piece % 2 != player {
                    let compared = piece - player;
                    if (index < 4 && matches!(compared, 7 | 9 | 13 | 19))
                        || (index >= 4 && matches!(compared, 3 | 9 | 13 | 19))
                    {
                        return true;
                    }
                }
                break;
            }
            r += dr;
            f += df;
        }
    }
    false
}

fn append_castling_moves(position: &Position, from: &[i32], color: i8, moves: &mut Vec<Move>) {
    let squares = board_at(&position.board, from[0], from[1]).unwrap();
    let row = &squares[from[2] as usize];
    for direction in [1, -1] {
        let one = from[3] + direction;
        let two = one + direction;
        if one < 0
            || two < 0
            || row.get(one as usize) != Some(&0)
            || row.get(two as usize) != Some(&0)
        {
            continue;
        }
        let mut rook = two + direction;
        while rook >= 0 && (rook as usize) < row.len() {
            if row[rook as usize] != 0 {
                if row[rook as usize] - color == -8
                    && !spatially_attacked(squares, from[2], from[3], color)
                    && !spatially_attacked(squares, from[2], one, color)
                    && !spatially_attacked(squares, from[2], two, color)
                {
                    moves.push(vec![
                        from.to_vec(),
                        vec![from[0], from[1], from[2], two],
                        vec![from[0], from[1], from[2], rook],
                        vec![from[0], from[1], from[2], one],
                    ]);
                }
                break;
            }
            rook += direction;
        }
    }
}

pub fn pseudo_moves(position: &Position) -> Vec<Move> {
    let mut moves = Vec::new();
    let color = (position.action % 2) as i8;
    let even = is_even_timeline(&position.board);
    let promotions = available_promotions(position);
    for (line, timeline) in position.board.iter().enumerate() {
        let Some(timeline) = timeline else {
            continue;
        };
        let Some(turn) = timeline.len().checked_sub(1) else {
            continue;
        };
        if turn % 2 != position.action % 2 {
            continue;
        }
        let Some(squares) = timeline[turn].as_ref() else {
            continue;
        };
        for (rank, row) in squares.iter().enumerate() {
            for (file, &piece) in row.iter().enumerate() {
                if piece == 0 || piece.abs() % 2 != color {
                    continue;
                }
                let from = vec![line as i32, turn as i32, rank as i32, file as i32];
                if matches!(piece.abs(), 1 | 2 | 15 | 16) {
                    append_pawn_moves(position, &from, piece, even, &mut moves, &promotions);
                    continue;
                }
                for vector in move_pos(piece) {
                    let to = vec![
                        timeline_move(line as i32, vector[0], even),
                        turn as i32 + vector[1] * 2,
                        rank as i32 + vector[2],
                        file as i32 + vector[3],
                    ];
                    if let Some(target) = piece_at(&position.board, &to) {
                        if (target == 0 || target.abs() % 2 != color) && !royal(target) {
                            moves.push(vec![from.clone(), to]);
                        }
                    }
                }
                for vector in move_vecs(piece) {
                    let mut to = vec![
                        timeline_move(line as i32, vector[0], even),
                        turn as i32 + vector[1] * 2,
                        rank as i32 + vector[2],
                        file as i32 + vector[3],
                    ];
                    while let Some(target) = piece_at(&position.board, &to) {
                        if target != 0 {
                            if target.abs() % 2 != color && !royal(target) {
                                moves.push(vec![from.clone(), to.clone()]);
                            }
                            break;
                        }
                        moves.push(vec![from.clone(), to.clone()]);
                        to[0] = timeline_move(to[0], vector[0], even);
                        to[1] += vector[1] * 2;
                        to[2] += vector[2];
                        to[3] += vector[3];
                    }
                }
                if piece == -11 || piece == -12 {
                    append_castling_moves(position, &from, color, &mut moves);
                }
            }
        }
    }
    moves
}

pub fn is_tactical_move(position: &Position, mv: &Move) -> bool {
    piece_at(&position.board, &mv[1]).is_some_and(|piece| piece != 0)
        || mv.len() == 3
        || mv[1].len() > 4
}

fn set_turn(history: &mut BoardHistory, line: usize, turn: usize, board: Board) {
    if history.len() <= line {
        history.resize_with(line + 1, || None);
    }
    let timeline = history[line].get_or_insert_with(Vec::new);
    if timeline.len() <= turn {
        timeline.resize_with(turn + 1, || None);
    }
    timeline[turn] = Some(Arc::new(board));
}

/// Apply a generated component. All historical boards remain immutable.
pub fn apply_move(position: &Position, mv: &Move) -> Position {
    let mut result = position.clone();
    if mv.len() < 2 || mv[0].len() < 4 || mv[1].len() < 4 {
        return result;
    }
    let (from, to) = (&mv[0], &mv[1]);
    let Some(piece) = piece_at(&position.board, from) else {
        return result;
    };
    if piece == 0
        || position.board[from[0] as usize].as_ref().unwrap().len() - 1 != from[1] as usize
    {
        return result;
    }
    let moving = to
        .get(4)
        .copied()
        .filter(|p| *p != 0)
        .map_or(piece.abs(), |p| p as i8);
    let mut source = board_at(&position.board, from[0], from[1]).unwrap().clone();
    source[from[2] as usize][from[3] as usize] = 0;
    if from[..2] == to[..2] {
        source[to[2] as usize][to[3] as usize] = moving;
    } else {
        let mut destination = board_at(&position.board, to[0], to[1])
            .expect("generated destination")
            .clone();
        destination[to[2] as usize][to[3] as usize] = moving;
        if position.board[to[0] as usize].as_ref().unwrap().len() - 1 == to[1] as usize {
            set_turn(
                &mut result.board,
                to[0] as usize,
                to[1] as usize + 1,
                destination,
            );
        } else {
            let mut outer = 0;
            for (index, timeline) in result.board.iter().enumerate().skip(1) {
                if timeline.is_some() && index % 2 == to[1] as usize % 2 {
                    outer = outer.max(index);
                }
            }
            let line = if outer == 0 {
                if to[1] % 2 == 0 {
                    2
                } else {
                    1
                }
            } else {
                outer + 2
            };
            set_turn(&mut result.board, line, to[1] as usize + 1, destination);
        }
    }
    set_turn(
        &mut result.board,
        from[0] as usize,
        from[1] as usize + 1,
        source,
    );
    if let Some(extra_source) = mv.get(2) {
        if let Some(extra_to) = mv.get(3) {
            let piece = extra_to.get(4).copied().filter(|p| *p != 0).map_or_else(
                || piece_at(&position.board, extra_source).unwrap().abs(),
                |p| p as i8,
            );
            let board = result.board[extra_to[0] as usize].as_mut().unwrap()
                [extra_to[1] as usize + 1]
                .as_mut()
                .unwrap();
            Arc::make_mut(board)[extra_to[2] as usize][extra_to[3] as usize] = piece;
        }
        let board = result.board[extra_source[0] as usize].as_mut().unwrap()
            [extra_source[1] as usize + 1]
            .as_mut()
            .unwrap();
        Arc::make_mut(board)[extra_source[2] as usize][extra_source[3] as usize] = 0;
    }
    result
}

pub fn present_timelines(position: &Position) -> Vec<usize> {
    let active = active_timelines(&position.board);
    let color = position.action % 2;
    let mut latest = Vec::with_capacity(active.len());
    let mut lowest = None;
    for &line in &active {
        let timeline = position.board[line].as_ref().unwrap();
        let end = timeline.iter().rposition(Option::is_some);
        latest.push(end.unwrap_or(0));
        let current = timeline
            .iter()
            .enumerate()
            .rev()
            .find_map(|(turn, board)| (turn % 2 == color && board.is_some()).then_some(turn));
        if let Some(turn) = current {
            lowest = Some(lowest.map_or(turn, |previous: usize| previous.min(turn)));
        }
    }
    let Some(lowest) = lowest else {
        return Vec::new();
    };
    if latest.iter().any(|&turn| turn < lowest) {
        return Vec::new();
    }
    active
        .into_iter()
        .zip(latest)
        .filter_map(|(line, turn)| (turn == lowest).then_some(line))
        .collect()
}

/// Capture threats originate on opponent-color frontiers and can target every
/// historical royal of that half-turn color, including inactive timelines.
pub fn attacked_by_next_player(position: &Position) -> bool {
    let color = ((position.action + 1) % 2) as i8;
    if !position.board.iter().flatten().any(|timeline| {
        !timeline.is_empty()
            && (timeline.len() - 1) % 2 == color as usize
            && timeline.last().unwrap().is_some()
    }) {
        return false;
    }
    let even = is_even_timeline(&position.board);
    let mut targets = Vec::new();
    for (line, timeline) in position.board.iter().enumerate() {
        for (turn, squares) in timeline.iter().flatten().enumerate() {
            if turn % 2 != color as usize {
                continue;
            }
            let Some(squares) = squares else {
                continue;
            };
            for (rank, row) in squares.iter().enumerate() {
                for (file, &piece) in row.iter().enumerate() {
                    if royal(piece) && piece.abs() % 2 != color {
                        targets.push((line, turn, rank as i32, file as i32));
                    }
                }
            }
        }
    }
    if targets.is_empty() {
        return false;
    }
    let mut can_promote = None;
    for (line, timeline) in position.board.iter().enumerate() {
        let Some(timeline) = timeline else {
            continue;
        };
        let Some(turn) = timeline.len().checked_sub(1) else {
            continue;
        };
        if turn % 2 != color as usize {
            continue;
        }
        let Some(squares) = timeline[turn].as_ref() else {
            continue;
        };
        for (rank, row) in squares.iter().enumerate() {
            for (file, &piece) in row.iter().enumerate() {
                let piece = piece.abs();
                if piece == 0 || piece % 2 != color {
                    continue;
                }
                for &(target_line, target_turn, target_rank, target_file) in &targets {
                    let delta = [
                        line_coordinate(target_line, even) - line_coordinate(line, even),
                        (target_turn as i32 - turn as i32) / 2,
                        target_rank - rank as i32,
                        target_file - file as i32,
                    ];
                    let [dl, dt, dr, df] = delta;
                    if matches!(piece, 1 | 2 | 15 | 16) {
                        let forward = if color == 0 { 1 } else { -1 };
                        let spatial = dl == 0 && dt == 0 && dr == forward && df.abs() == 1;
                        let temporal = dl == -forward && dt.abs() == 1 && dr == 0 && df == 0;
                        let brawn = piece >= 15
                            && ((dl == -forward
                                && dt == 0
                                && ((dr == 0 && df.abs() == 1) || (dr == forward && df == 0)))
                                || (dl == 0 && dt == -1 && dr == forward && df == 0));
                        if !spatial && !temporal && !brawn {
                            continue;
                        }
                        let promotion_rank = if color == 0 {
                            squares.len() as i32 - 1
                        } else {
                            0
                        };
                        if !temporal && target_rank == promotion_rank {
                            let promotion = *can_promote.get_or_insert_with(|| {
                                available_promotions(position)
                                    .iter()
                                    .any(|piece| piece % 2 == color)
                            });
                            if !promotion {
                                continue;
                            }
                        }
                        return true;
                    }
                    if move_pos(piece).contains(&delta) {
                        return true;
                    }
                    let distance = delta.iter().map(|value| value.abs()).max().unwrap();
                    if distance == 0
                        || delta
                            .iter()
                            .any(|value| *value != 0 && value.abs() != distance)
                    {
                        continue;
                    }
                    let step = delta.map(i32::signum);
                    if !move_vecs(piece).contains(&step) {
                        continue;
                    }
                    let blocked = (1..distance).any(|offset| {
                        piece_at(
                            &position.board,
                            &[
                                timeline_move(line as i32, step[0] * offset, even),
                                turn as i32 + step[1] * offset * 2,
                                rank as i32 + step[2] * offset,
                                file as i32 + step[3] * offset,
                            ],
                        ) != Some(0)
                    });
                    if !blocked {
                        return true;
                    }
                }
            }
        }
    }
    false
}

pub fn can_submit(position: &Position) -> bool {
    present_timelines(position).is_empty() && !attacked_by_next_player(position)
}

pub fn in_check(position: &Position) -> bool {
    let mut passed = position.clone();
    for line in present_timelines(position) {
        let timeline = passed.board[line].as_mut().unwrap();
        if !timeline.is_empty() && (timeline.len() - 1) % 2 == position.action % 2 {
            timeline.push(
                timeline
                    .last()
                    .cloned()
                    .flatten()
                    .filter(|board| !board.is_empty()),
            );
        }
    }
    attacked_by_next_player(&passed)
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub position: Position,
    pub moves: Action,
}

struct Frame {
    position: Position,
    path: Action,
    tactical: bool,
    entered: bool,
    moves: Option<std::vec::IntoIter<Move>>,
}

/// Lazy whole-turn enumeration. The budget callback is invoked before each
/// partial-state visit and component application; timeout is never a mate.
pub struct ActionGenerator {
    initial: Position,
    initial_moves: Option<Vec<Move>>,
    preferred: Option<Action>,
    preferred_key: Option<Vec<u8>>,
    stack: Vec<Frame>,
    visited: HashSet<Vec<u8>>,
    restricted: bool,
    tactical_only: bool,
    pub policy_pruned: bool,
}

impl ActionGenerator {
    pub fn new(
        position: &Position,
        preferred: Option<Action>,
        restricted: bool,
        tactical_only: bool,
    ) -> Self {
        Self {
            initial: position.clone(),
            initial_moves: None,
            preferred,
            preferred_key: None,
            stack: vec![Frame {
                position: position.clone(),
                path: Vec::new(),
                tactical: false,
                entered: false,
                moves: None,
            }],
            visited: HashSet::new(),
            restricted,
            tactical_only,
            policy_pruned: false,
        }
    }

    // Keys include exact appended boards and their indices. Common history is
    // fixed for this iterator, so sharing it in every key is unnecessary.
    fn state_key(&self, position: &Position) -> Vec<u8> {
        let mut key = Vec::new();
        for (line, timeline) in position.board.iter().enumerate() {
            let Some(timeline) = timeline else {
                continue;
            };
            let start = self
                .initial
                .board
                .get(line)
                .and_then(Option::as_ref)
                .map_or(0, Vec::len);
            if timeline.len() <= start {
                continue;
            }
            key.extend_from_slice(&(line as u32).to_le_bytes());
            key.extend_from_slice(&(timeline.len() as u32).to_le_bytes());
            for board in &timeline[start..] {
                match board {
                    None => key.push(0),
                    Some(board) => {
                        key.push(1);
                        key.extend_from_slice(&(board.len() as u32).to_le_bytes());
                        for row in board.iter() {
                            key.extend_from_slice(&(row.len() as u32).to_le_bytes());
                            key.extend(row.iter().map(|piece| *piece as u8));
                        }
                    }
                }
            }
        }
        key
    }

    fn available_moves(&mut self, position: &Position, restricted: bool) -> Vec<Move> {
        if self.initial_moves.is_none() {
            self.initial_moves = Some(pseudo_moves(&self.initial));
        }
        let present = if restricted {
            present_timelines(position)
        } else {
            Vec::new()
        };
        self.initial_moves
            .as_ref()
            .unwrap()
            .iter()
            .filter(|mv| {
                let (line, turn) = (mv[0][0] as usize, mv[0][1] as usize);
                if position.board[line].as_ref().unwrap().len() - 1 != turn {
                    return false;
                }
                if restricted && mv[0][..2] == mv[1][..2] && !present.contains(&line) {
                    self.policy_pruned = true;
                    return false;
                }
                true
            })
            .cloned()
            .collect()
    }

    pub fn next<E>(
        &mut self,
        tick: &mut impl FnMut() -> Result<(), E>,
        order_moves: &mut impl FnMut(&Position, &mut Vec<Move>),
    ) -> Result<Option<Candidate>, E> {
        if let Some(preferred) = self.preferred.take() {
            tick()?;
            let mut current = self.initial.clone();
            let mut replay = Vec::new();
            let mut tactical = false;
            let mut legal = true;
            for wanted in preferred {
                tick()?;
                let available = self.available_moves(&current, self.restricted);
                let Some(mv) = available.into_iter().find(|mv| *mv == wanted) else {
                    legal = false;
                    break;
                };
                tactical |= is_tactical_move(&current, &mv);
                current = apply_move(&current, &mv);
                replay.push(mv);
                tick()?;
            }
            if legal && (!self.tactical_only || tactical) && can_submit(&current) {
                self.preferred_key = Some(self.state_key(&current));
                current.action += 1;
                return Ok(Some(Candidate {
                    position: current,
                    moves: replay,
                }));
            }
        }
        while let Some(mut frame) = self.stack.pop() {
            if !frame.entered {
                tick()?;
                if attacked_by_next_player(&frame.position) {
                    continue;
                }
                let state_key = self.state_key(&frame.position);
                let mut visited_key = state_key.clone();
                visited_key.push(u8::from(self.tactical_only && frame.tactical));
                if !self.visited.insert(visited_key) {
                    continue;
                }
                frame.entered = true;
                if self.preferred_key.as_ref() != Some(&state_key)
                    && (!self.tactical_only || frame.tactical)
                    && present_timelines(&frame.position).is_empty()
                {
                    let mut position = frame.position.clone();
                    position.action += 1;
                    let moves = frame.path.clone();
                    self.stack.push(frame);
                    return Ok(Some(Candidate { position, moves }));
                }
            }
            if frame.moves.is_none() {
                let mut moves = self.available_moves(&frame.position, self.restricted);
                if self.tactical_only
                    && !frame.tactical
                    && !moves.iter().any(|mv| is_tactical_move(&frame.position, mv))
                {
                    if !self.restricted {
                        continue;
                    }
                    let active = active_timelines(&frame.position.board);
                    let potential = self
                        .available_moves(&frame.position, false)
                        .iter()
                        .any(|mv| {
                            !active.contains(&(mv[0][0] as usize))
                                && is_tactical_move(&frame.position, mv)
                        });
                    if !potential {
                        continue;
                    }
                }
                order_moves(&frame.position, &mut moves);
                frame.moves = Some(moves.into_iter());
            }
            if let Some(mv) = frame.moves.as_mut().unwrap().next() {
                tick()?;
                let position = apply_move(&frame.position, &mv);
                let tactical = frame.tactical || is_tactical_move(&frame.position, &mv);
                let mut path = frame.path.clone();
                path.push(mv);
                self.stack.push(frame);
                self.stack.push(Frame {
                    position,
                    path,
                    tactical,
                    entered: false,
                    moves: None,
                });
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn position(board: Board) -> Position {
        Position {
            board: vec![Some(vec![Some(Arc::new(board))])],
            action: 0,
            promotions: vec![10, 9, 8, 7, 6, 5, 4, 3],
        }
    }

    #[test]
    fn geometry_covers_all_dimensions_and_pinned_direction_order() {
        assert_eq!(move_pos(6).len(), 48);
        assert_eq!(move_pos(12).len(), 80);
        assert_eq!(move_vecs(4).len(), 24);
        assert_eq!(move_vecs(8).len(), 8);
        assert_eq!(move_vecs(10).len(), 80);
        assert_eq!(move_vecs(14).len(), 32);
        assert_eq!(move_vecs(22).len(), 32);
        assert_eq!(move_vecs(24).len(), 16);
        assert_eq!(move_pos(6)[0], [0, 0, 1, 2]);
        assert_eq!(move_vecs(10)[5], [0, 0, -1, 1]);
        assert_eq!(timeline_move(0, -1, false), 1);
        assert_eq!(timeline_move(1, 1, false), 0);
        assert_eq!(timeline_move(2, -1, true), 1);
    }

    #[test]
    fn spatial_move_shares_old_history_and_consumes_source() {
        let initial = position(vec![vec![12, 0, 8], vec![0, 0, 0], vec![0, 0, 11]]);
        let moved = apply_move(&initial, &vec![vec![0, 0, 0, 2], vec![0, 0, 1, 2]]);
        assert_eq!(initial.board[0].as_ref().unwrap().len(), 1);
        assert_eq!(moved.board[0].as_ref().unwrap().len(), 2);
        assert!(Arc::ptr_eq(
            initial.board[0].as_ref().unwrap()[0].as_ref().unwrap(),
            moved.board[0].as_ref().unwrap()[0].as_ref().unwrap()
        ));
        assert_eq!(piece_at(&moved.board, &[0, 0, 0, 2]), Some(8));
        assert_eq!(piece_at(&moved.board, &[0, 1, 0, 2]), Some(0));
        assert!(present_timelines(&moved).is_empty());
    }

    #[test]
    fn historical_arrival_creates_sparse_branch() {
        let mut initial = position(vec![vec![12, 0, 8], vec![0, 0, 0], vec![0, 0, 11]]);
        let board = initial.board[0].as_ref().unwrap()[0].clone();
        initial.board[0]
            .as_mut()
            .unwrap()
            .extend([board.clone(), board]);
        let moved = apply_move(&initial, &vec![vec![0, 2, 0, 2], vec![0, 0, 1, 2]]);
        assert_eq!(moved.board.len(), 3);
        assert!(moved.board[1].is_none());
        assert!(moved.board[2].as_ref().unwrap()[0].is_none());
        assert_eq!(piece_at(&moved.board, &[2, 1, 1, 2]), Some(8));
        assert_eq!(piece_at(&initial.board, &[0, 0, 1, 2]), Some(0));
    }

    #[test]
    fn whole_turns_never_capture_royals_or_submit_an_exposed_king() {
        let initial = position(vec![vec![12, 8, 7], vec![0, 0, 0], vec![0, 0, 11]]);
        let pseudo = pseudo_moves(&initial);
        assert!(pseudo
            .iter()
            .all(|mv| !piece_at(&initial.board, &mv[1]).is_some_and(royal)));
        let mut generator = ActionGenerator::new(&initial, None, false, false);
        let mut count = 0;
        while let Some(candidate) = generator
            .next(&mut || Ok::<_, ()>(()), &mut |_, _| {})
            .unwrap()
        {
            let mut unsubmitted = candidate.position.clone();
            unsubmitted.action -= 1;
            assert!(can_submit(&unsubmitted));
            assert_eq!(candidate.moves.len(), 1);
            count += 1;
        }
        assert!(count > 0);
    }
}
