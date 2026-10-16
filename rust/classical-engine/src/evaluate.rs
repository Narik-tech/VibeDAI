//! Hand-tuned classical evaluation, ported from the JavaScript reference.
//! Only timeline frontiers contribute material. Historical boards contribute
//! royal exposure and travel opportunities, never extra material copies.
use crate::heuristics::{schema, Heuristics};
use crate::model::{Board, BoardHistory, Position, Timeline};
use crate::rules::{active_timelines, is_even_timeline, move_pos, move_vecs, timeline_move};
use serde_json::{json, Value};
use std::collections::HashMap;

pub const PIECE_VALUES: [f64; 13] = [
    0.0, 100.0, 355.0, 340.0, 550.0, 1150.0, 0.0, 900.0, 140.0, 370.0, 0.0, 450.0, 350.0,
];
const VALUE_KEYS: [&str; 13] = [
    "",
    "pawnValue",
    "bishopValue",
    "knightValue",
    "rookValue",
    "queenValue",
    "",
    "princessValue",
    "brawnValue",
    "commonKingValue",
    "",
    "unicornValue",
    "dragonValue",
];
const COMPONENTS: [&str; 6] = [
    "material",
    "activity",
    "kingSafety",
    "temporal",
    "timelines",
    "travel",
];
const COMPONENT_WEIGHTS: [&str; 6] = [
    "materialWeight",
    "activityWeight",
    "kingSafetyWeight",
    "temporalWeight",
    "timelinesWeight",
    "travelWeight",
];
const AXES: [[i32; 2]; 4] = [[1, 0], [-1, 0], [0, 1], [0, -1]];
const DIAGONALS: [[i32; 2]; 4] = [[1, 1], [1, -1], [-1, 1], [-1, -1]];
const DIRECTIONS: [[i32; 2]; 8] = [
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
    [1, 1],
    [1, -1],
    [-1, 1],
    [-1, -1],
];
const KNIGHT_STEPS: [[i32; 2]; 8] = [
    [1, 2],
    [2, 1],
    [-1, 2],
    [-2, 1],
    [1, -2],
    [2, -1],
    [-1, -2],
    [-2, -1],
];
type Features = Option<HashMap<&'static str, f64>>;

#[inline]
fn owner(piece: i8) -> usize {
    piece.unsigned_abs() as usize % 2
}
#[inline]
fn kind(piece: i8) -> usize {
    (piece.unsigned_abs() as usize).div_ceil(2)
}
#[inline]
fn royal(piece_type: usize) -> bool {
    matches!(piece_type, 6 | 10)
}
#[inline]
fn sign(color: usize) -> f64 {
    if color == 0 {
        1.0
    } else {
        -1.0
    }
}
#[inline]
fn square(board: &Board, r: i32, f: i32) -> Option<i8> {
    if r < 0 || f < 0 {
        return None;
    }
    board
        .get(r as usize)
        .and_then(|row| row.get(f as usize))
        .copied()
}
fn history_square(board: &BoardHistory, l: i32, t: i32, r: i32, f: i32) -> Option<i8> {
    if l < 0 || t < 0 {
        return None;
    }
    let squares = board.get(l as usize)?.as_ref()?.get(t as usize)?.as_ref()?;
    square(squares, r, f)
}
fn coordinate(index: usize, even: bool) -> i32 {
    let value = if index % 2 == 1 {
        -((index as i32 + 1) / 2)
    } else {
        index as i32 / 2
    };
    if even && value > 0 {
        value - 1
    } else {
        value
    }
}
fn timeline_index(coordinate: i32, even: bool) -> usize {
    if coordinate < 0 {
        (-coordinate * 2 - 1) as usize
    } else {
        ((coordinate + i32::from(even)) * 2) as usize
    }
}
fn record(features: &mut Features, key: &'static str, value: f64) {
    if let Some(features) = features {
        *features.entry(key).or_insert(0.0) += value;
    }
}
fn feature_component(key: &str) -> Option<usize> {
    if VALUE_KEYS.contains(&key) && !key.is_empty() {
        return Some(0);
    }
    match key {
        "pawnAdvanceWeight" | "pawnCenterWeight" | "doubledPawnWeight" | "isolatedPawnWeight"
        | "passedPawnWeight" | "knightCenterWeight" | "bishopCenterWeight"
        | "pieceCenterWeight" | "royalCenterWeight" | "mobilityWeight" | "developmentWeight"
        | "rookFileWeight" => Some(1),
        "shelterWeight" | "nearbyEnemyWeight" | "flankSafetyWeight" | "corridorWeight"
        | "worstKingWeight" => Some(2),
        "temporalPressureWeight" => Some(3),
        "reserveWeight" | "overextensionWeight" | "weakBoardWeight" => Some(4),
        "travelOpportunityWeight" => Some(5),
        _ => None,
    }
}

pub fn piece_values_for(settings: &Heuristics) -> [f64; 13] {
    std::array::from_fn(|index| {
        if VALUE_KEYS[index].is_empty() {
            0.0
        } else {
            settings.get(VALUE_KEYS[index])
        }
    })
}
pub fn piece_value(piece: i8, settings: &Heuristics) -> f64 {
    let key = VALUE_KEYS.get(kind(piece)).copied().unwrap_or("");
    if key.is_empty() {
        0.0
    } else {
        settings.get(key)
    }
}

fn corridor_mask(piece_type: usize, color: usize) -> u8 {
    let rays = move_vecs((piece_type * 2) as i8);
    let mut mask = 0;
    for df in -1..=1 {
        if rays.contains(&[0, -1, if color == 0 { -1 } else { 1 }, -df]) {
            mask |= 1 << (df + 1);
        }
    }
    mask
}

struct Resources {
    available: [i32; 2],
    reserve: f64,
    overextension: f64,
}
fn resources(board: &BoardHistory, settings: &Heuristics) -> Resources {
    let mut spent = [0; 2];
    for (l, timeline) in board.iter().enumerate().skip(1) {
        if timeline.is_some() {
            spent[l % 2] = l.div_ceil(2) as i32;
        }
    }
    let available = std::array::from_fn(|color| (1 + spent[1 - color] - spent[color]).max(0));
    let excess: [i32; 2] =
        std::array::from_fn(|color| (spent[color] - spent[1 - color] - 1).max(0));
    Resources {
        available,
        reserve: 90.0
            * (available[0].min(4) - available[1].min(4)) as f64
            * settings.get("reserveWeight"),
        overextension: 45.0
            * (excess[1].min(3) - excess[0].min(3)) as f64
            * settings.get("overextensionWeight"),
    }
}

#[derive(Clone)]
struct Target {
    r: i32,
    f: i32,
    color: usize,
    pawn: bool,
    defenders: usize,
    importance: f64,
}
impl Target {
    fn king(r: i32, f: i32, color: usize) -> Self {
        Self {
            r,
            f,
            color,
            pawn: false,
            defenders: 0,
            importance: 1.5,
        }
    }
}
#[derive(Clone)]
struct Participant {
    l: usize,
    line: i32,
    t: i32,
    r: i32,
    f: i32,
    piece_type: usize,
    color: usize,
    weight: f64,
}

fn temporal_attack(
    board: &BoardHistory,
    attacker: &Participant,
    target: &Participant,
    even: bool,
    source_time: i32,
) -> bool {
    if (source_time - target.t) % 2 != 0 {
        return false;
    }
    let dl = target.line - attacker.line;
    let dt = (target.t - source_time) / 2;
    let dr = target.r - attacker.r;
    let df = target.f - attacker.f;
    if matches!(attacker.piece_type, 1 | 8) {
        let forward = if attacker.color == 0 { 1 } else { -1 };
        if dl == -forward && dt.abs() == 1 && dr == 0 && df == 0 {
            return true;
        }
        return attacker.piece_type == 8
            && ((dl == -forward
                && dt == 0
                && ((dr == 0 && df.abs() == 1) || (dr == forward && df == 0)))
                || (dl == 0 && dt == -1 && dr == forward && df == 0));
    }
    let distance = dl.abs().max(dt.abs()).max(dr.abs()).max(df.abs());
    let piece = (attacker.piece_type * 2) as i8;
    if distance <= 2 && move_pos(piece).contains(&[dl, dt, dr, df]) {
        return true;
    }
    if distance == 0
        || [dl, dt, dr, df]
            .iter()
            .any(|v| *v != 0 && v.abs() != distance)
    {
        return false;
    }
    let [sl, st, sr, sf] = [dl.signum(), dt.signum(), dr.signum(), df.signum()];
    if !move_vecs(piece).contains(&[sl, st, sr, sf]) {
        return false;
    }
    for offset in 1..distance {
        let line = timeline_move(attacker.l as i32, sl * offset, even);
        if history_square(
            board,
            line,
            source_time + st * offset * 2,
            attacker.r + sr * offset,
            attacker.f + sf * offset,
        ) != Some(0)
        {
            return false;
        }
    }
    true
}

fn pawn_defenders(squares: &Board, r: i32, f: i32, color: usize) -> usize {
    let mut defenders = 0;
    for [dr, df] in DIRECTIONS {
        let mut distance = 1;
        while let Some(piece) = square(squares, r + dr * distance, f + df * distance) {
            if piece == 0 {
                distance += 1;
                continue;
            }
            let piece_type = kind(piece);
            if owner(piece) == color && !royal(piece_type) {
                if matches!(piece_type, 1 | 8) {
                    if distance == 1 && dr == if color == 0 { -1 } else { 1 } && df != 0 {
                        defenders += 1;
                    }
                } else if move_vecs((piece_type * 2) as i8).contains(&[0, 0, -dr, -df])
                    || (distance <= 2
                        && move_pos((piece_type * 2) as i8).contains(&[
                            0,
                            0,
                            -dr * distance,
                            -df * distance,
                        ]))
                {
                    defenders += 1;
                }
            }
            break;
        }
    }
    for [dr, df] in KNIGHT_STEPS {
        if let Some(piece) = square(squares, r + dr, f + df) {
            if piece != 0 && owner(piece) == color && kind(piece) == 3 {
                defenders += 1;
            }
        }
    }
    defenders
}

fn board_kings(squares: &Board) -> Vec<Target> {
    let mut kings = vec![];
    for (r, row) in squares.iter().enumerate() {
        for (f, &piece) in row.iter().enumerate() {
            if royal(kind(piece)) {
                kings.push(Target::king(r as i32, f as i32, owner(piece)));
            }
        }
    }
    kings
}
fn king_zone(squares: &Board, kings: &[Target]) -> Vec<Target> {
    let shared = kings.len() > 2 || (kings.len() == 2 && kings[0].color == kings[1].color);
    let mut targets: Vec<Target> = vec![];
    let mut append = |target: Target| {
        if shared {
            if let Some(existing) = targets
                .iter_mut()
                .find(|existing| existing.r == target.r && existing.f == target.f)
            {
                *existing = target;
                return;
            }
        }
        targets.push(target);
    };
    for king in kings {
        append(king.clone());
        for [dr, df] in DIRECTIONS {
            let r = king.r + dr;
            let f = king.f + df;
            let Some(piece) = square(squares, r, f) else {
                continue;
            };
            if piece == 0 || owner(piece) != king.color || !matches!(kind(piece), 1 | 8) {
                continue;
            }
            let defenders = pawn_defenders(squares, r, f, king.color);
            append(Target {
                r,
                f,
                color: king.color,
                pawn: true,
                defenders,
                importance: 1.0 / (1.0 + 0.5 * defenders as f64),
            });
        }
    }
    targets
}

fn corridor_risk(
    timeline: &Timeline,
    latest: usize,
    target: &Target,
    enemy_corridors: u8,
    target_time: usize,
) -> f64 {
    let forward = if target.color == 0 { 1 } else { -1 };
    let mut risk = 0.0;
    for df in -1..=1 {
        if enemy_corridors & (1 << (df + 1)) == 0 {
            continue;
        }
        let mut open = 0;
        let mut enemy_entry = false;
        for distance in 1..=6 {
            let t = (target_time + distance as usize * 2).min(latest);
            let Some(squares) = timeline[t].as_ref() else {
                break;
            };
            let Some(piece) = square(
                squares,
                target.r + forward * distance,
                target.f + df * distance,
            ) else {
                break;
            };
            if piece != 0 && owner(piece) == target.color {
                break;
            }
            open += 1;
            if piece != 0 {
                enemy_entry = corridor_mask(kind(piece), target.color) & (1 << (df + 1)) != 0;
                break;
            }
        }
        let exposure = if enemy_entry {
            1.0
        } else {
            ((open as f64 - 1.0).max(0.0) / 3.0).min(1.0)
        };
        risk += 24.0 * exposure * target.importance;
    }
    risk
}

struct HistoryEvaluation {
    zone_risk: [f64; 2],
    royals: Vec<(usize, Target)>,
    pawns: Vec<(usize, Target)>,
}
fn timeline_evaluation(
    timeline: &Timeline,
    spatial: &SpatialEvaluation,
    settings: &Heuristics,
) -> HistoryEvaluation {
    let t = timeline.len() - 1;
    let mut result = HistoryEvaluation {
        zone_risk: [0.0; 2],
        royals: vec![],
        pawns: vec![],
    };
    let mut entry_times = vec![t];
    let mut first_times = [None; 2];
    for (parity, first_time) in first_times.iter_mut().enumerate() {
        for first in (parity..=t).step_by(2) {
            if timeline[first].is_some() {
                *first_time = Some(first);
                if first != t {
                    entry_times.push(first);
                }
                break;
            }
        }
    }
    for sampled in 0..6 {
        let Some(past) = t.checked_sub(2 + sampled * 2) else {
            break;
        };
        if !first_times.contains(&Some(past)) {
            entry_times.push(past);
        }
    }
    let corridor_samples = entry_times.len();
    for sampled in 0..6 {
        let Some(past) = t.checked_sub(1 + sampled * 2) else {
            break;
        };
        if !first_times.contains(&Some(past)) {
            entry_times.push(past);
        }
    }
    for (index, past) in entry_times.into_iter().enumerate() {
        let Some(snapshot) = timeline[past].as_ref() else {
            continue;
        };
        let historical_kings;
        let kings = if past == t {
            &spatial.kings
        } else {
            historical_kings = board_kings(snapshot);
            &historical_kings
        };
        if past < t && (t - past).is_multiple_of(2) {
            result
                .royals
                .extend(kings.iter().cloned().map(|king| (past, king)));
        }
        let mut risk = [0.0; 2];
        for target in king_zone(snapshot, kings) {
            if index < corridor_samples {
                risk[target.color] += corridor_risk(
                    timeline,
                    t,
                    &target,
                    spatial.enemy_corridors[target.color],
                    past,
                );
            }
            if past < t && target.pawn && target.defenders == 0 {
                result.pawns.push((past, target));
            }
        }
        for (color, danger) in risk.into_iter().enumerate() {
            result.zone_risk[color] = result.zone_risk[color]
                .max(danger * spatial.middle_game * settings.get("corridorWeight"));
        }
    }
    result
}

fn spatial_activity(board: &Board, r: i32, f: i32, piece_type: usize, color: usize) -> f64 {
    let steps: &[[i32; 2]] = match piece_type {
        3 => &KNIGHT_STEPS,
        2 => &DIAGONALS,
        4 => &AXES,
        5 | 6 | 7 | 9 | 10 => &DIRECTIONS,
        _ => &[],
    };
    let slider = matches!(piece_type, 2 | 4 | 5 | 7 | 10);
    let mut count = 0.0;
    for &[dr, df] in steps {
        let (mut y, mut x) = (r + dr, f + df);
        while let Some(target) = square(board, y, x) {
            if target == 0 || owner(target) != color {
                count += 1.0;
            }
            if target != 0 || !slider {
                break;
            }
            y += dr;
            x += df;
        }
    }
    count
}

struct SpatialEvaluation {
    pieces: Vec<(i32, i32, i8)>,
    kings: Vec<Target>,
    enemy_corridors: [u8; 2],
    material: f64,
    activity: f64,
    king_safety: f64,
    middle_game: f64,
    royal_risk: [f64; 2],
    features: Features,
}
fn spatial_evaluation(squares: &Board, settings: &Heuristics, inspect: bool) -> SpatialEvaluation {
    let piece_values = piece_values_for(settings);
    let mut result = SpatialEvaluation {
        pieces: vec![],
        kings: vec![],
        enemy_corridors: [0; 2],
        material: 0.0,
        activity: 0.0,
        king_safety: 0.0,
        middle_game: 0.0,
        royal_risk: [-1.0; 2],
        features: inspect.then(HashMap::new),
    };
    let width = squares.iter().map(Vec::len).max().unwrap_or(0);
    let mut pawn_files = [vec![0; width], vec![0; width]];
    let mut pawn_ranks = [vec![None; width], vec![None; width]];
    let mut phase = 0.0;
    for (r, row) in squares.iter().enumerate() {
        for (f, &piece) in row.iter().enumerate() {
            if piece == 0 {
                continue;
            }
            let piece_type = kind(piece);
            let color = owner(piece);
            let signed = sign(color);
            result.pieces.push((r as i32, f as i32, piece));
            let value = piece_values.get(piece_type).copied().unwrap_or(0.0);
            result.material += signed * value;
            if let Some(&key) = VALUE_KEYS.get(piece_type).filter(|key| !key.is_empty()) {
                record(&mut result.features, key, signed * value);
            }
            if matches!(piece_type, 2 | 3 | 4 | 5 | 7) {
                phase += value;
            }
            if matches!(piece_type, 1 | 8) {
                pawn_files[color][f] += 1;
                if color == 1 || pawn_ranks[color][f].is_none() {
                    pawn_ranks[color][f] = Some(r as i32);
                }
            }
            result.enemy_corridors[1 - color] |= corridor_mask(piece_type, 1 - color);
            if royal(piece_type) {
                result.kings.push(Target::king(r as i32, f as i32, color));
            }
        }
    }
    result.middle_game = (phase / settings.get("phaseDivisor")).min(1.0);
    for &(r, f, piece) in &result.pieces {
        let piece_type = kind(piece);
        let color = owner(piece);
        let signed = sign(color);
        let rank = if color == 0 {
            r
        } else {
            squares.len() as i32 - 1 - r
        } as f64;
        let center = (squares.len() as f64 - 1.0) / 2.0;
        let centrality = (4.0
            - ((r as f64 - center).abs()
                + (f as f64 - (squares[r as usize].len() as f64 - 1.0) / 2.0).abs())
                / 2.0)
            .max(0.0);
        if matches!(piece_type, 1 | 8) {
            let advance = (rank * 7.0 + (rank - 3.0).max(0.0).powi(2) * 7.0)
                * settings.get("pawnAdvanceWeight");
            let center_score = centrality * 3.0 * settings.get("pawnCenterWeight");
            result.activity += signed * (advance + center_score);
            record(&mut result.features, "pawnAdvanceWeight", signed * advance);
            record(
                &mut result.features,
                "pawnCenterWeight",
                signed * center_score,
            );
            if pawn_files[color][f as usize] > 1 {
                let penalty = -signed * 9.0 * settings.get("doubledPawnWeight");
                result.activity += penalty;
                record(&mut result.features, "doubledPawnWeight", penalty);
            }
            let file_count = |file: i32| {
                if file < 0 {
                    0
                } else {
                    pawn_files[color].get(file as usize).copied().unwrap_or(0)
                }
            };
            if file_count(f - 1) == 0 && file_count(f + 1) == 0 {
                let penalty = -signed * 9.0 * settings.get("isolatedPawnWeight");
                result.activity += penalty;
                record(&mut result.features, "isolatedPawnWeight", penalty);
            }
            let blocked = (f - 1..=f + 1)
                .filter(|file| *file >= 0)
                .filter_map(|file| pawn_ranks[1 - color].get(file as usize).copied().flatten())
                .any(|enemy_rank| {
                    if color == 0 {
                        enemy_rank > r
                    } else {
                        enemy_rank < r
                    }
                });
            if !blocked {
                let passed = signed * (8.0 + rank * rank * 2.0) * settings.get("passedPawnWeight");
                result.activity += passed;
                record(&mut result.features, "passedPawnWeight", passed);
            }
        } else if !royal(piece_type) {
            let mobility = spatial_activity(squares, r, f, piece_type, color);
            let (central_weight, central_key) = match piece_type {
                3 => (11.0, "knightCenterWeight"),
                2 => (6.0, "bishopCenterWeight"),
                _ => (3.0, "pieceCenterWeight"),
            };
            let mobility_score =
                mobility * if piece_type == 5 { 2.0 } else { 4.0 } * settings.get("mobilityWeight");
            let center_score = centrality * central_weight * settings.get(central_key);
            result.activity += signed * (mobility_score + center_score);
            record(
                &mut result.features,
                "mobilityWeight",
                signed * mobility_score,
            );
            record(&mut result.features, central_key, signed * center_score);
            if piece < 0 && matches!(piece_type, 2 | 3) {
                let development =
                    signed * 12.0 * result.middle_game * settings.get("developmentWeight");
                result.activity -= development;
                record(&mut result.features, "developmentWeight", -development);
            }
            if piece_type == 4 && pawn_files[color][f as usize] == 0 {
                let bonus = signed * 14.0 * settings.get("rookFileWeight");
                result.activity += bonus;
                record(&mut result.features, "rookFileWeight", bonus);
            }
        } else {
            let forward = if color == 0 { 1 } else { -1 };
            let shield = (-1..=1)
                .filter_map(|df| square(squares, r + forward, f + df))
                .filter(|piece| {
                    *piece != 0 && owner(*piece) == color && matches!(kind(*piece), 1 | 8)
                })
                .count();
            let nearby_enemies = result
                .pieces
                .iter()
                .filter(|&&(er, ef, ep)| {
                    owner(ep) != color && kind(ep) != 1 && (er - r).abs().max((ef - f).abs()) <= 3
                })
                .count();
            let shelter =
                (3 - shield) as f64 * 12.0 * result.middle_game * settings.get("shelterWeight");
            let nearby = nearby_enemies as f64 * 8.0 * settings.get("nearbyEnemyWeight");
            let flank = if rank == 0.0 && (f <= 2 || f >= squares[r as usize].len() as i32 - 2) {
                15.0
            } else {
                0.0
            } * settings.get("flankSafetyWeight");
            let risk = (shelter + nearby - flank).max(0.0);
            result.king_safety -= signed * risk;
            let royal_center = signed
                * centrality
                * 10.0
                * (1.0 - result.middle_game)
                * settings.get("royalCenterWeight");
            result.activity += royal_center;
            record(&mut result.features, "shelterWeight", -signed * shelter);
            record(&mut result.features, "nearbyEnemyWeight", -signed * nearby);
            record(
                &mut result.features,
                "flankSafetyWeight",
                signed * flank.min(shelter + nearby),
            );
            record(&mut result.features, "royalCenterWeight", royal_center);
            result.royal_risk[color] = result.royal_risk[color].max(risk);
        }
    }
    result
}

struct Evaluation {
    totals: [f64; 6],
    features: Features,
    boards: Vec<Value>,
}
impl Evaluation {
    fn total(&self) -> f64 {
        js_round(self.totals.iter().sum())
    }
    fn report(self) -> Value {
        let mut result = serde_json::Map::new();
        for (name, score) in COMPONENTS.iter().zip(self.totals) {
            result.insert((*name).into(), json!(js_round(score)));
        }
        result.insert("total".into(), json!(self.total()));
        if let Some(features) = self.features {
            result.insert("features".into(), Value::Array(schema().iter().filter_map(|entry| {
                feature_component(&entry.key).map(|component| json!({
                    "key": entry.key, "label": entry.label, "description": entry.description,
                    "component": COMPONENTS[component], "value": features.get(entry.key.as_str()).copied().unwrap_or(0.0),
                }))
            }).collect()));
            result.insert("boards".into(), Value::Array(self.boards));
        }
        Value::Object(result)
    }
}
// JavaScript Math.round sends half-centipawn ties towards positive infinity.
fn js_round(value: f64) -> f64 {
    (value + 0.5).floor()
}

fn evaluate_position(position: &Position, settings: &Heuristics, inspect: bool) -> Evaluation {
    let board = &position.board;
    let piece_values = piece_values_for(settings);
    let active = active_timelines(board);
    let even = is_even_timeline(board);
    let resources = resources(board, settings);
    let mut result = Evaluation {
        totals: [0.0; 6],
        features: inspect.then(HashMap::new),
        boards: vec![],
    };
    let mut royals: [Vec<Participant>; 4] = std::array::from_fn(|_| vec![]);
    let mut entry_pawns: [Vec<Participant>; 4] = std::array::from_fn(|_| vec![]);
    let mut attackers = vec![];
    let mut total_weight = 0.0;
    let mut active_count = 0;
    let mut lowest_material = f64::INFINITY;
    let mut highest_material = f64::NEG_INFINITY;
    let mut worst_king = [0.0_f64; 2];
    for (l, timeline) in board.iter().enumerate() {
        let Some(timeline) = timeline.as_ref().filter(|timeline| !timeline.is_empty()) else {
            continue;
        };
        let t = timeline.len() - 1;
        let Some(squares) = timeline[t].as_ref() else {
            continue;
        };
        let is_active = active.contains(&l);
        let weight = if is_active {
            1.0
        } else {
            settings.get("inactiveWeight")
        };
        let line = coordinate(l, even);
        total_weight += weight;
        let spatial = spatial_evaluation(squares, settings, inspect);
        if let Some(features) = &spatial.features {
            for (&key, &value) in features {
                record(&mut result.features, key, weight * value);
            }
        }
        for &(r, f, piece) in &spatial.pieces {
            let piece_type = kind(piece);
            let color = owner(piece);
            if piece_type == 1 {
                let target_line = timeline_index(line + if color == 0 { -1 } else { 1 }, even);
                if board.get(target_line).and_then(Option::as_ref).is_none() {
                    continue;
                }
            }
            let participant = Participant {
                l,
                line,
                t: t as i32,
                r,
                f,
                piece_type,
                color,
                weight,
            };
            if royal(piece_type) {
                royals[color * 2 + t % 2].push(participant.clone());
            }
            attackers.push(participant);
        }
        let history = timeline_evaluation(timeline, &spatial, settings);
        for (past, king) in history.royals {
            royals[king.color * 2 + past % 2].push(Participant {
                l,
                line,
                t: past as i32,
                r: king.r,
                f: king.f,
                piece_type: 6,
                color: king.color,
                weight: weight * settings.get("historicalPressureWeight"),
            });
        }
        for (past, pawn) in history.pawns {
            if resources.available[1 - pawn.color] <= 0 {
                continue;
            }
            entry_pawns[pawn.color * 2 + past % 2].push(Participant {
                l,
                line,
                t: past as i32,
                r: pawn.r,
                f: pawn.f,
                piece_type: 1,
                color: pawn.color,
                weight,
            });
        }
        if is_active {
            for (color, worst) in worst_king.iter_mut().enumerate() {
                if spatial.royal_risk[color] >= 0.0 {
                    *worst = worst.max(spatial.royal_risk[color] + history.zone_risk[color]);
                }
            }
        }
        let corridor = history.zone_risk[1] - history.zone_risk[0];
        let king_safety = spatial.king_safety + corridor;
        record(&mut result.features, "corridorWeight", weight * corridor);
        result.totals[0] += weight * spatial.material;
        result.totals[1] += weight * spatial.activity;
        result.totals[2] += weight * king_safety;
        if is_active {
            active_count += 1;
            lowest_material = lowest_material.min(spatial.material);
            highest_material = highest_material.max(spatial.material);
        }
        if inspect {
            result.boards.push(json!({ "timeline": l, "coordinate": line, "turn": t, "active": is_active,
                "weight": weight, "material": weight * spatial.material, "activity": weight * spatial.activity,
                "kingSafety": weight * king_safety, "phase": spatial.middle_game }));
        }
    }
    if total_weight != 0.0 {
        for total in &mut result.totals[..3] {
            *total /= total_weight;
        }
        if let Some(features) = &mut result.features {
            for value in features.values_mut() {
                *value /= total_weight;
            }
        }
        for entry in &mut result.boards {
            for key in &COMPONENTS[..3] {
                entry[*key] = json!(entry[*key].as_f64().unwrap() / total_weight);
            }
        }
        let worst = (worst_king[1] - worst_king[0]) * 0.45 * settings.get("worstKingWeight");
        result.totals[2] += worst;
        record(&mut result.features, "worstKingWeight", worst);
        let mut pressure = [0.0; 2];
        let mut travel = [0.0_f64; 2];
        for attacker in &attackers {
            let mut best = 0.0_f64;
            let enemy_group = (1 - attacker.color) * 2;
            let max_pressure = 20.0 * attacker.weight * settings.get("temporalPressureWeight");
            for king in &royals[enemy_group + attacker.t as usize % 2] {
                if best >= max_pressure {
                    break;
                }
                if king.l == attacker.l && king.t == attacker.t {
                    continue;
                }
                if temporal_attack(board, attacker, king, even, attacker.t) {
                    best = best.max(
                        20.0 * attacker.weight.min(king.weight)
                            * settings.get("temporalPressureWeight"),
                    );
                }
            }
            pressure[attacker.color] += best;
            if resources.available[attacker.color] == 0 || royal(attacker.piece_type) {
                continue;
            }
            let ready = attacker.t as usize % 2 == attacker.color;
            let source_time = attacker.t + i32::from(!ready);
            let base_travel = 140.0
                * if ready { 1.0 } else { 0.5 }
                * (piece_values[attacker.piece_type] / 340.0).min(1.0);
            let max_travel =
                base_travel * attacker.weight * settings.get("travelOpportunityWeight");
            for target in &entry_pawns[enemy_group + source_time as usize % 2] {
                if travel[attacker.color] >= max_travel {
                    break;
                }
                if !temporal_attack(board, attacker, target, even, source_time) {
                    continue;
                }
                let value = base_travel
                    * attacker.weight.min(target.weight)
                    * settings.get("travelOpportunityWeight");
                travel[attacker.color] = travel[attacker.color].max(value);
            }
        }
        result.totals[3] = (pressure[0] - pressure[1]).clamp(-100.0, 100.0);
        result.totals[5] = travel[0] - travel[1];
        result.totals[4] = resources.reserve + resources.overextension;
        record(
            &mut result.features,
            "temporalPressureWeight",
            result.totals[3],
        );
        record(
            &mut result.features,
            "travelOpportunityWeight",
            result.totals[5],
        );
        record(&mut result.features, "reserveWeight", resources.reserve);
        record(
            &mut result.features,
            "overextensionWeight",
            resources.overextension,
        );
        if active_count > 1 {
            let weak = (lowest_material.min(0.0) + highest_material.max(0.0))
                * 0.12
                * settings.get("weakBoardWeight");
            result.totals[4] += weak;
            record(&mut result.features, "weakBoardWeight", weak);
        }
    }
    for (index, total) in result.totals.iter_mut().enumerate() {
        *total *= settings.get(COMPONENT_WEIGHTS[index]);
    }
    if let Some(features) = &mut result.features {
        for (key, value) in features {
            *value *= settings.get(COMPONENT_WEIGHTS[feature_component(key).unwrap()]);
        }
    }
    for entry in &mut result.boards {
        for index in 0..3 {
            entry[COMPONENTS[index]] = json!(
                entry[COMPONENTS[index]].as_f64().unwrap() * settings.get(COMPONENT_WEIGHTS[index])
            );
        }
    }
    result
}

/// Positive centipawns favor White; mate scores belong to the search layer.
pub fn evaluate(position: &Position, settings: &Heuristics) -> f64 {
    evaluate_position(position, settings, false).total()
}
pub fn evaluate_detailed(position: &Position, settings: &Heuristics) -> Value {
    evaluate_position(position, settings, false).report()
}
pub fn inspect_evaluation(position: &Position, settings: &Heuristics) -> Value {
    evaluate_position(position, settings, true).report()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn javascript_rounding_is_preserved() {
        assert_eq!(js_round(-1.5), -1.0);
        assert_eq!(js_round(1.5), 2.0);
    }
    #[test]
    fn custom_piece_values_apply_to_signed_unmoved_pieces() {
        let settings = Heuristics::from_value(Some(&json!({"queenValue": 1500}))).unwrap();
        assert_eq!(piece_value(-9, &settings), 1500.0);
        assert_eq!(piece_value(12, &settings), 0.0);
    }
}
