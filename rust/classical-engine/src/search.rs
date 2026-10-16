//! Complete-turn PVS, iterative deepening and bounded tactical continuation.
//! Kairos is a design reference for the separation of lazy action traversal,
//! native search and protocol. This port retains this project's search policy.
use crate::evaluate::evaluate;
use crate::heuristics::Heuristics;
use crate::model::{Action, Move, Position};
use crate::rules::{self, ActionGenerator, Candidate};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    mpsc, Arc,
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

pub const MATE_SCORE: f64 = 100_000.0;
const MATE_THRESHOLD: f64 = MATE_SCORE - 1000.0;
const INF: f64 = 1_000_000.0;
type SearchResult<T> = Result<T, &'static str>;
type Key = (Arc<Vec<u8>>, i32);

fn option(options: &Value, name: &str, fallback: f64, min: f64, max: f64) -> f64 {
    options
        .get(name)
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite())
        .unwrap_or(fallback)
        .clamp(min, max)
}
fn sign(pos: &Position) -> f64 {
    if pos.action.is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}
fn to_table(score: f64, ply: usize) -> f64 {
    if score > MATE_THRESHOLD {
        score + ply as f64
    } else if score < -MATE_THRESHOLD {
        score - ply as f64
    } else {
        score
    }
}
fn from_table(score: f64, ply: usize) -> f64 {
    if score > MATE_THRESHOLD {
        score - ply as f64
    } else if score < -MATE_THRESHOLD {
        score + ply as f64
    } else {
        score
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Bound {
    Exact,
    Lower,
    Upper,
}
#[derive(Clone)]
struct Entry {
    depth: i32,
    horizon: i32,
    score: f64,
    bound: Bound,
    pv: Vec<Action>,
    bytes: usize,
}
struct Table {
    entries: HashMap<Key, Entry>,
    order: VecDeque<Key>,
    bytes: usize,
    max_bytes: usize,
    max_entries: usize,
}
impl Table {
    fn store(&mut self, key: Key, mut entry: Entry) {
        if self.max_entries == 0 || self.max_bytes == 0 {
            return;
        }
        entry.bytes = 192
            + key.0.len()
            + entry
                .pv
                .iter()
                .map(|a| {
                    24 + a
                        .iter()
                        .map(|m| 24 + m.iter().map(|c| 24 + c.len() * 4).sum::<usize>())
                        .sum::<usize>()
                })
                .sum::<usize>();
        if entry.bytes > self.max_bytes {
            return;
        }
        let mut queued = false;
        if let Some(previous) = self.entries.remove(&key) {
            self.bytes -= previous.bytes;
            queued = true;
        }
        while self.bytes + entry.bytes > self.max_bytes || self.entries.len() >= self.max_entries {
            if let Some(old) = self.order.pop_front() {
                if old == key {
                    queued = false;
                }
                if let Some(value) = self.entries.remove(&old) {
                    self.bytes -= value.bytes;
                }
            } else {
                break;
            }
        }
        // An updated key may itself have been the oldest eviction candidate.
        if !queued {
            self.order.push_back(key.clone());
        }
        self.bytes += entry.bytes;
        self.entries.insert(key, entry);
    }
}

#[derive(Default)]
struct SharedBudget {
    nodes: AtomicU64,
    search_nodes: AtomicU64,
    generation_nodes: AtomicU64,
    qnodes: AtomicU64,
    stop_iteration: AtomicBool,
}

struct Budget {
    started: Instant,
    deadline: Option<Instant>,
    cancelled: Arc<AtomicBool>,
    max_nodes: u64,
    nodes: u64,
    search_nodes: u64,
    generation_nodes: u64,
    qnodes: u64,
    shared: Option<Arc<SharedBudget>>,
}
impl Budget {
    fn tick(&mut self, kind: u8) -> SearchResult<()> {
        if self.cancelled.load(Ordering::Relaxed) {
            return Err("cancelled");
        }
        if self.deadline.is_some_and(|d| Instant::now() >= d) {
            return Err("time");
        }
        if let Some(shared) = &self.shared {
            if shared.stop_iteration.load(Ordering::Relaxed) {
                return Err("iteration");
            }
            let mut used = shared.nodes.load(Ordering::Relaxed);
            loop {
                if used >= self.max_nodes {
                    return Err("nodes");
                }
                match shared.nodes.compare_exchange_weak(
                    used,
                    used + 1,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(actual) => used = actual,
                }
            }
            if kind == 0 {
                shared.generation_nodes.fetch_add(1, Ordering::Relaxed);
            } else {
                shared.search_nodes.fetch_add(1, Ordering::Relaxed);
                if kind == 2 {
                    shared.qnodes.fetch_add(1, Ordering::Relaxed);
                }
            }
        } else if self.nodes >= self.max_nodes {
            return Err("nodes");
        }
        self.nodes += 1;
        if kind == 0 {
            self.generation_nodes += 1;
        } else {
            self.search_nodes += 1;
            if kind == 2 {
                self.qnodes += 1;
            }
        }
        Ok(())
    }
}

#[derive(Clone)]
struct Line {
    score: f64,
    pv: Vec<Action>,
    policy: bool,
}
impl Line {
    fn leaf(score: f64) -> Self {
        Self {
            score,
            pv: vec![],
            policy: false,
        }
    }
}

enum WorkerJob {
    Search {
        candidate: Candidate,
        remaining: i32,
        horizon: i32,
        alpha: f64,
        beta: f64,
    },
    Shutdown,
}

#[derive(Default)]
struct WorkerStats {
    tt_hits: u64,
    qtt_hits: u64,
    cutoffs: u64,
    policy_leaves: u64,
    entries: usize,
    bytes: usize,
    selective_depth: usize,
}

struct WorkerReply {
    result: SearchResult<Line>,
    stats: WorkerStats,
}
struct RootWorker {
    jobs: mpsc::Sender<WorkerJob>,
    replies: mpsc::Receiver<WorkerReply>,
    handle: Option<JoinHandle<()>>,
    pending: Option<Action>,
    used: bool,
    stats: WorkerStats,
}

pub struct Search<'a> {
    root: Position,
    options: Value,
    heuristics: Heuristics,
    values: [f64; 13],
    budget: Budget,
    table: Table,
    killers: HashMap<usize, Vec<Action>>,
    history: HashMap<Vec<i32>, f64>,
    qdepth: i32,
    max_depth: usize,
    active_qdepth: i32,
    completed_qdepth: i32,
    depth: usize,
    searching_depth: usize,
    selective_depth: usize,
    root_actions: usize,
    best: Option<Action>,
    score: Option<f64>,
    pv: Vec<Action>,
    root_partial: Option<Line>,
    completed: bool,
    status: &'static str,
    stopped: Option<&'static str>,
    tt_hits: u64,
    qtt_hits: u64,
    cutoffs: u64,
    policy_leaves: u64,
    mate_probe: Option<Value>,
    workers: Vec<RootWorker>,
    threads_used: usize,
    cache_bytes: usize,
    cache_entries: usize,
    progress: &'a mut dyn FnMut(Value),
    last_progress: Instant,
}

fn history_key(pos: &Position, mv: &Move) -> Vec<i32> {
    let from = &mv[0];
    let to = &mv[1];
    vec![
        rules::piece_at(&pos.board, from).unwrap_or(0).abs() as i32,
        if from[0] == to[0] { -1 } else { from[0] },
        if from[0] == to[0] { -1 } else { to[0] },
        to[1] - from[1],
        from[2],
        from[3],
        to[2],
        to[3],
        *to.get(4).unwrap_or(&0),
    ]
}

fn order_moves(
    pos: &Position,
    moves: &mut Vec<Move>,
    preferred: &[Move],
    killers: &[Move],
    values: &[f64; 13],
    heuristics: &Heuristics,
    history: &HashMap<Vec<i32>, f64>,
) {
    let priority = |mv: &Move| {
        let from = &mv[0];
        let to = &mv[1];
        let captured = rules::piece_at(&pos.board, to).unwrap_or(0).unsigned_abs() as usize;
        let capture = captured != 0 || mv.len() == 3;
        let mover = rules::piece_at(&pos.board, from)
            .unwrap_or(0)
            .unsigned_abs() as usize;
        let mover_value = values[mover.div_ceil(2)];
        let promotion = to
            .get(4)
            .map_or(0.0, |p| values[(*p as usize).div_ceil(2)] - mover_value);
        let mut score = if preferred.contains(mv) {
            10_000_000.0
        } else {
            0.0
        };
        score += promotion * 100.0;
        if capture {
            score += 1_000_000.0
                + if captured != 0 {
                    values[captured.div_ceil(2)] * 100.0
                } else {
                    values[1] * 100.0
                }
                - mover_value;
        } else {
            if killers.contains(mv) {
                score += heuristics.get("killerBonus");
            }
            score += history.get(&history_key(pos, mv)).copied().unwrap_or(0.0);
            score += ((from[2] as f64 - 3.5).abs() + (from[3] as f64 - 3.5).abs()
                - (to[2] as f64 - 3.5).abs()
                - (to[3] as f64 - 3.5).abs())
                * heuristics.get("quietCentralization");
        }
        if from[0] != to[0] || from[1] != to[1] {
            score -= heuristics.get("temporalMovePenalty");
        }
        score
    };
    let mut ranked: Vec<_> = std::mem::take(moves)
        .into_iter()
        .map(|m| (priority(&m), m))
        .collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    moves.extend(ranked.into_iter().map(|(_, m)| m));
}

impl<'a> Search<'a> {
    pub fn new(
        root: Position,
        options: Value,
        cancelled: Arc<AtomicBool>,
        progress: &'a mut dyn FnMut(Value),
    ) -> Result<Self, String> {
        let configured = Heuristics::from_value(options.get("heuristics"))?;
        let mut hv = configured.to_value();
        if options.get("quiescenceDepth").is_some() {
            hv["quiescenceDepth"] =
                json!(option(&options, "quiescenceDepth", 2.0, 0.0, 8.0).floor());
        }
        let heuristics = Heuristics::from_value(Some(&hv))?;
        let mut values = [0.0; 13];
        for (i, name) in [
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
        ]
        .iter()
        .enumerate()
        {
            if !name.is_empty() {
                values[i] = heuristics.get(name);
            }
        }
        let time_ms = option(&options, "timeMs", 3000.0, 0.0, 3_600_000.0);
        let started = Instant::now();
        let deadline = if time_ms == 0.0 || options["unlimitedTime"] == true {
            None
        } else {
            Some(started + Duration::from_secs_f64(time_ms / 1000.0))
        };
        let max_nodes = option(&options, "maxNodes", 2_000_000.0, 0.0, 1_000_000_000.0) as u64;
        let max_entries = option(&options, "maxTableEntries", 100_000.0, 0.0, 1_000_000.0) as usize;
        let max_bytes =
            (option(&options, "cacheMemoryMb", 128.0, 0.0, 4096.0) * 1024.0 * 1024.0) as usize;
        let threads = option(&options, "threads", 1.0, 1.0, 16.0) as usize;
        Ok(Self {
            max_depth: option(&options, "maxDepth", 8.0, 1.0, 64.0) as usize,
            qdepth: heuristics.get("quiescenceDepth") as i32,
            root,
            options,
            heuristics,
            values,
            budget: Budget {
                started,
                deadline,
                cancelled,
                max_nodes,
                nodes: 0,
                search_nodes: 0,
                generation_nodes: 0,
                qnodes: 0,
                shared: if threads > 1 {
                    Some(Arc::new(SharedBudget::default()))
                } else {
                    None
                },
            },
            table: Table {
                entries: HashMap::new(),
                order: VecDeque::new(),
                bytes: 0,
                max_bytes: max_bytes / threads,
                max_entries: max_entries / threads,
            },
            killers: HashMap::new(),
            history: HashMap::new(),
            active_qdepth: 0,
            completed_qdepth: 0,
            depth: 0,
            searching_depth: 0,
            selective_depth: 0,
            root_actions: 0,
            best: None,
            score: None,
            pv: vec![],
            root_partial: None,
            completed: false,
            status: "incomplete",
            stopped: None,
            tt_hits: 0,
            qtt_hits: 0,
            cutoffs: 0,
            policy_leaves: 0,
            mate_probe: None,
            workers: vec![],
            threads_used: 1,
            cache_bytes: max_bytes,
            cache_entries: max_entries,
            progress,
            last_progress: started,
        })
    }

    fn static_score(&self, pos: &Position) -> f64 {
        (evaluate(pos, &self.heuristics) * sign(pos))
            .clamp(-MATE_THRESHOLD + 1.0, MATE_THRESHOLD - 1.0)
    }
    fn terminal(&self, pos: &Position, ply: usize) -> f64 {
        if rules::in_check(pos) {
            -MATE_SCORE + ply as f64
        } else {
            0.0
        }
    }

    fn generator(
        &mut self,
        pos: &Position,
        mut preferred: Option<Action>,
        restricted: bool,
        tactical: bool,
    ) -> SearchResult<ActionGenerator> {
        if preferred.is_none()
            && !tactical
            && pos.board.len() > 1
            && rules::present_timelines(pos).len() > 1
            && rules::in_check(pos)
        {
            self.budget.tick(0)?;
            for mv in rules::pseudo_moves(pos) {
                if mv[0][0] == mv[1][0] && mv[0][1] == mv[1][1] {
                    continue;
                }
                self.budget.tick(0)?;
                let next = rules::apply_move(pos, &mv);
                if rules::can_submit(&next) {
                    preferred = Some(vec![mv]);
                    break;
                }
            }
        }
        Ok(ActionGenerator::new(pos, preferred, restricted, tactical))
    }

    fn next(
        &mut self,
        iter: &mut ActionGenerator,
        preferred: &[Move],
        ply: usize,
    ) -> SearchResult<Option<Candidate>> {
        let killers: Vec<Move> = self
            .killers
            .get(&ply)
            .into_iter()
            .flatten()
            .flatten()
            .cloned()
            .collect();
        let budget = &mut self.budget;
        let values = &self.values;
        let heuristics = &self.heuristics;
        let history = &self.history;
        iter.next(&mut || budget.tick(0), &mut |pos, moves| {
            order_moves(pos, moves, preferred, &killers, values, heuristics, history)
        })
    }

    fn empty(&mut self, pos: &Position, ply: usize, pruned: bool) -> SearchResult<Line> {
        if pruned {
            let mut full = self.generator(pos, None, false, false)?;
            if self.next(&mut full, &[], ply)?.is_some() {
                self.policy_leaves += 1;
                return Ok(Line {
                    score: self.static_score(pos),
                    pv: vec![],
                    policy: true,
                });
            }
        }
        Ok(Line::leaf(self.terminal(pos, ply)))
    }

    // The original alpha/beta window and root-relative ply are required to
    // encode the bound and mate distance accurately in the transposition table.
    #[allow(clippy::too_many_arguments)]
    fn store(
        &mut self,
        key: Key,
        depth: i32,
        score: f64,
        alpha: f64,
        beta: f64,
        pv: Vec<Action>,
        ply: usize,
    ) {
        let bound = if score <= alpha {
            Bound::Upper
        } else if score >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        self.table.store(
            key,
            Entry {
                depth,
                horizon: self.active_qdepth,
                score: to_table(score, ply),
                bound,
                pv,
                bytes: 0,
            },
        );
    }

    fn quiescence(
        &mut self,
        pos: &Position,
        mut alpha: f64,
        mut beta: f64,
        remaining: i32,
        ply: usize,
    ) -> SearchResult<Line> {
        self.selective_depth = self.selective_depth.max(ply);
        self.budget.tick(2)?;
        alpha = alpha.max(-MATE_SCORE + ply as f64);
        beta = beta.min(MATE_SCORE - ply as f64 - 1.0);
        if alpha >= beta {
            return Ok(Line::leaf(alpha));
        }
        let key = (Arc::new(pos.key()), remaining + 2);
        let entry = self.table.entries.get(&key).cloned();
        if let Some(e) = &entry {
            self.tt_hits += 1;
            self.qtt_hits += 1;
            let score = from_table(e.score, ply);
            match e.bound {
                Bound::Exact => {
                    return Ok(Line {
                        score,
                        pv: e.pv.clone(),
                        policy: false,
                    })
                }
                Bound::Lower => alpha = alpha.max(score),
                Bound::Upper => beta = beta.min(score),
            }
            if alpha >= beta {
                return Ok(Line {
                    score,
                    pv: e.pv.clone(),
                    policy: false,
                });
            }
        }
        let search_alpha = alpha;
        let search_beta = beta;
        let checked = remaining >= 0 && rules::in_check(pos);
        let mut best = if checked || remaining < 0 {
            -INF
        } else {
            self.static_score(pos)
        };
        let tactical = remaining > 0 && !checked && best < beta;
        let preferred = entry
            .as_ref()
            .and_then(|e| e.pv.first())
            .cloned()
            .unwrap_or_default();
        let mut iter = self.generator(
            pos,
            if preferred.is_empty() {
                None
            } else {
                Some(preferred.clone())
            },
            true,
            tactical,
        )?;
        let mut next = self.next(&mut iter, &preferred, ply)?;
        if next.is_none() && tactical {
            iter = self.generator(pos, None, true, false)?;
            if self.next(&mut iter, &[], ply)?.is_none() {
                best = self.empty(pos, ply, iter.policy_pruned)?.score;
            }
            self.store(key, remaining, best, -INF, INF, vec![], ply);
            return Ok(Line::leaf(best));
        }
        if next.is_none() {
            let result = self.empty(pos, ply, iter.policy_pruned)?;
            self.store(key, remaining, result.score, -INF, INF, vec![], ply);
            return Ok(result);
        }
        // Establish a legal witness before returning static evaluation, including
        // after the one additional evasion allowed at a checked horizon.
        if remaining < 0 || (!checked && (best >= beta || remaining == 0)) {
            if remaining < 0 {
                best = self.static_score(pos);
            }
            self.store(
                key,
                remaining,
                best,
                if remaining <= 0 { -INF } else { search_alpha },
                if remaining <= 0 { INF } else { search_beta },
                vec![],
                ply,
            );
            return Ok(Line::leaf(best));
        }
        if !checked {
            alpha = alpha.max(best);
        }
        let mut pv = vec![];
        while let Some(candidate) = next {
            let child =
                self.quiescence(&candidate.position, -beta, -alpha, remaining - 1, ply + 1)?;
            let value = -child.score;
            if value > best {
                best = value;
                pv = vec![candidate.moves.clone()];
                pv.extend(child.pv);
            }
            alpha = alpha.max(value);
            if alpha >= beta {
                self.cutoffs += 1;
                break;
            }
            next = self.next(&mut iter, &preferred, ply)?;
        }
        self.store(
            key,
            remaining,
            best,
            search_alpha,
            search_beta,
            pv.clone(),
            ply,
        );
        Ok(Line {
            score: best,
            pv,
            policy: false,
        })
    }

    fn remember_cutoff(&mut self, pos: &Position, action: &Action, ply: usize, depth: i32) {
        self.cutoffs += 1;
        if action.iter().any(|m| rules::is_tactical_move(pos, m)) {
            return;
        }
        let killers = self.killers.entry(ply).or_default();
        killers.retain(|a| a != action);
        killers.insert(0, action.clone());
        killers.truncate(2);
        for mv in action {
            let value = self.history.entry(history_key(pos, mv)).or_default();
            *value = (*value + (depth * depth) as f64 * self.heuristics.get("historyBonus"))
                .min(50_000.0);
        }
    }

    fn negamax(
        &mut self,
        pos: &Position,
        remaining: i32,
        mut alpha: f64,
        mut beta: f64,
        ply: usize,
        preferred: Option<Action>,
    ) -> SearchResult<Line> {
        self.selective_depth = self.selective_depth.max(ply);
        if remaining <= 0 {
            return self.quiescence(pos, alpha, beta, self.active_qdepth, ply);
        }
        if ply == 0 {
            self.root_actions = 0;
        }
        self.budget.tick(1)?;
        alpha = alpha.max(-MATE_SCORE + ply as f64);
        beta = beta.min(MATE_SCORE - ply as f64 - 1.0);
        if alpha >= beta {
            return Ok(Line::leaf(alpha));
        }
        let key = (Arc::new(pos.key()), 0);
        let entry = self.table.entries.get(&key).cloned();
        if let Some(e) = &entry {
            if e.depth >= remaining && e.horizon >= self.active_qdepth {
                self.tt_hits += 1;
                let score = from_table(e.score, ply);
                match e.bound {
                    Bound::Exact => {
                        return Ok(Line {
                            score,
                            pv: e.pv.clone(),
                            policy: false,
                        })
                    }
                    Bound::Lower => alpha = alpha.max(score),
                    Bound::Upper => beta = beta.min(score),
                }
                if alpha >= beta {
                    return Ok(Line {
                        score,
                        pv: e.pv.clone(),
                        policy: false,
                    });
                }
            }
        }
        let search_alpha = alpha;
        let search_beta = beta;
        let preferred = preferred
            .or_else(|| entry.and_then(|e| e.pv.first().cloned()))
            .unwrap_or_default();
        let mut iter = self.generator(
            pos,
            if preferred.is_empty() {
                None
            } else {
                Some(preferred.clone())
            },
            true,
            false,
        )?;
        let mut best = -INF;
        let mut pv = vec![];
        let mut count = 0;
        while let Some(candidate) = self.next(&mut iter, &preferred, ply)? {
            let mut child = if count == 0 {
                self.negamax(
                    &candidate.position,
                    remaining - 1,
                    -beta,
                    -alpha,
                    ply + 1,
                    None,
                )?
            } else {
                self.negamax(
                    &candidate.position,
                    remaining - 1,
                    -alpha - 1.0,
                    -alpha,
                    ply + 1,
                    None,
                )?
            };
            if count > 0 && -child.score > alpha && -child.score < beta {
                child = self.negamax(
                    &candidate.position,
                    remaining - 1,
                    -beta,
                    -alpha,
                    ply + 1,
                    None,
                )?;
            }
            count += 1;
            if ply == 0 {
                self.root_actions = count;
            }
            let value = -child.score;
            if value > best {
                best = value;
                pv = vec![candidate.moves.clone()];
                pv.extend(child.pv);
                if ply == 0 {
                    self.root_partial = Some(Line {
                        score: best,
                        pv: pv.clone(),
                        policy: false,
                    });
                }
            }
            alpha = alpha.max(value);
            self.maybe_progress();
            if alpha >= beta {
                self.remember_cutoff(pos, &candidate.moves, ply, remaining);
                break;
            }
        }
        if count == 0 {
            let result = self.empty(pos, ply, iter.policy_pruned)?;
            self.store(key, remaining, result.score, -INF, INF, vec![], ply);
            return Ok(result);
        }
        self.store(
            key,
            remaining,
            best,
            search_alpha,
            search_beta,
            pv.clone(),
            ply,
        );
        Ok(Line {
            score: best,
            pv,
            policy: false,
        })
    }

    fn ensure_workers(&mut self) -> SearchResult<()> {
        if !self.workers.is_empty() {
            return Ok(());
        }
        let threads = option(&self.options, "threads", 1.0, 1.0, 16.0) as usize;
        for index in 1..threads {
            let (jobs, requests) = mpsc::channel();
            let (results, replies) = mpsc::channel();
            let root = self.root.clone();
            let mut options = self.options.clone();
            options["threads"] = json!(1);
            let cancelled = self.budget.cancelled.clone();
            let shared = self.budget.shared.clone();
            let started = self.budget.started;
            let deadline = self.budget.deadline;
            let cache_bytes = self.cache_bytes / threads;
            let cache_entries = self.cache_entries / threads;
            let handle = thread::Builder::new()
                .name(format!("classical-root-{index}"))
                .spawn(move || {
                    let mut quiet = |_: Value| {};
                    let Ok(mut search) = Search::new(root, options, cancelled, &mut quiet) else {
                        return;
                    };
                    search.budget.shared = shared;
                    search.budget.started = started;
                    search.budget.deadline = deadline;
                    search.table.max_bytes = cache_bytes;
                    search.table.max_entries = cache_entries;
                    while let Ok(job) = requests.recv() {
                        let WorkerJob::Search {
                            candidate,
                            remaining,
                            horizon,
                            alpha,
                            beta,
                        } = job
                        else {
                            break;
                        };
                        search.active_qdepth = horizon;
                        let result =
                            search.negamax(&candidate.position, remaining, -beta, -alpha, 1, None);
                        let stats = WorkerStats {
                            tt_hits: search.tt_hits,
                            qtt_hits: search.qtt_hits,
                            cutoffs: search.cutoffs,
                            policy_leaves: search.policy_leaves,
                            entries: search.table.entries.len(),
                            bytes: search.table.bytes,
                            selective_depth: search.selective_depth,
                        };
                        if results.send(WorkerReply { result, stats }).is_err() {
                            break;
                        }
                    }
                })
                .map_err(|_| "worker")?;
            self.workers.push(RootWorker {
                jobs,
                replies,
                handle: Some(handle),
                pending: None,
                used: false,
                stats: WorkerStats::default(),
            });
        }
        Ok(())
    }

    fn receive_worker(&mut self, index: usize) -> SearchResult<(Action, SearchResult<Line>)> {
        loop {
            match self.workers[index]
                .replies
                .recv_timeout(Duration::from_millis(25))
            {
                Ok(reply) => {
                    let worker = &mut self.workers[index];
                    worker.stats = reply.stats;
                    return Ok((
                        worker.pending.take().expect("pending root action"),
                        reply.result,
                    ));
                }
                Err(mpsc::RecvTimeoutError::Timeout) => self.maybe_progress(),
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    self.workers[index].pending = None;
                    return Err("worker");
                }
            }
        }
    }

    fn drain_workers(&mut self) {
        if let Some(shared) = &self.budget.shared {
            shared.stop_iteration.store(true, Ordering::Relaxed);
        }
        for index in 0..self.workers.len() {
            if self.workers[index].pending.is_some() {
                let _ = self.receive_worker(index);
            }
        }
        if let Some(shared) = &self.budget.shared {
            shared.stop_iteration.store(false, Ordering::Relaxed);
        }
    }

    fn accept_root(
        &mut self,
        action: Action,
        child: Line,
        best: &mut f64,
        pv: &mut Vec<Action>,
        alpha: &mut f64,
    ) {
        self.root_actions += 1;
        let score = -child.score;
        if score > *best {
            *best = score;
            *pv = vec![action];
            pv.extend(child.pv);
            self.root_partial = Some(Line {
                score,
                pv: pv.clone(),
                policy: false,
            });
        }
        *alpha = alpha.max(score);
        self.maybe_progress();
    }

    /// Split complete actions at the root, preserving lazy generation. Every
    /// bounded batch uses the same established alpha and is accepted in source
    /// order, so thread scheduling cannot decide between tied root moves.
    fn parallel_root(
        &mut self,
        pos: &Position,
        remaining: i32,
        mut alpha: f64,
        mut beta: f64,
        preferred: Option<Action>,
    ) -> SearchResult<Line> {
        self.root_actions = 0;
        self.budget.tick(1)?;
        alpha = alpha.max(-MATE_SCORE);
        beta = beta.min(MATE_SCORE - 1.0);
        if alpha >= beta {
            return Ok(Line::leaf(alpha));
        }
        let original_alpha = alpha;
        let original_beta = beta;
        let preferred = preferred.unwrap_or_default();
        let mut iterator = self.generator(
            pos,
            if preferred.is_empty() {
                None
            } else {
                Some(preferred.clone())
            },
            true,
            false,
        )?;
        let Some(first) = self.next(&mut iterator, &preferred, 0)? else {
            return self.empty(pos, 0, iterator.policy_pruned);
        };
        let first_child = self.negamax(&first.position, remaining - 1, -beta, -alpha, 1, None)?;
        let mut best = -INF;
        let mut pv = vec![];
        self.accept_root(first.moves, first_child, &mut best, &mut pv, &mut alpha);
        let outcome = (|| -> SearchResult<()> {
            while alpha < beta {
                let Some(first) = self.next(&mut iterator, &preferred, 0)? else {
                    break;
                };
                self.ensure_workers()?;
                let batch_alpha = alpha;
                let mut candidate = Some(first);
                let mut exhausted = false;
                for index in 0..self.workers.len() {
                    let next = if index == 0 {
                        candidate.take()
                    } else {
                        self.next(&mut iterator, &preferred, 0)?
                    };
                    let Some(next) = next else {
                        exhausted = true;
                        break;
                    };
                    let action = next.moves.clone();
                    let worker = &mut self.workers[index];
                    worker
                        .jobs
                        .send(WorkerJob::Search {
                            candidate: next,
                            remaining: remaining - 1,
                            horizon: self.active_qdepth,
                            alpha: batch_alpha,
                            beta,
                        })
                        .map_err(|_| "worker")?;
                    worker.pending = Some(action);
                    if !worker.used {
                        worker.used = true;
                        self.threads_used += 1;
                    }
                }
                // The coordinator searches one sibling while the other native
                // threads work. There is at most one queued job per worker.
                let local = if exhausted {
                    None
                } else {
                    self.next(&mut iterator, &preferred, 0)?
                };
                let local = if let Some(candidate) = local {
                    let child = self.negamax(
                        &candidate.position,
                        remaining - 1,
                        -beta,
                        -batch_alpha,
                        1,
                        None,
                    )?;
                    Some((candidate.moves, child))
                } else {
                    exhausted = true;
                    None
                };
                for index in 0..self.workers.len() {
                    if self.workers[index].pending.is_none() {
                        continue;
                    }
                    let (action, result) = self.receive_worker(index)?;
                    self.accept_root(action, result?, &mut best, &mut pv, &mut alpha);
                    if alpha >= beta {
                        return Ok(());
                    }
                }
                if let Some((action, child)) = local {
                    self.accept_root(action, child, &mut best, &mut pv, &mut alpha);
                }
                if exhausted {
                    break;
                }
            }
            Ok(())
        })();
        // A cutoff or interruption cancels outstanding jobs only for this root
        // window. Drain before another iteration reuses the workers and caches.
        self.drain_workers();
        outcome?;
        self.store(
            (Arc::new(pos.key()), 0),
            remaining,
            best,
            original_alpha,
            original_beta,
            pv.clone(),
            0,
        );
        Ok(Line {
            score: best,
            pv,
            policy: false,
        })
    }

    fn root_iteration(
        &mut self,
        pos: &Position,
        remaining: i32,
        alpha: f64,
        beta: f64,
        preferred: Option<Action>,
    ) -> SearchResult<Line> {
        // A serial depth-one warmup secures a legal completed result before
        // worker creation consumes any time from a short analysis.
        if remaining > 1 && self.budget.shared.is_some() {
            self.parallel_root(pos, remaining, alpha, beta, preferred)
        } else {
            self.negamax(pos, remaining, alpha, beta, 0, preferred)
        }
    }

    fn maybe_progress(&mut self) {
        if self.best.is_some() && self.last_progress.elapsed() >= Duration::from_millis(250) {
            self.last_progress = Instant::now();
            let result = self.snapshot();
            (self.progress)(result);
        }
    }

    fn mate_probe(&mut self, pos: &Position, max_work: u64) -> SearchResult<Option<Action>> {
        let start = self.budget.nodes;
        let mut candidates = 0;
        let mut checks = 0;
        let mut seen = HashSet::new();
        let mut skipped = false;
        let run = (|| -> SearchResult<Option<Action>> {
            for restricted in [true, false] {
                let mut iter = ActionGenerator::new(pos, self.best.clone(), restricted, false);
                loop {
                    let budget = &mut self.budget;
                    let values = &self.values;
                    let heuristics = &self.heuristics;
                    let history = &self.history;
                    let mut charge = || {
                        if budget.nodes - start >= max_work {
                            Err("probe-work")
                        } else {
                            budget.tick(0)
                        }
                    };
                    let next = iter.next(&mut charge, &mut |p, m| {
                        order_moves(p, m, &[], &[], values, heuristics, history)
                    })?;
                    let Some(candidate) = next else {
                        break;
                    };
                    if !seen.insert(candidate.position.key()) {
                        continue;
                    }
                    candidates += 1;
                    charge()?;
                    if !rules::in_check(&candidate.position) {
                        continue;
                    }
                    checks += 1;
                    // A temporal evasion may advance several required boards at
                    // once. Probe it before building unrelated combinations.
                    let mut quick_reply = false;
                    if candidate.position.board.len() > 1
                        && rules::present_timelines(&candidate.position).len() > 1
                    {
                        charge()?;
                        for mv in rules::pseudo_moves(&candidate.position) {
                            if mv[0][..2] == mv[1][..2] {
                                continue;
                            }
                            charge()?;
                            if rules::can_submit(&rules::apply_move(&candidate.position, &mv)) {
                                quick_reply = true;
                                break;
                            }
                        }
                    }
                    if quick_reply {
                        continue;
                    }
                    let mut replies = ActionGenerator::new(&candidate.position, None, false, false);
                    if replies
                        .next(&mut charge, &mut |p, m| {
                            order_moves(p, m, &[], &[], values, heuristics, history)
                        })?
                        .is_none()
                    {
                        return Ok(Some(candidate.moves));
                    }
                }
                skipped |= iter.policy_pruned;
                if !skipped {
                    break;
                }
            }
            Ok(None)
        })();
        let status = match &run {
            Ok(Some(_)) => "mate",
            Ok(None) => "no-mate",
            Err(_) => "unknown",
        };
        self.mate_probe = Some(
            json!({"status":status,"work":self.budget.nodes-start,"candidates":candidates,"checks":checks,"reason":if run==Err("probe-work"){Some("work")}else{None}}),
        );
        match run {
            Err("probe-work") => Ok(None),
            other => other,
        }
    }

    fn run(&mut self) -> SearchResult<()> {
        let root = self.root.clone();
        let mut fallback = self.generator(&root, None, true, false)?;
        let first = self.next(&mut fallback, &[], 0)?;
        if let Some(candidate) = &first {
            self.best = Some(candidate.moves.clone());
            self.pv = vec![candidate.moves.clone()];
        }
        let probe_work = option(&self.options, "mateProbeWork", 4096.0, 0.0, 1_000_000.0) as u64;
        let probe_work = probe_work.min(self.budget.max_nodes / 8);
        if (first.is_some() || fallback.policy_pruned)
            && root.board.len() > 1
            && root.board.iter().flatten().any(|l| l.len() > 2)
            && probe_work > 0
        {
            if let Some(action) = self.mate_probe(&root, probe_work)? {
                self.best = Some(action.clone());
                self.pv = vec![action];
                self.score = Some(MATE_SCORE - 1.0);
                self.depth = 1;
                self.searching_depth = 1;
                self.selective_depth = 1;
                self.completed = true;
                self.status = "ok";
                self.stopped = Some("mate");
                self.root_actions = self
                    .mate_probe
                    .as_ref()
                    .and_then(|probe| probe["candidates"].as_u64())
                    .unwrap_or(0) as usize;
                let result = self.snapshot();
                (self.progress)(result);
                return Ok(());
            }
        }
        if first.is_none() {
            let empty = self.empty(&root, 0, fallback.policy_pruned)?;
            if empty.policy {
                self.stopped = Some("policy");
            } else {
                self.score = Some(empty.score);
                self.status = if empty.score == 0.0 {
                    "stalemate"
                } else {
                    "checkmate"
                };
                self.completed = true;
            }
            return Ok(());
        }
        let mut iterations = vec![];
        if self.options["quiescenceWarmup"] != false {
            for horizon in 0..self.qdepth {
                iterations.push((1, horizon));
            }
        }
        for depth in 1..=self.max_depth {
            iterations.push((depth, self.qdepth));
        }
        for (depth, horizon) in iterations {
            self.searching_depth = depth;
            self.active_qdepth = horizon;
            self.root_partial = None;
            let window = if depth > 1
                && self.heuristics.get("aspirationWindow") > 0.0
                && self.score.unwrap_or(0.0).abs() < MATE_THRESHOLD
            {
                self.heuristics.get("aspirationWindow")
            } else {
                INF
            };
            let (lower, upper) = if window == INF {
                (-INF, INF)
            } else {
                (
                    self.score.unwrap_or(0.0) - window,
                    self.score.unwrap_or(0.0) + window,
                )
            };
            let mut result =
                self.root_iteration(&root, depth as i32, lower, upper, self.best.clone())?;
            if result.score <= lower || result.score >= upper {
                result = self.root_iteration(
                    &root,
                    depth as i32,
                    -INF,
                    INF,
                    result.pv.first().cloned().or(self.best.clone()),
                )?;
            }
            self.score = Some(result.score);
            self.depth = depth;
            self.pv = result.pv;
            if let Some(best) = self.pv.first() {
                self.best = Some(best.clone());
            }
            self.completed_qdepth = horizon;
            self.completed = true;
            self.status = "ok";
            self.last_progress = Instant::now();
            let snapshot = self.snapshot();
            (self.progress)(snapshot);
            if result.score.abs() >= MATE_SCORE - depth as f64 {
                self.stopped = Some("mate");
                break;
            }
        }
        if self.stopped.is_none() {
            self.stopped = Some("depth");
        }
        Ok(())
    }

    pub fn analyze(mut self) -> Value {
        if let Err(reason) = self.run() {
            self.stopped = Some(reason);
            if !self.completed {
                if let Some(partial) = self.root_partial.take() {
                    self.best = partial.pv.first().cloned();
                    self.score = Some(partial.score);
                    self.pv = partial.pv;
                }
            }
        }
        self.snapshot()
    }

    pub fn snapshot(&self) -> Value {
        let elapsed = self.budget.started.elapsed().as_secs_f64() * 1000.0;
        let score = self.score.map(|s| (s * sign(&self.root) + 0.5).floor());
        let mate = self.score.is_some_and(|s| s.abs() > MATE_THRESHOLD);
        // Read the subset first: concurrent quiescence work cannot make the
        // reported qnode count exceed the subsequent search-node snapshot.
        let (qnodes, search_nodes, generation_nodes) = if let Some(shared) = &self.budget.shared {
            (
                shared.qnodes.load(Ordering::Relaxed),
                shared.search_nodes.load(Ordering::Relaxed),
                shared.generation_nodes.load(Ordering::Relaxed),
            )
        } else {
            (
                self.budget.qnodes,
                self.budget.search_nodes,
                self.budget.generation_nodes,
            )
        };
        let qnodes = qnodes.min(search_nodes);
        let nodes = search_nodes + generation_nodes;
        let tt_hits = self.tt_hits + self.workers.iter().map(|w| w.stats.tt_hits).sum::<u64>();
        let qtt_hits = self.qtt_hits + self.workers.iter().map(|w| w.stats.qtt_hits).sum::<u64>();
        let cutoffs = self.cutoffs + self.workers.iter().map(|w| w.stats.cutoffs).sum::<u64>();
        let policy_leaves = self.policy_leaves
            + self
                .workers
                .iter()
                .map(|w| w.stats.policy_leaves)
                .sum::<u64>();
        let entries =
            self.table.entries.len() + self.workers.iter().map(|w| w.stats.entries).sum::<usize>();
        let bytes = self.table.bytes + self.workers.iter().map(|w| w.stats.bytes).sum::<usize>();
        let selective_depth = self
            .workers
            .iter()
            .map(|w| w.stats.selective_depth)
            .max()
            .unwrap_or(0)
            .max(self.selective_depth);
        json!({
            "backend":"rust","threadsUsed":self.threads_used,"threadsRequested":option(&self.options,"threads",1.0,1.0,16.0) as usize,
            "bestAction":self.best,"score":score,"depth":self.depth,"nodes":nodes,
            "searchNodes":search_nodes,"generationNodes":generation_nodes,"qnodes":qnodes,
            "ttHits":tt_hits,"qTtHits":qtt_hits,"cutoffs":cutoffs,"elapsedMs":elapsed.round(),
            "searchingDepth":self.searching_depth,"rootActionsSearched":self.root_actions,"selectiveDepth":selective_depth,
            "nps":if elapsed>0.0{(nodes as f64*1000.0/elapsed).round()}else{0.0},
            "pv":self.pv,"status":self.status,"completed":self.completed,"stoppedReason":self.stopped,
            "tableEntries":entries,"cacheMemoryBytes":bytes,
            "searchPolicy":if self.mate_probe.as_ref().is_some_and(|p|p["status"]=="mate"){"full-rules-mate"}else{"present-spatial"},
            "policyLeaves":policy_leaves,"heuristics":self.heuristics.to_value(),"mateProbe":self.mate_probe,
            "effectiveQuiescenceDepth":if self.completed{self.completed_qdepth}else{self.active_qdepth},
            "scoreType":if self.score.is_none(){"unavailable"}else if mate{"mate"}else{"cp"},
            "mateIn":if mate {self.score.map(|s|(s*sign(&self.root)).signum()*(MATE_SCORE-s.abs()))}else{None},
            "limits":{"timeMs":option(&self.options,"timeMs",3000.0,0.0,3_600_000.0),"maxDepth":self.max_depth,
                "maxNodes":self.budget.max_nodes,"quiescenceDepth":self.qdepth,"cacheMemoryMb":self.cache_bytes as f64/1048576.0,
                "maxTableEntries":self.cache_entries,"threads":option(&self.options,"threads",1.0,1.0,16.0) as usize,
                "mateProbeWork":(option(&self.options,"mateProbeWork",4096.0,0.0,1_000_000.0) as u64).min(self.budget.max_nodes/8)}
        })
    }
}

impl Drop for Search<'_> {
    fn drop(&mut self) {
        if self.workers.is_empty() {
            return;
        }
        if let Some(shared) = &self.budget.shared {
            shared.stop_iteration.store(true, Ordering::Relaxed);
        }
        for worker in &self.workers {
            let _ = worker.jobs.send(WorkerJob::Shutdown);
        }
        for worker in &mut self.workers {
            if let Some(handle) = worker.handle.take() {
                let _ = handle.join();
            }
        }
    }
}

#[cfg(test)]
mod parallel_tests {
    use super::*;

    #[test]
    fn concurrent_work_claims_never_exceed_the_global_budget() {
        let shared = Arc::new(SharedBudget::default());
        let cancelled = Arc::new(AtomicBool::new(false));
        thread::scope(|scope| {
            for index in 0..8 {
                let shared = shared.clone();
                let cancelled = cancelled.clone();
                scope.spawn(move || {
                    let mut budget = Budget {
                        started: Instant::now(),
                        deadline: None,
                        cancelled,
                        max_nodes: 1001,
                        nodes: 0,
                        search_nodes: 0,
                        generation_nodes: 0,
                        qnodes: 0,
                        shared: Some(shared),
                    };
                    while budget.tick(if index % 2 == 0 { 0 } else { 2 }).is_ok() {}
                });
            }
        });
        let nodes = shared.nodes.load(Ordering::Relaxed);
        let search = shared.search_nodes.load(Ordering::Relaxed);
        let generation = shared.generation_nodes.load(Ordering::Relaxed);
        assert_eq!(nodes, 1001);
        assert_eq!(nodes, search + generation);
        assert_eq!(shared.qnodes.load(Ordering::Relaxed), search);
    }
}
