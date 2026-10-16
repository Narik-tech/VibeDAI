use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub type Board = Vec<Vec<i8>>;
pub type Timeline = Vec<Option<Arc<Board>>>;
pub type BoardHistory = Vec<Option<Timeline>>;
pub type Move = Vec<Vec<i32>>;
pub type Action = Vec<Move>;

/// The wire format deliberately matches 5d-chess-js, including sparse history,
/// unmoved-piece signs, and promotion/en-passant/castling auxiliary coordinates.
/// Arc shares immutable past boards when a child appends a new frontier.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Position {
    pub board: BoardHistory,
    pub action: usize,
    #[serde(default)]
    pub promotions: Vec<i8>,
}

impl Position {
    pub fn validate(&self) -> Result<(), String> {
        if self.board.len() > 131 || self.action > 1_000_000 {
            return Err("Position exceeds supported timeline/action limits".into());
        }
        let mut present = false;
        for timeline in self.board.iter().flatten() {
            if timeline.len() > 8192 {
                return Err("Position history exceeds 8192 half-turns".into());
            }
            for board in timeline.iter().flatten() {
                present = true;
                if board.is_empty()
                    || board.len() > 16
                    || board[0].is_empty()
                    || board[0].len() > 16
                    || board.iter().any(|row| {
                        row.len() != board[0].len() || row.iter().any(|p| !(-24..=24).contains(p))
                    })
                {
                    return Err("Malformed board: expected a rectangular 1–16 square board with valid pieces".into());
                }
            }
        }
        if !present {
            return Err("Position contains no boards".into());
        }
        if self.promotions.iter().any(|p| !(1..=24).contains(p)) {
            return Err("Invalid promotion piece".into());
        }
        Ok(())
    }

    /// Collision-free full-history key. The length prefixes distinguish missing
    /// boards from empty timelines and retain unmoved flags and promotion rules.
    pub fn key(&self) -> Vec<u8> {
        let mut key = Vec::new();
        key.push((self.action % 2) as u8);
        key.extend_from_slice(&(self.promotions.len() as u32).to_le_bytes());
        key.extend(self.promotions.iter().map(|&p| p as u8));
        key.extend_from_slice(&(self.board.len() as u32).to_le_bytes());
        for timeline in &self.board {
            match timeline {
                None => key.push(0),
                Some(timeline) => {
                    key.push(1);
                    key.extend_from_slice(&(timeline.len() as u32).to_le_bytes());
                    for board in timeline {
                        match board {
                            None => key.push(0),
                            Some(board) => {
                                key.push(1);
                                key.push(board.len() as u8);
                                key.push(board.first().map_or(0, Vec::len) as u8);
                                for row in board.iter() {
                                    key.extend(row.iter().map(|&p| p as u8));
                                }
                            }
                        }
                    }
                }
            }
        }
        key
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn position() -> Position {
        Position {
            board: vec![Some(vec![Some(Arc::new(vec![vec![12, -2], vec![0, 11]]))])],
            action: 0,
            promotions: vec![10, 9, 6, 5],
        }
    }

    #[test]
    fn exact_keys_retain_history_holes_unmoved_flags_and_promotions() {
        let initial = position();
        let mut moved = initial.clone();
        moved.board[0].as_mut().unwrap()[0] = Some(Arc::new(vec![vec![12, 2], vec![0, 11]]));
        assert_ne!(initial.key(), moved.key());
        let mut history = initial.clone();
        history.board[0].as_mut().unwrap().insert(0, None);
        assert_ne!(initial.key(), history.key());
        let mut promotion = initial.clone();
        promotion.promotions.reverse();
        assert_ne!(initial.key(), promotion.key());
        let mut parity = initial.clone();
        parity.action = 2;
        assert_eq!(initial.key(), parity.key());
        parity.action = 1;
        assert_ne!(initial.key(), parity.key());
    }

    #[test]
    fn wire_validation_rejects_malformed_boards_and_preserves_shared_history() {
        let initial = position();
        initial.validate().unwrap();
        let decoded: Position =
            serde_json::from_str(&serde_json::to_string(&initial).unwrap()).unwrap();
        assert_eq!(initial, decoded);
        let mut invalid = initial.clone();
        invalid.board[0].as_mut().unwrap()[0] = Some(Arc::new(vec![vec![12], vec![0, 11]]));
        assert!(invalid.validate().is_err());
        assert_eq!(
            initial.board[0].as_ref().unwrap()[0].as_ref().unwrap()[0],
            vec![12, -2]
        );
    }
}
