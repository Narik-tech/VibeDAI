//! The classical engine's validated, tunable evaluation and search profile.
//! Keep this schema aligned with src/heuristics.js at the JavaScript boundary.
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, sync::OnceLock};

#[derive(Debug, Deserialize, Serialize)]
pub struct Setting {
    pub key: String,
    pub label: String,
    pub description: String,
    pub group: String,
    #[serde(rename = "default")]
    pub default_value: f64,
    pub min: f64,
    pub max: f64,
    pub step: f64,
    pub unit: String,
}

pub fn schema() -> &'static [Setting] {
    static SETTINGS: OnceLock<Vec<Setting>> = OnceLock::new();
    SETTINGS
        .get_or_init(|| serde_json::from_str(SCHEMA_JSON).expect("valid embedded heuristic schema"))
}

#[derive(Clone, Debug)]
pub struct Heuristics {
    values: HashMap<&'static str, f64>,
}

impl Default for Heuristics {
    fn default() -> Self {
        Self {
            values: schema()
                .iter()
                .map(|setting| (setting.key.as_str(), setting.default_value))
                .collect(),
        }
    }
}

impl Heuristics {
    pub fn from_value(value: Option<&Value>) -> Result<Self, String> {
        let mut result = Self::default();
        let Some(value) = value else {
            return Ok(result);
        };
        let object = value
            .as_object()
            .ok_or("Heuristics must be a plain object of numeric settings.")?;
        for (key, value) in object {
            let entry = schema()
                .iter()
                .find(|setting| setting.key == *key)
                .ok_or_else(|| format!("Unknown heuristic setting: {key}."))?;
            let number = value
                .as_f64()
                .filter(|number| number.is_finite())
                .ok_or_else(|| format!("{key} must be a finite number."))?;
            if number < entry.min || number > entry.max {
                return Err(format!(
                    "{key} must be between {} and {}.",
                    entry.min, entry.max
                ));
            }
            if key == "quiescenceDepth" && number.fract() != 0.0 {
                return Err("quiescenceDepth must be a whole number of turns.".into());
            }
            result.values.insert(entry.key.as_str(), number);
        }
        Ok(result)
    }

    #[inline]
    pub fn get(&self, key: &str) -> f64 {
        *self.values.get(key).expect("known heuristic setting")
    }

    pub fn to_value(&self) -> Value {
        serde_json::to_value(&self.values).expect("finite heuristic settings")
    }
}

const SCHEMA_JSON: &str = r#"[
  {
    "key": "materialWeight",
    "label": "Material",
    "description": "Weighted average of material on timeline frontiers. Historical copies never add material.",
    "group": "Components",
    "default": 1,
    "min": 0,
    "max": 4,
    "step": 0.05,
    "unit": "×"
  },
  {
    "key": "activityWeight",
    "label": "Activity",
    "description": "Weighted average of development, mobility, centralization, and pawn structure.",
    "group": "Components",
    "default": 1,
    "min": 0,
    "max": 4,
    "step": 0.05,
    "unit": "×"
  },
  {
    "key": "kingSafetyWeight",
    "label": "King safety",
    "description": "Frontier shelter, persistent historical corridors, and the weakest active royal.",
    "group": "Components",
    "default": 1,
    "min": 0,
    "max": 4,
    "step": 0.05,
    "unit": "×"
  },
  {
    "key": "temporalWeight",
    "label": "Temporal pressure",
    "description": "Potential attacks on enemy royals across boards and time, capped before this multiplier.",
    "group": "Components",
    "default": 1,
    "min": 0,
    "max": 4,
    "step": 0.05,
    "unit": "×"
  },
  {
    "key": "timelinesWeight",
    "label": "Timeline resources",
    "description": "Branch reserves, overextension, and material weaknesses across active boards.",
    "group": "Components",
    "default": 1,
    "min": 0,
    "max": 4,
    "step": 0.05,
    "unit": "×"
  },
  {
    "key": "travelWeight",
    "label": "Historical travel",
    "description": "The best historical capture route to an undefended pawn beside an enemy royal.",
    "group": "Components",
    "default": 1,
    "min": 0,
    "max": 4,
    "step": 0.05,
    "unit": "×"
  },
  {
    "key": "pawnValue",
    "label": "Pawn",
    "description": "Material value of one pawn. Also affects capture ordering and historical travel discounts.",
    "group": "Piece values",
    "default": 100,
    "min": 0,
    "max": 5000,
    "step": 5,
    "unit": "cp"
  },
  {
    "key": "bishopValue",
    "label": "Bishop",
    "description": "Material value of one bishop. Also affects capture ordering and historical travel discounts. Counts toward middlegame material phase.",
    "group": "Piece values",
    "default": 355,
    "min": 0,
    "max": 5000,
    "step": 5,
    "unit": "cp"
  },
  {
    "key": "knightValue",
    "label": "Knight",
    "description": "Material value of one knight. Also affects capture ordering and historical travel discounts. Counts toward middlegame material phase.",
    "group": "Piece values",
    "default": 340,
    "min": 0,
    "max": 5000,
    "step": 5,
    "unit": "cp"
  },
  {
    "key": "rookValue",
    "label": "Rook",
    "description": "Material value of one rook. Also affects capture ordering and historical travel discounts. Counts toward middlegame material phase.",
    "group": "Piece values",
    "default": 550,
    "min": 0,
    "max": 5000,
    "step": 5,
    "unit": "cp"
  },
  {
    "key": "queenValue",
    "label": "Queen",
    "description": "Material value of one queen. Also affects capture ordering and historical travel discounts. Counts toward middlegame material phase.",
    "group": "Piece values",
    "default": 1150,
    "min": 0,
    "max": 5000,
    "step": 5,
    "unit": "cp"
  },
  {
    "key": "princessValue",
    "label": "Princess",
    "description": "Material value of one princess. Also affects capture ordering and historical travel discounts. Counts toward middlegame material phase.",
    "group": "Piece values",
    "default": 900,
    "min": 0,
    "max": 5000,
    "step": 5,
    "unit": "cp"
  },
  {
    "key": "brawnValue",
    "label": "Brawn",
    "description": "Material value of one brawn. Also affects capture ordering and historical travel discounts.",
    "group": "Piece values",
    "default": 140,
    "min": 0,
    "max": 5000,
    "step": 5,
    "unit": "cp"
  },
  {
    "key": "commonKingValue",
    "label": "Common king",
    "description": "Material value of one common king. Also affects capture ordering and historical travel discounts.",
    "group": "Piece values",
    "default": 370,
    "min": 0,
    "max": 5000,
    "step": 5,
    "unit": "cp"
  },
  {
    "key": "unicornValue",
    "label": "Unicorn",
    "description": "Material value of one unicorn. Also affects capture ordering and historical travel discounts.",
    "group": "Piece values",
    "default": 450,
    "min": 0,
    "max": 5000,
    "step": 5,
    "unit": "cp"
  },
  {
    "key": "dragonValue",
    "label": "Dragon",
    "description": "Material value of one dragon. Also affects capture ordering and historical travel discounts.",
    "group": "Piece values",
    "default": 350,
    "min": 0,
    "max": 5000,
    "step": 5,
    "unit": "cp"
  },
  {
    "key": "phaseDivisor",
    "label": "Middlegame material scale",
    "description": "Middlegame factor is min(1, combined bishop, knight, rook, queen, and princess material divided by this value). Scales development, shelter, corridors, and royal centralization.",
    "group": "Position weighting",
    "default": 6000,
    "min": 100,
    "max": 30000,
    "step": 100,
    "unit": "cp"
  },
  {
    "key": "inactiveWeight",
    "label": "Inactive timeline weight",
    "description": "Active frontier boards have weight 1. This is the weight of inactive frontiers in averaged evaluation and temporal opportunities.",
    "group": "Position weighting",
    "default": 0.2,
    "min": 0,
    "max": 1,
    "step": 0.05,
    "unit": "×"
  },
  {
    "key": "historicalPressureWeight",
    "label": "Historical royal weight",
    "description": "Historical royal targets receive this fraction of their timeline weight when estimating direct temporal pressure.",
    "group": "Position weighting",
    "default": 0.65,
    "min": 0,
    "max": 1,
    "step": 0.05,
    "unit": "×"
  },
  {
    "key": "pawnAdvanceWeight",
    "label": "Pawn advancement",
    "description": "Pawn and brawn advancement: 7r + 7 × max(0, r − 3)², with r measured from the home rank.",
    "group": "Pawn structure",
    "default": 1,
    "min": 0,
    "max": 4,
    "step": 0.05,
    "unit": "×"
  },
  {
    "key": "pawnCenterWeight",
    "label": "Pawn centralization",
    "description": "Pawn and brawn centrality contributes 3 cp per unit; centrality follows actual board dimensions.",
    "group": "Pawn structure",
    "default": 1,
    "min": 0,
    "max": 4,
    "step": 0.05,
    "unit": "×"
  },
  {
    "key": "doubledPawnWeight",
    "label": "Doubled pawns",
    "description": "Each pawn or brawn sharing its file with another friendly pawn or brawn loses 9 cp.",
    "group": "Pawn structure",
    "default": 1,
    "min": 0,
    "max": 4,
    "step": 0.05,
    "unit": "×"
  },
  {
    "key": "isolatedPawnWeight",
    "label": "Isolated pawns",
    "description": "Each pawn or brawn without a friendly pawn or brawn on an adjacent file loses 9 cp.",
    "group": "Pawn structure",
    "default": 1,
    "min": 0,
    "max": 4,
    "step": 0.05,
    "unit": "×"
  },
  {
    "key": "passedPawnWeight",
    "label": "Passed pawns",
    "description": "A pawn or brawn with no enemy pawn or brawn ahead on the same or adjacent file earns 8 + 2r² cp.",
    "group": "Pawn structure",
    "default": 1,
    "min": 0,
    "max": 4,
    "step": 0.05,
    "unit": "×"
  },
  {
    "key": "knightCenterWeight",
    "label": "Knight centralization",
    "description": "Knights earn 11 cp per unit of board-aware centrality.",
    "group": "Activity",
    "default": 1,
    "min": 0,
    "max": 4,
    "step": 0.05,
    "unit": "×"
  },
  {
    "key": "bishopCenterWeight",
    "label": "Bishop centralization",
    "description": "Bishops earn 6 cp per unit of board-aware centrality.",
    "group": "Activity",
    "default": 1,
    "min": 0,
    "max": 4,
    "step": 0.05,
    "unit": "×"
  },
  {
    "key": "pieceCenterWeight",
    "label": "Other piece centralization",
    "description": "Other nonroyal, non-pawn pieces earn 3 cp per unit of board-aware centrality.",
    "group": "Activity",
    "default": 1,
    "min": 0,
    "max": 4,
    "step": 0.05,
    "unit": "×"
  },
  {
    "key": "royalCenterWeight",
    "label": "Endgame royal centralization",
    "description": "Royals earn 10 × (1 − middlegame factor) cp per centrality unit.",
    "group": "Activity",
    "default": 1,
    "min": 0,
    "max": 4,
    "step": 0.05,
    "unit": "×"
  },
  {
    "key": "mobilityWeight",
    "label": "Spatial mobility",
    "description": "Queens earn 2 cp per reachable square; other nonroyals earn 4. Uses ordinary board-plane reach without testing destination safety.",
    "group": "Activity",
    "default": 1,
    "min": 0,
    "max": 4,
    "step": 0.05,
    "unit": "×"
  },
  {
    "key": "developmentWeight",
    "label": "Minor-piece development",
    "description": "Unmoved bishops and knights lose 12 × middlegame factor cp.",
    "group": "Activity",
    "default": 1,
    "min": 0,
    "max": 4,
    "step": 0.05,
    "unit": "×"
  },
  {
    "key": "rookFileWeight",
    "label": "Rook files",
    "description": "Rooks earn 14 cp on files without a friendly pawn or brawn; enemy pawns do not remove the bonus.",
    "group": "Activity",
    "default": 1,
    "min": 0,
    "max": 4,
    "step": 0.05,
    "unit": "×"
  },
  {
    "key": "shelterWeight",
    "label": "Royal pawn shield",
    "description": "Each missing forward shield pawn or brawn contributes 12 × middlegame factor cp of royal danger.",
    "group": "King safety",
    "default": 1,
    "min": 0,
    "max": 4,
    "step": 0.05,
    "unit": "×"
  },
  {
    "key": "nearbyEnemyWeight",
    "label": "Nearby royal threats",
    "description": "Each enemy non-pawn within three squares contributes 8 cp of royal danger. Brawns count as threats.",
    "group": "King safety",
    "default": 1,
    "min": 0,
    "max": 4,
    "step": 0.05,
    "unit": "×"
  },
  {
    "key": "flankSafetyWeight",
    "label": "Home-rank flank shelter",
    "description": "A royal near a home-rank flank reduces local danger by 15 cp. Local danger cannot fall below zero.",
    "group": "King safety",
    "default": 1,
    "min": 0,
    "max": 4,
    "step": 0.05,
    "unit": "×"
  },
  {
    "key": "corridorWeight",
    "label": "Historical king-zone corridors",
    "description": "Time/rank and time/rank/file approaches add up to 24 cp per route, scaled by target importance and phase. Uses the worst sampled historical exposure.",
    "group": "King safety",
    "default": 1,
    "min": 0,
    "max": 4,
    "step": 0.05,
    "unit": "×"
  },
  {
    "key": "worstKingWeight",
    "label": "Weakest active royal",
    "description": "Adds 45% of each side’s worst active royal danger, including local shelter and historical corridors, outside the frontier average.",
    "group": "King safety",
    "default": 1,
    "min": 0,
    "max": 4,
    "step": 0.05,
    "unit": "×"
  },
  {
    "key": "temporalPressureWeight",
    "label": "Direct temporal attacks",
    "description": "Each frontier attacker contributes its best cross-board royal attack, up to 20 cp times board weight. The White-minus-Black pressure is capped at ±100 cp before component weighting.",
    "group": "Multiverse",
    "default": 1,
    "min": 0,
    "max": 4,
    "step": 0.05,
    "unit": "×"
  },
  {
    "key": "reserveWeight",
    "label": "Branch reserves",
    "description": "Each unused active branch reserve contributes 90 cp, up to four per side. The first unmatched branch changes the reserve advantage by 180 cp.",
    "group": "Multiverse",
    "default": 1,
    "min": 0,
    "max": 4,
    "step": 0.05,
    "unit": "×"
  },
  {
    "key": "overextensionWeight",
    "label": "Branch overextension",
    "description": "Branches beyond the opponent’s extent plus one cost 45 cp each, capped at three. Sparse and inactive timelines still consume capacity.",
    "group": "Multiverse",
    "default": 1,
    "min": 0,
    "max": 4,
    "step": 0.05,
    "unit": "×"
  },
  {
    "key": "travelOpportunityWeight",
    "label": "Historical pawn entry",
    "description": "Best historical route per side to a king-adjacent pawn with no nonroyal spatial defender: up to 140 cp, discounted for low-value attackers and setups awaiting a reply. Requires branch reserve.",
    "group": "Multiverse",
    "default": 1,
    "min": 0,
    "max": 4,
    "step": 0.05,
    "unit": "×"
  },
  {
    "key": "weakBoardWeight",
    "label": "Material weakness across boards",
    "description": "With multiple active boards, adds 12% of the lowest negative material balance plus the highest positive balance to preserve extreme imbalances.",
    "group": "Multiverse",
    "default": 1,
    "min": 0,
    "max": 4,
    "step": 0.05,
    "unit": "×"
  },
  {
    "key": "quiescenceDepth",
    "label": "Tactical extension depth",
    "description": "Maximum extra complete turns containing a capture or promotion. Checked horizons also search an evasion turn.",
    "group": "Search",
    "default": 2,
    "min": 0,
    "max": 8,
    "step": 1,
    "unit": "turns"
  },
  {
    "key": "aspirationWindow",
    "label": "Aspiration window",
    "description": "From depth two, initially search this far above and below the previous score; retry failures with a full window. Zero disables aspiration windows.",
    "group": "Search",
    "default": 60,
    "min": 0,
    "max": 500,
    "step": 5,
    "unit": "cp"
  },
  {
    "key": "temporalMovePenalty",
    "label": "Temporal move ordering penalty",
    "description": "Subtract these ordering points from cross-board moves when spatial moves are preferred. Changes ordering, not evaluation or legality.",
    "group": "Search",
    "default": 2000000,
    "min": 0,
    "max": 5000000,
    "step": 10000,
    "unit": "points"
  },
  {
    "key": "quietCentralization",
    "label": "Quiet centralization ordering",
    "description": "Ordering points per improvement in distance to the fixed 8×8 center (3.5, 3.5). Does not change evaluation.",
    "group": "Search",
    "default": 10,
    "min": 0,
    "max": 100,
    "step": 1,
    "unit": "points"
  },
  {
    "key": "killerBonus",
    "label": "Quiet killer priority",
    "description": "Ordering bonus for quiet turns that previously caused a cutoff; the search remembers two per ply.",
    "group": "Search",
    "default": 100000,
    "min": 0,
    "max": 500000,
    "step": 1000,
    "unit": "points"
  },
  {
    "key": "historyBonus",
    "label": "Quiet history learning",
    "description": "Quiet cutoff history increases by this coefficient × remaining depth², capped at 50,000 points.",
    "group": "Search",
    "default": 20,
    "min": 0,
    "max": 200,
    "step": 1,
    "unit": "points"
  }
]"#;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn validates_profiles_without_changing_defaults() {
        assert_eq!(Heuristics::default().get("knightValue"), 340.0);
        assert_eq!(
            Heuristics::from_value(Some(&json!({"pawnValue": 250})))
                .unwrap()
                .get("pawnValue"),
            250.0
        );
        assert!(Heuristics::from_value(Some(&json!({"unknown": 1}))).is_err());
        assert!(Heuristics::from_value(Some(&json!({"phaseDivisor": 0}))).is_err());
        assert!(Heuristics::from_value(Some(&json!({"quiescenceDepth": 1.5}))).is_err());
        assert!(Heuristics::from_value(Some(&Value::Null)).is_err());
    }
}
