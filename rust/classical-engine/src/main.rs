//! One-request NDJSON transport. Keeping stdin open permits cooperative cancel;
//! EOF means the parent went away and stops analysis without an orphan process.
use classical_engine::{
    evaluate,
    heuristics::Heuristics,
    model::{Action, Position},
    rules,
    search::Search,
};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

fn emit(value: Value) {
    let mut out = io::stdout().lock();
    if serde_json::to_writer(&mut out, &value).is_err()
        || writeln!(out).is_err()
        || out.flush().is_err()
    {
        std::process::exit(0);
    }
}

fn execute(request: Value) -> Result<Value, String> {
    let kind = request
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("analyze");
    let position: Position =
        serde_json::from_value(request.get("position").cloned().ok_or("Missing position")?)
            .map_err(|e| format!("Invalid position: {e}"))?;
    position.validate()?;
    let options = request.get("options").cloned().unwrap_or_else(|| json!({}));
    match kind {
        "analyze" => {
            let cancelled = Arc::new(AtomicBool::new(false));
            let signal = cancelled.clone();
            std::thread::spawn(move || {
                for line in io::stdin().lock().lines() {
                    match line {
                        Ok(line) => {
                            if serde_json::from_str::<Value>(&line)
                                .ok()
                                .is_some_and(|v| v["type"] == "cancel")
                            {
                                signal.store(true, Ordering::Relaxed);
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
                signal.store(true, Ordering::Relaxed);
            });
            let mut progress = |result| emit(json!({"type":"progress","result":result}));
            let result = Search::new(position, options, cancelled, &mut progress)?.analyze();
            Ok(result)
        }
        "evaluate" | "inspect" => {
            let settings = Heuristics::from_value(options.get("heuristics"))?;
            Ok(if kind == "inspect" {
                evaluate::inspect_evaluation(&position, &settings)
            } else {
                evaluate::evaluate_detailed(&position, &settings)
            })
        }
        "moves" => Ok(
            json!({"moves":rules::pseudo_moves(&position),"active":rules::active_timelines(&position.board),
            "present":rules::present_timelines(&position),"inCheck":rules::in_check(&position),"canSubmit":rules::can_submit(&position)}),
        ),
        "apply" => {
            let action: Action =
                serde_json::from_value(request.get("action").cloned().ok_or("Missing action")?)
                    .map_err(|e| format!("Invalid action: {e}"))?;
            let mut next = position;
            for mv in action {
                if !rules::pseudo_moves(&next).contains(&mv) {
                    return Err("Action contains an illegal component move".into());
                }
                next = rules::apply_move(&next, &mv);
            }
            if request["submit"] != false {
                if !rules::can_submit(&next) {
                    return Err("Action cannot be submitted".into());
                }
                next.action += 1;
            }
            serde_json::to_value(next).map_err(|e| e.to_string())
        }
        "actions" => {
            let mut iter = rules::ActionGenerator::new(
                &position,
                None,
                options["skipOptionalSpatial"] == true,
                options["tacticalOnly"] == true,
            );
            let max_work = options["maxNodes"]
                .as_u64()
                .unwrap_or(100_000)
                .min(1_000_000);
            let mut work = 0;
            let mut actions = vec![];
            let mut complete = false;
            loop {
                let candidate = iter.next(
                    &mut || {
                        if work >= max_work {
                            Err(())
                        } else {
                            work += 1;
                            Ok(())
                        }
                    },
                    &mut |_, _| {},
                );
                match candidate {
                    Ok(Some(c)) => actions.push(json!({"moves":c.moves,"position":c.position})),
                    Ok(None) => {
                        complete = true;
                        break;
                    }
                    Err(()) => break,
                }
            }
            Ok(
                json!({"actions":actions,"work":work,"complete":complete,"policyPruned":iter.policy_pruned}),
            )
        }
        _ => Err(format!("Unknown request type: {kind}")),
    }
}

fn main() {
    let mut line = String::new();
    let read = io::stdin().read_line(&mut line);
    let result = match read {
        Ok(0) => Err("Expected one JSON request on stdin".into()),
        Ok(_) => serde_json::from_str(&line)
            .map_err(|e| format!("Invalid JSON: {e}"))
            .and_then(execute),
        Err(e) => Err(format!("Could not read request: {e}")),
    };
    match result {
        Ok(result) => emit(json!({"type":"result","result":result})),
        Err(error) => {
            emit(json!({"type":"error","error":error}));
            std::process::exit(1);
        }
    }
}
