import path from 'node:path';
import { DEFAULT_PYTHON, PROJECT_ROOT } from './transformer-runtime.js';

export const LEELA_ID = 'leela';
export const LEELA_NAME = 'Leela in a 5D Trenchcoat';
export const LEELA_ARCHITECTURE = '5d-lc0-transfer-v1';
export const DEFAULT_LEELA_CHECKPOINT = path.join(PROJECT_ROOT, 'artifacts/lc0/best.pt');
export const DEFAULT_LEELA_RUN_DIR = path.join(PROJECT_ROOT, 'artifacts/lc0/selfplay');
export const DEFAULT_LEELA_SEED_DATA = path.join(PROJECT_ROOT, 'artifacts/lc0/curriculum.jsonl');
export const DEFAULT_LEELA_SUITE = path.join(PROJECT_ROOT, 'examples/matches/lc0-training.json');

/** Analysis and UI training must select the same Leela checkpoint. */
export function resolveLeelaConfig() {
  return {
    checkpoint: path.resolve(process.env.LEELA_CHECKPOINT || DEFAULT_LEELA_CHECKPOINT),
    python: path.resolve(process.env.LEELA_PYTHON || process.env.TRANSFORMER_PYTHON || DEFAULT_PYTHON),
    device: process.env.LEELA_DEVICE || process.env.TRANSFORMER_DEVICE || 'auto',
    runDir: path.resolve(process.env.LEELA_RUN_DIR || DEFAULT_LEELA_RUN_DIR),
    seedData: path.resolve(process.env.LEELA_SEED_DATA || DEFAULT_LEELA_SEED_DATA),
    suite: path.resolve(process.env.LEELA_SUITE || DEFAULT_LEELA_SUITE),
  };
}

export function isNeuralEngine(engine) {
  return engine === 'transformer' || engine === LEELA_ID;
}
