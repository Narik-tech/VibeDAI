import { existsSync } from 'node:fs';
import { TransformerRuntime } from './transformer-runtime.js';
import { LEELA_ID, LEELA_NAME, LEELA_ARCHITECTURE, resolveLeelaConfig } from './leela-config.js';

/** An independent service and checkpoint for the LCZero transfer engine. */
export class LeelaRuntime extends TransformerRuntime {
  constructor(options = {}) {
    super({ ...resolveLeelaConfig(), ...options });
  }

  describe() {
    const info = super.describe();
    const missing = !existsSync(this.checkpoint)
      ? 'No trained Leela checkpoint. Train the LCZero transfer model with node scripts/lc0.js train; see docs/lc0-transfer.md.'
      : !existsSync(this.python) ? 'Python environment missing. Run npm run transformer:setup.' : null;
    return { ...info, id: LEELA_ID, name: LEELA_NAME,
      description: 'LCZero spatial features with trained 5D timeline, value, and move adapters. Experimental strength.',
      error: missing || info.error, checkpoint: this.checkpoint,
    };
  }

  async start() {
    const info = await super.start();
    if (info.model.architecture !== LEELA_ARCHITECTURE) {
      const error = new Error(`${LEELA_NAME} requires a trained LCZero transfer checkpoint (${LEELA_ARCHITECTURE}).`);
      this.state = 'error';
      this.error = error.message;
      this.stopProcess(error);
      throw error;
    }
    return info;
  }
}
