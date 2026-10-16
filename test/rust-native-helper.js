import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { DEFAULT_CLASSICAL_ENGINE_PATH } from '../src/rust-engine.js';

export const nativeExecutable = process.env.CLASSICAL_ENGINE_PATH || DEFAULT_CLASSICAL_ENGINE_PATH;
export const nativeTestOptions = {
  skip: !existsSync(nativeExecutable) && 'Build the native engine first with npm run build:classical.',
};

export function nativeRequest(request) {
  const child = spawnSync(nativeExecutable, [], {
    input: JSON.stringify(request) + '\n', encoding: 'utf8', windowsHide: true,
    timeout: 15_000, maxBuffer: 64 * 1024 * 1024,
  });
  assert.ifError(child.error);
  assert.equal(child.status, 0, `${request.type}: ${child.stderr || child.stdout}`);
  const messages = child.stdout.trim().split('\n').filter(Boolean).map(line => JSON.parse(line));
  const error = messages.find(message => message.type === 'error');
  assert.equal(error, undefined, `${request.type}: ${error?.error}`);
  const final = messages.find(message => message.type === 'result');
  assert.ok(final, `${request.type}: native process did not return a result: ${child.stdout}`);
  return final.result;
}
