#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { delimiter, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const manifest = fileURLToPath(new URL('../rust/classical-engine/Cargo.toml', import.meta.url));
// Pin the output directory so CARGO_TARGET_DIR cannot move the executable away
// from the path used by the Node adapter.
const target = fileURLToPath(new URL('../rust/classical-engine/target', import.meta.url));
const cargoName = process.platform === 'win32' ? 'cargo.exe' : 'cargo';
const searchPath = process.env.PATH || process.env.Path || '';
const cargoOnPath = searchPath.split(delimiter).some(directory => directory && existsSync(join(directory, cargoName)));
const localToolchain = fileURLToPath(new URL('../artifacts/rust-toolchain', import.meta.url));
const localCargo = join(localToolchain, 'cargo', 'bin', cargoName);
const useLocal = !cargoOnPath && existsSync(localCargo);
const env = { ...process.env };
if (useLocal) {
  env.CARGO_HOME = join(localToolchain, 'cargo');
  env.RUSTUP_HOME = join(localToolchain, 'rustup');
  // Do not pass duplicate Path/PATH keys to Windows subprocesses.
  for (const key of Object.keys(env)) if (key.toLowerCase() === 'path') delete env[key];
  env.PATH = join(localToolchain, 'cargo', 'bin') + delimiter + searchPath;
}
const command = process.argv.includes('--test') ? 'test' : 'build';
const child = spawn(useLocal ? localCargo : 'cargo', [command, '--release', '--locked', '--manifest-path', manifest, '--target-dir', target], {
  stdio: 'inherit', windowsHide: true, env,
});
child.on('error', error => {
  console.error(error.code === 'ENOENT'
    ? 'Rust is required to build the classical engine. Install the stable toolchain from https://rustup.rs, reopen your terminal, and run npm run build:classical.'
    : `Unable to run Cargo: ${error.message}`);
  process.exitCode = 1;
});
child.on('exit', (code, signal) => {
  process.exitCode = signal ? 1 : code ?? 1;
});
