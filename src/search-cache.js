// A deterministic accounting budget, not a measurement of the JS heap. Each
// retained entry is charged for its full UTF-16 position history, object/map
// metadata and move arrays. Shared arrays are conservatively charged per entry.
const ENTRY_BYTES = 256;
const ARRAY_BYTES = 32;
const SLOT_BYTES = 8;
const MAX_EVICTIONS = 64;

export class SearchCache {
  constructor(maxEntries, maxBytes) {
    this.maxEntries = maxEntries;
    this.maxBytes = maxBytes;
    this.entries = new Map();
    this.arraySizes = new WeakMap();
    this.memoryBytes = 0;
  }

  get size() { return this.entries.size; }
  get(key) { return this.entries.get(key)?.entry; }

  arrayBytes(array, limit) {
    if (!array) return 0;
    const cached = this.arraySizes.get(array);
    if (cached !== undefined) return cached;
    let bytes = ARRAY_BYTES + array.length * SLOT_BYTES;
    if (bytes > limit) return Infinity;
    for (const value of array) {
      if (Array.isArray(value)) bytes += this.arrayBytes(value, limit - bytes);
      if (bytes > limit) return Infinity;
    }
    // Actions and coordinates are immutable once generated. Remember their
    // sizes without retaining them or serializing every suffix of every PV.
    this.arraySizes.set(array, bytes);
    return bytes;
  }

  store(key, entry) {
    if (!this.maxEntries || !this.maxBytes) return false;
    const old = this.entries.get(key);
    if (old && entry.depth < old.entry.depth && entry.flag !== 'exact') return false;
    let bytes = ENTRY_BYTES + key.length * 2;
    if (bytes > this.maxBytes) return false;
    bytes += this.arrayBytes(entry.pv, this.maxBytes - bytes);
    // Normal entries reference the first PV action twice, without retaining a
    // second copy. Charge a separate action only when it is not in the PV.
    if (entry.bestAction && !entry.pv?.includes(entry.bestAction)) {
      bytes += this.arrayBytes(entry.bestAction, this.maxBytes - bytes);
    }
    if (bytes > this.maxBytes) return false;

    let remainingBytes = this.memoryBytes - (old?.bytes || 0);
    let remainingEntries = this.size - (old ? 1 : 0);
    // Most writes fit without evicting anything. Keep that path free of an
    // eviction list and predicate closure; both otherwise allocate per node.
    if (remainingBytes + bytes > this.maxBytes || remainingEntries >= this.maxEntries) {
      const evictions = [];
      for (const [oldKey, value] of this.entries) {
        if (oldKey === key) continue;
        evictions.push(oldKey);
        remainingBytes -= value.bytes;
        remainingEntries--;
        if (remainingBytes + bytes <= this.maxBytes && remainingEntries < this.maxEntries) break;
        // A very large history must not stall search by flushing an entire
        // table. Skip that entry and leave the existing cache intact instead.
        if (evictions.length === MAX_EVICTIONS) return false;
      }
      if (remainingBytes + bytes > this.maxBytes || remainingEntries >= this.maxEntries) return false;
      for (const oldKey of evictions) this.entries.delete(oldKey);
    }
    if (old) {
      old.entry = entry;
      old.bytes = bytes;
    } else {
      this.entries.set(key, { entry, bytes });
    }
    this.memoryBytes = remainingBytes + bytes;
    return true;
  }
}
