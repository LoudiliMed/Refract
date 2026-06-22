// sample_app.js — realistic fixture for Refract JS/TS compression benchmarks.
// A tiny task-queue + HTTP client, ~140 lines, exercising imports, classes,
// methods, async functions, arrow functions and exported declarations.

import http from "http";
import { EventEmitter } from "events";
import { setTimeout as sleep } from "timers/promises";
const crypto = require("crypto");

export const DEFAULTS = {
  retries: 3,
  backoffMs: 250,
  timeoutMs: 5000,
  concurrency: 4,
};

const STATUS = {
  PENDING: "pending",
  RUNNING: "running",
  DONE: "done",
  FAILED: "failed",
};

function uid() {
  return crypto.randomBytes(8).toString("hex");
}

function backoff(attempt, base = DEFAULTS.backoffMs) {
  return base * Math.pow(2, attempt);
}

const isRetryable = (status) => status >= 500 || status === 429;

export async function fetchJson(url, opts = {}) {
  const timeout = opts.timeoutMs || DEFAULTS.timeoutMs;
  let lastError = null;
  for (let attempt = 0; attempt < DEFAULTS.retries; attempt++) {
    try {
      const res = await request(url, timeout);
      if (isRetryable(res.status)) {
        await sleep(backoff(attempt));
        continue;
      }
      return JSON.parse(res.body);
    } catch (err) {
      lastError = err;
      await sleep(backoff(attempt));
    }
  }
  throw lastError || new Error("fetchJson failed");
}

function request(url, timeout) {
  return new Promise((resolve, reject) => {
    const req = http.get(url, (res) => {
      let body = "";
      res.on("data", (chunk) => {
        body += chunk;
      });
      res.on("end", () => resolve({ status: res.statusCode, body }));
    });
    req.setTimeout(timeout, () => req.destroy(new Error("timeout")));
    req.on("error", reject);
  });
}

export class Task {
  constructor(fn, name) {
    this.id = uid();
    this.fn = fn;
    this.name = name || "task";
    this.status = STATUS.PENDING;
    this.result = null;
    this.error = null;
  }

  async run() {
    this.status = STATUS.RUNNING;
    try {
      this.result = await this.fn();
      this.status = STATUS.DONE;
      return this.result;
    } catch (err) {
      this.error = err;
      this.status = STATUS.FAILED;
      throw err;
    }
  }

  describe() {
    return `${this.name}#${this.id} [${this.status}]`;
  }
}

export default class Queue extends EventEmitter {
  constructor(concurrency = DEFAULTS.concurrency) {
    super();
    this.concurrency = concurrency;
    this.pending = [];
    this.active = 0;
  }

  add(fn, name) {
    const task = new Task(fn, name);
    this.pending.push(task);
    this.emit("queued", task.describe());
    this.drain();
    return task.id;
  }

  drain() {
    while (this.active < this.concurrency && this.pending.length > 0) {
      const task = this.pending.shift();
      this.active++;
      task
        .run()
        .then((value) => this.emit("done", task.id, value))
        .catch((err) => this.emit("failed", task.id, err))
        .finally(() => {
          this.active--;
          this.drain();
        });
    }
  }

  get size() {
    return this.pending.length + this.active;
  }
}

export function createQueue(opts = {}) {
  return new Queue(opts.concurrency);
}
