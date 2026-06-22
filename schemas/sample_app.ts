// sample_app.ts — TypeScript fixture for Refract compression tests.
// Exercises interfaces, type aliases, generics, typed params, abstract classes.

import { EventEmitter } from "events";
import type { Logger } from "./logger";

export interface Job<T> {
  id: string;
  payload: T;
  attempts: number;
}

export type Handler<T, R> = (payload: T) => Promise<R>;

type Result<R> = { ok: true; value: R } | { ok: false; error: Error };

export const SETTINGS = {
  maxAttempts: 5,
  timeoutMs: 3000,
};

function nextId(prefix: string): string {
  return `${prefix}-${Math.random().toString(36).slice(2)}`;
}

export async function runHandler<T, R>(
  handler: Handler<T, R>,
  payload: T,
): Promise<Result<R>> {
  try {
    const value = await handler(payload);
    return { ok: true, value };
  } catch (error) {
    return { ok: false, error: error as Error };
  }
}

const wrap = <T>(value: T): Job<T> => ({
  id: nextId("job"),
  payload: value,
  attempts: 0,
});

export abstract class Worker<T, R> extends EventEmitter {
  protected logger?: Logger;

  constructor(logger?: Logger) {
    super();
    this.logger = logger;
  }

  abstract handle(job: Job<T>): Promise<R>;

  async process(payload: T): Promise<Result<R>> {
    const job = wrap(payload);
    return runHandler((p) => this.handle(wrap(p)), payload);
  }
}

export class EchoWorker extends Worker<string, string> {
  async handle(job: Job<string>): Promise<string> {
    return job.payload;
  }
}
