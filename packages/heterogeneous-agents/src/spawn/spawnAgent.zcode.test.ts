import type * as childProcess from 'node:child_process';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import * as resolveCliCommand from './resolveCliCommand';

const spawnCalls: Array<{ args: string[]; command: string; options: any }> = [];
let nextFakeProc: any = null;

vi.mock('node:child_process', async () => {
  const actual = await vi.importActual<typeof childProcess>('node:child_process');
  return {
    ...actual,
    spawn: vi.fn((command: string, args: string[], options: any) => {
      spawnCalls.push({ args, command, options });
      return nextFakeProc;
    }),
  };
});

vi.mock('./resolveCliCommand', async () => {
  const actual = await vi.importActual<typeof resolveCliCommand>('./resolveCliCommand');
  return { ...actual, detectHeterogeneousCliCommand: vi.fn() };
});

const detectHeterogeneousCliCommandMock = vi.mocked(
  resolveCliCommand.detectHeterogeneousCliCommand,
);

const createFakeProc = ({ stdoutChunks = [] as string[] } = {}) => {
  const proc = new EventEmitter() as any;
  const stdout = new PassThrough();
  const stderr = new PassThrough();
  const stdinWrites: string[] = [];
  proc.stdout = stdout;
  proc.stderr = stderr;
  proc.stdin = {
    end: vi.fn(),
    once: vi.fn(),
    write: vi.fn((chunk: string, cb?: () => void) => {
      stdinWrites.push(chunk);
      cb?.();
      return true;
    }),
  };
  proc.kill = vi.fn();
  proc.killed = false;
  proc.pid = 12_345;

  const start = () => {
    setImmediate(() => {
      for (const c of stdoutChunks) stdout.write(c);
      stdout.end();
      stderr.end();
      proc.emit('exit', 0, null);
    });
  };

  return { proc, start, stdinWrites };
};

const zcodeTurn = `${JSON.stringify({
  eventId: 'evt-1',
  payload: {
    intent: { modelSelection: { modelId: 'GLM-5.3', providerId: 'zai-coding-plan' } },
    messageId: 'msg_user_1',
  },
  sessionId: 'sess_zcode_spawn',
  seq: 1,
  timestamp: 1795179602870,
  traceId: 'trace_zcode_spawn',
  turnId: 'turn_zcode_spawn',
  type: 'turn.started',
})}\n`;

const zcodeResult = `${JSON.stringify({
  response: 'done',
  sessionId: 'sess_zcode_spawn',
  traceId: 'trace_zcode_spawn',
  turnId: 'turn_zcode_spawn',
  type: 'result',
  usage: {
    cacheReadTokens: 0,
    cacheWriteTokens: 0,
    inputTokens: 10,
    outputTokens: 5,
    reasoningTokens: 0,
    totalTokens: 15,
  },
})}\n`;

describe('spawnAgent (zcode)', () => {
  beforeEach(() => {
    spawnCalls.length = 0;
    nextFakeProc = null;
    detectHeterogeneousCliCommandMock.mockResolvedValue({ available: true, path: 'zcode' });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('spawns ZCode with its stream-json protocol, inline -p prompt, and resume id', async () => {
    const fake = createFakeProc({ stdoutChunks: [zcodeTurn, zcodeResult] });
    nextFakeProc = fake.proc;

    const { spawnAgent } = await import('./spawnAgent');
    const handle = await spawnAgent({
      agentType: 'zcode',
      operationId: 'op-zcode',
      prompt: 'continue with ZCode',
      resumeSessionId: 'sess_zcode_prev',
    });
    fake.start();

    const events: any[] = [];
    for await (const event of handle.events) events.push(event);
    await handle.exit;

    expect(spawnCalls[0]).toMatchObject({
      command: 'zcode',
    });
    expect(spawnCalls[0].args).toEqual([
      '--output-format',
      'stream-json',
      '--mode',
      'yolo',
      '--disallowed-tools',
      'AskUserQuestion',
      '--resume',
      'sess_zcode_prev',
      '-p',
      'continue with ZCode',
    ]);
    // ZCode takes the prompt over argv; stdin carries only the empty EOF write.
    expect(fake.stdinWrites).toEqual(['']);

    // The stream-json protocol still produces the shared terminal events; the
    // fixture omits the mid-round `finish`, so `result` closes the open stream.
    expect(events.map(({ type }) => type)).toEqual([
      'stream_start',
      'stream_end',
      'visible_output_end',
      'step_complete',
      'agent_runtime_end',
    ]);
    expect(events.find(({ type }) => type === 'step_complete')?.data).toMatchObject({
      phase: 'result_usage',
      provider: 'zcode',
      usage: { totalInputTokens: 10, totalOutputTokens: 5, totalTokens: 15 },
    });
  });
});
