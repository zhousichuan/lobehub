import { readFile } from 'node:fs/promises';

import { describe, expect, it } from 'vitest';

import { ZcodeAdapter } from './zcode';

const adaptJsonl = async (adapter: ZcodeAdapter, fixture: string) => {
  const jsonl = await readFile(new URL(`./__fixtures__/zcode/${fixture}`, import.meta.url), 'utf8');
  return jsonl
    .trim()
    .split('\n')
    .flatMap((line) => adapter.adapt(JSON.parse(line)));
};

const envelope = (type: string, payload?: unknown) => ({
  eventId: `evt-${Math.random().toString(36).slice(2)}`,
  sessionId: 'sess_zcode_fixture',
  timestamp: 1795179602874,
  traceId: 'trace_zcode_fixture',
  turnId: 'turn_zcode_fixture',
  type,
  ...(payload !== undefined ? { payload } : {}),
});

describe('ZcodeAdapter', () => {
  it('maps a full real trace: two steps, tool lifecycle, and terminal usage', async () => {
    const adapter = new ZcodeAdapter();
    const events = await adaptJsonl(adapter, 'basic.jsonl');

    expect(adapter.sessionId).toBe('sess_zcode_fixture');
    expect(events.map(({ type }) => type)).toEqual([
      'stream_start',
      // round 1: reasoning deltas stream within step 0 (empty deltas skipped)
      'stream_chunk',
      'stream_chunk',
      // tool_call announces the pending tool on the first assistant round
      'stream_chunk',
      // the assistant round finishes BEFORE its tool result lands
      'stream_end',
      'tool_result',
      'tool_end',
      // second model round (final answer) opens a new step
      'stream_start',
      'stream_chunk',
      'stream_chunk',
      'stream_chunk',
      'stream_chunk',
      'stream_chunk',
      'stream_end',
      'visible_output_end',
      'step_complete',
      'agent_runtime_end',
    ]);

    // The opener carries the session, provider, and model captured from the
    // turn intent / session updates.
    expect(events[0]).toMatchObject({
      data: {
        model: 'GLM-5.3',
        provider: 'zcode',
        sessionId: 'sess_zcode_fixture',
      },
      stepIndex: 0,
      type: 'stream_start',
    });
    expect(events[0].data.newStep).toBeUndefined();

    // Reasoning deltas stream as reasoning chunks within step 0.
    expect(events[1].data).toEqual({ chunkType: 'reasoning', reasoning: 'The' });
    expect(events[1].stepIndex).toBe(0);

    // The tool_call event carries the complete input object.
    expect(events[3].data).toEqual({
      chunkType: 'tools_calling',
      toolsCalling: [
        {
          apiName: 'Read',
          arguments: '{"file_path":"/data/work/lobehub/zcode-cli-test/e2e/test.txt"}',
          id: 'call_40eab8ba37b74e7c937bf9de',
          identifier: 'zcode',
          type: 'default',
        },
      ],
    });

    // tool.updated/result resolves by callId with success flag and content.
    expect(events[5].data).toMatchObject({
      content: '1\thello from lobeHub integration test\n2\t',
      isError: false,
      toolCallId: 'call_40eab8ba37b74e7c937bf9de',
    });
    expect(events[6]).toMatchObject({
      data: { isSuccess: true, toolCallId: 'call_40eab8ba37b74e7c937bf9de' },
      type: 'tool_end',
    });

    // The new assistantMessageId after the tool round is the step boundary.
    expect(events[7]).toMatchObject({
      data: { newStep: true, provider: 'zcode' },
      stepIndex: 1,
      type: 'stream_start',
    });
    // Round 2 streams its reasoning, then the final text answer.
    expect(events[8].data).toEqual({ chunkType: 'reasoning', reasoning: 'The' });
    expect(events[10].data).toEqual({ chunkType: 'text', content: 'The' });
    expect(events[10].stepIndex).toBe(1);
    expect(events[12].data.chunkType).toBe('text');

    // Terminal result record is the authoritative usage source, mapped once.
    expect(events[14].type).toBe('visible_output_end');
    expect(events[15]).toMatchObject({
      data: {
        model: 'GLM-5.3',
        phase: 'result_usage',
        provider: 'zcode',
        usage: {
          inputCachedTokens: 30336,
          inputCacheMissTokens: 10743,
          inputWriteCacheTokens: 0,
          outputReasoningTokens: 0,
          outputTextTokens: 97,
          totalInputTokens: 41079,
          totalOutputTokens: 97,
          totalTokens: 41176,
        },
      },
      stepIndex: 1,
      type: 'step_complete',
    });
    expect(events[16].type).toBe('agent_runtime_end');
  });

  it('opens the first stream from turn.started and captures the model from the intent', () => {
    const adapter = new ZcodeAdapter();
    adapter.adapt(
      envelope('session.updated', {
        modelId: 'GLM-5.3',
        providerId: 'account:bigmodel-individual-coding-plan',
      }),
    );

    const events = adapter.adapt(
      envelope('turn.started', {
        intent: {
          modelSelection: { modelId: 'GLM-4.6', providerId: 'zai-coding-plan' },
        },
        messageId: 'msg_user_1',
      }),
    );

    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({
      data: { model: 'GLM-4.6', provider: 'zcode', sessionId: 'sess_zcode_fixture' },
      stepIndex: 0,
      type: 'stream_start',
    });
  });

  it('still streams the first round when turn.started never arrives', () => {
    const adapter = new ZcodeAdapter();
    const events = adapter.adapt(
      envelope('model.streaming', {
        assistantMessageId: 'msg_a1',
        delta: 'hello',
        kind: 'text_delta',
      }),
    );

    expect(events.map(({ type }) => type)).toEqual(['stream_start', 'stream_chunk']);
    expect(events[0].data.provider).toBe('zcode');
  });

  it('maps turn.failed into a classified terminal error and stops the run', () => {
    const adapter = new ZcodeAdapter();
    const events = adapter.adapt(envelope('turn.failed', { error: 'usage limit reached' }));

    expect(events.map(({ type }) => type)).toEqual(['visible_output_end', 'error']);
    expect(events[1].data).toMatchObject({
      agentType: 'zcode',
      message: 'usage limit reached',
      stderr: 'usage limit reached',
    });
    expect(events[1].data.code).toBeUndefined();
    // The run is settled: later events and flush are inert.
    expect(adapter.adapt(envelope('result', { usage: {} }))).toEqual([]);
    expect(adapter.flush()).toEqual([]);
  });

  it('classifies auth failures from turn.failed for the shared status guide', () => {
    const adapter = new ZcodeAdapter();
    const events = adapter.adapt(envelope('turn.failed', { error: 'Not logged in' }));

    expect(events[1].data).toMatchObject({
      agentType: 'zcode',
      code: 'auth_required',
      docsUrl: 'https://github.com/kingsword09/zcode-cli#readme',
      message: 'ZCode could not authenticate. Run `zcode login`, then retry.',
    });
  });

  it('emits a protocol error when stdout drains without the terminal result', () => {
    const adapter = new ZcodeAdapter();
    adapter.adapt(envelope('turn.started', {}));

    const events = adapter.validateCompletion();
    expect(events.map(({ type }) => type)).toEqual(['stream_end', 'visible_output_end', 'error']);
    expect(events[2].data).toMatchObject({
      agentType: 'zcode',
      code: 'protocol_error',
      details: { expectedEventType: 'result', sessionId: 'sess_zcode_fixture' },
    });
    expect(events[2].data.error).toContain('without a terminal result record');
    // Second validation is inert.
    expect(adapter.validateCompletion()).toEqual([]);
  });

  it('drains unresolved tool calls as failed tool_end on flush', () => {
    const adapter = new ZcodeAdapter();
    adapter.adapt(
      envelope('model.streaming', {
        assistantMessageId: 'msg_flush',
        input: { command: 'sleep 100' },
        kind: 'tool_call',
        toolCallId: 'call_flush_1',
        toolName: 'Bash',
      }),
    );

    expect(adapter.flush().map(({ type }) => type)).toEqual(['tool_end']);
    expect(adapter.flush()).toEqual([]);
  });

  it('ignores malformed input, unknown event types, and duplicate tool results', () => {
    const adapter = new ZcodeAdapter();
    expect(adapter.adapt(null)).toEqual([]);
    expect(adapter.adapt('bad')).toEqual([]);
    expect(adapter.adapt({ type: 'future_event' })).toEqual([]);
    expect(
      adapter.adapt(envelope('model.streaming', { kind: 'tool_input_delta', delta: '{"fi' })),
    ).toEqual([]);

    adapter.adapt(
      envelope('model.streaming', {
        assistantMessageId: 'msg_dedupe',
        input: {},
        kind: 'tool_call',
        toolCallId: 'call_dupe',
        toolName: 'Read',
      }),
    );
    adapter.adapt(
      envelope('tool.updated', {
        kind: 'result',
        result: { success: true, content: 'a' },
        toolCallId: 'call_dupe',
      }),
    );
    expect(
      adapter.adapt(
        envelope('tool.updated', {
          kind: 'result',
          result: { success: true, content: 'a' },
          toolCallId: 'call_dupe',
        }),
      ),
    ).toEqual([]);
    // A second tool_call for the same id is also suppressed.
    expect(
      adapter.adapt(
        envelope('model.streaming', {
          assistantMessageId: 'msg_dedupe',
          input: {},
          kind: 'tool_call',
          toolCallId: 'call_dupe',
          toolName: 'Read',
        }),
      ),
    ).toEqual([]);
  });
});
