import { getHeterogeneousAgentConfigOrThrow } from '../config';
import type {
  AgentEventAdapter,
  HeterogeneousAgentEvent,
  HeterogeneousTerminalErrorData,
  StepCompleteData,
  StreamStartData,
  ToolCallPayload,
  ToolResultData,
  UsageData,
} from '../types';

export const ZCODE_IDENTIFIER = 'zcode';

const ZCODE_AUTH_DOCS_URL = getHeterogeneousAgentConfigOrThrow(ZCODE_IDENTIFIER).auth.docsUrl;
const ZCODE_MISSING_RESULT_MESSAGE =
  'ZCode exited without a terminal result record — the run may be incomplete.';

const finiteNumber = (value: unknown): number =>
  typeof value === 'number' && Number.isFinite(value) ? value : 0;

/**
 * ZCode usage mirrors Anthropic field semantics: `inputTokens`/`outputTokens`
 * already include the cache and reasoning splits respectively, and
 * `totalTokens = inputTokens + outputTokens`.
 */
const toUsageData = (usage: unknown): UsageData | undefined => {
  if (!usage || typeof usage !== 'object') return undefined;
  const record = usage as Record<string, unknown>;
  if (finiteNumber(record.inputTokens) === 0 && finiteNumber(record.outputTokens) === 0) {
    return undefined;
  }

  const input = finiteNumber(record.inputTokens);
  const output = finiteNumber(record.outputTokens);
  const cacheRead = finiteNumber(record.cacheReadTokens);
  const cacheWrite = finiteNumber(record.cacheWriteTokens);
  const reasoning = finiteNumber(record.reasoningTokens);

  return {
    inputCachedTokens: cacheRead,
    inputCacheMissTokens: Math.max(0, input - cacheRead),
    inputWriteCacheTokens: cacheWrite,
    outputReasoningTokens: reasoning,
    outputTextTokens: Math.max(0, output - reasoning),
    totalInputTokens: input,
    totalOutputTokens: output,
    totalTokens: finiteNumber(record.totalTokens) || input + output,
  };
};

const terminalErrorMessage = (payload: any): string => {
  const candidates = [
    payload?.error,
    payload?.message,
    payload?.errorMessage,
    payload?.reason,
    payload?.details?.message,
  ];
  for (const candidate of candidates) {
    if (typeof candidate === 'string' && candidate.trim()) return candidate;
  }
  return 'ZCode execution failed';
};

/**
 * Maps ZCode's `--output-format stream-json` protocol into shared stream
 * events.
 *
 * Every line is an envelope ({eventId, seq, sessionId, turnId, type, payload});
 * `turn.started`/`turn.completed` frame the run, each model round announces a
 * fresh `assistantMessageId` under `model.streaming` (the step boundary), tool
 * lifecycle is keyed by a stable `toolCallId` under `tool.updated`, and a bare
 * terminal `result` record carries the authoritative run usage. ZCode
 * guarantees "exit 0 plus a terminal result record" only as a pair, so
 * {@linkcode validateCompletion} treats a drained stdout without `result` as a
 * protocol error even when the process exit code was 0.
 */
export class ZcodeAdapter implements AgentEventAdapter {
  sessionId?: string;

  private completedToolCallIds = new Set<string>();
  private currentAssistantMessageId?: string;
  private everOpened = false;
  private model?: string;
  private pendingToolPayloads = new Map<string, ToolCallPayload>();
  private settled = false;
  private stepIndex = 0;
  private streamOpen = false;
  private terminalEmitted = false;

  adapt(raw: any): HeterogeneousAgentEvent[] {
    if (!raw || typeof raw !== 'object') return [];
    if (typeof raw.sessionId === 'string' && raw.sessionId) this.sessionId = raw.sessionId;

    switch (raw.type) {
      case 'result': {
        return this.handleResult(raw);
      }
      case 'session.created':
      case 'session.resumed':
      case 'session.titleUpdated':
      case 'session.updated': {
        this.captureModel(raw.payload);
        return [];
      }
      case 'tool.updated': {
        return this.handleToolUpdated(raw.payload);
      }
      case 'turn.completed': {
        // The terminal `result` record is the authoritative usage source; a
        // per-turn emission here would double-count it.
        return [];
      }
      case 'turn.failed': {
        return this.handleTurnFailed(raw.payload);
      }
      case 'turn.started': {
        this.captureModel(raw.payload?.intent?.modelSelection);
        return this.ensureStream();
      }
      case 'model.streaming': {
        return this.handleStreaming(raw.payload);
      }
      default: {
        return [];
      }
    }
  }

  flush(): HeterogeneousAgentEvent[] {
    if (this.settled || this.terminalEmitted) return [];
    return this.drainPendingToolEnds();
  }

  validateCompletion(): HeterogeneousAgentEvent[] {
    if (this.settled || this.terminalEmitted) return [];
    this.terminalEmitted = true;

    const events = [...this.drainPendingToolEnds(), ...this.closeStream()];
    if (this.everOpened || events.length > 0) {
      events.push(this.makeEvent('visible_output_end', {}));
    }
    events.push(
      this.makeEvent('error', {
        agentType: ZCODE_IDENTIFIER,
        clearEchoedContent: true,
        code: 'protocol_error',
        details: {
          expectedEventType: 'result',
          ...(this.sessionId ? { sessionId: this.sessionId } : {}),
        },
        error: ZCODE_MISSING_RESULT_MESSAGE,
        message: ZCODE_MISSING_RESULT_MESSAGE,
      } satisfies HeterogeneousTerminalErrorData),
    );

    return events;
  }

  /**
   * A model round switch (new `assistantMessageId`) is ZCode's step boundary:
   * close the open stream, advance the step, and open the next one.
   */
  private ensureAssistantStep(assistantMessageId: string | undefined): HeterogeneousAgentEvent[] {
    if (!assistantMessageId || assistantMessageId === this.currentAssistantMessageId) return [];

    const firstRound = this.currentAssistantMessageId === undefined;
    this.currentAssistantMessageId = assistantMessageId;
    if (firstRound) {
      // `turn.started` normally opens the first stream; open it here too so a
      // stream that never saw turn.started still yields the round's chunks.
      return this.ensureStream();
    }

    const events = this.closeStream();
    this.stepIndex += 1;
    events.push(...this.openStream());
    return events;
  }

  private captureModel(payload: any): void {
    const modelId = payload?.modelId ?? payload?.modelSelection?.modelId;
    if (typeof modelId === 'string' && modelId) this.model = modelId;
  }

  private ensureStream(): HeterogeneousAgentEvent[] {
    if (this.streamOpen) return [];
    return this.openStream();
  }

  private openStream(): HeterogeneousAgentEvent[] {
    this.streamOpen = true;
    this.everOpened = true;
    const data: StreamStartData & { newStep?: boolean } = {
      model: this.model,
      provider: ZCODE_IDENTIFIER,
      sessionId: this.sessionId,
      ...(this.stepIndex > 0 ? { newStep: true } : {}),
    };
    return [this.makeEvent('stream_start', data)];
  }

  private closeStream(): HeterogeneousAgentEvent[] {
    if (!this.streamOpen) return [];
    this.streamOpen = false;
    return [this.makeEvent('stream_end', {})];
  }

  private handleStreaming(payload: any): HeterogeneousAgentEvent[] {
    if (!payload || typeof payload !== 'object') return [];

    const stepEvents = this.ensureAssistantStep(payload.assistantMessageId);
    if (!this.streamOpen) return stepEvents;

    const kind = payload.kind;
    const delta = typeof payload.delta === 'string' ? payload.delta : '';

    switch (kind) {
      case 'text_delta': {
        if (!delta) return stepEvents;
        return [
          ...stepEvents,
          this.makeEvent('stream_chunk', { chunkType: 'text', content: delta }),
        ];
      }
      case 'reasoning_delta': {
        if (!delta) return stepEvents;
        return [
          ...stepEvents,
          this.makeEvent('stream_chunk', { chunkType: 'reasoning', reasoning: delta }),
        ];
      }
      case 'tool_call': {
        return [...stepEvents, ...this.registerToolCall(payload)];
      }
      case 'finish': {
        return [...stepEvents, ...this.closeStream()];
      }
      default: {
        // start/text_start/text_end/reasoning_start/reasoning_end and the
        // streamed tool-argument deltas carry no state the executor needs:
        // `tool_call` already delivers the complete input object.
        return stepEvents;
      }
    }
  }

  private registerToolCall(payload: any): HeterogeneousAgentEvent[] {
    const toolCallId = payload?.toolCallId;
    if (typeof toolCallId !== 'string' || !toolCallId) return [];
    if (this.completedToolCallIds.has(toolCallId)) return [];
    if (this.pendingToolPayloads.has(toolCallId)) return [];

    const apiName =
      typeof payload.toolName === 'string' && payload.toolName ? payload.toolName : 'unknown';
    const tool: ToolCallPayload = {
      apiName,
      arguments: JSON.stringify(payload.input ?? {}),
      id: toolCallId,
      identifier: ZCODE_IDENTIFIER,
      type: 'default',
    };
    this.pendingToolPayloads.set(toolCallId, tool);

    return [this.makeEvent('stream_chunk', { chunkType: 'tools_calling', toolsCalling: [tool] })];
  }

  private handleToolUpdated(payload: any): HeterogeneousAgentEvent[] {
    if (!payload || typeof payload !== 'object') return [];
    if (payload.kind !== 'result') return [];

    const toolCallId = payload.toolCallId;
    if (typeof toolCallId !== 'string' || !toolCallId) return [];
    if (this.completedToolCallIds.has(toolCallId)) return [];
    this.completedToolCallIds.add(toolCallId);

    const toolCalling = this.pendingToolPayloads.get(toolCallId);
    this.pendingToolPayloads.delete(toolCallId);

    const isError = payload.result?.success !== true;
    const content =
      typeof payload.result?.content === 'string'
        ? payload.result.content
        : payload.result?.error
          ? String(payload.result.error)
          : '';
    const resultData: ToolResultData = { content, isError, toolCallId };

    return [
      this.makeEvent('tool_result', resultData),
      this.makeEvent('tool_end', {
        isSuccess: !isError,
        ...(toolCalling ? { payload: { toolCalling } } : {}),
        result: { content, success: !isError },
        toolCallId,
      }),
    ];
  }

  private handleResult(raw: any): HeterogeneousAgentEvent[] {
    if (this.settled || this.terminalEmitted) return [];
    this.settled = true;

    const usage = toUsageData(raw.usage);
    const data: StepCompleteData = {
      model: this.model,
      phase: 'result_usage',
      provider: ZCODE_IDENTIFIER,
      ...(usage ? { usage } : {}),
    };

    return [
      ...this.drainPendingToolEnds(),
      ...this.closeStream(),
      this.makeEvent('visible_output_end', {}),
      this.makeEvent('step_complete', data),
      this.makeEvent('agent_runtime_end', {}),
    ];
  }

  private handleTurnFailed(payload: any): HeterogeneousAgentEvent[] {
    if (this.settled || this.terminalEmitted) return [];
    this.terminalEmitted = true;

    const message = terminalErrorMessage(payload);
    const authRequired = this.isAuthError(message);
    const data: HeterogeneousTerminalErrorData = {
      agentType: ZCODE_IDENTIFIER,
      clearEchoedContent: true,
      ...(authRequired ? { code: 'auth_required', docsUrl: ZCODE_AUTH_DOCS_URL } : {}),
      details:
        payload && typeof payload === 'object' && Object.keys(payload).length > 0
          ? payload
          : undefined,
      message: authRequired
        ? getHeterogeneousAgentConfigOrThrow(ZCODE_IDENTIFIER).auth.errorMessage
        : message,
      stderr: message,
    };

    return [
      ...this.drainPendingToolEnds(),
      ...this.closeStream(),
      this.makeEvent('visible_output_end', {}),
      this.makeEvent('error', data),
    ];
  }

  private isAuthError(message: string): boolean {
    return /not logged in|please (?:log|sign) in|run `zcode login`/i.test(message);
  }

  private drainPendingToolEnds(): HeterogeneousAgentEvent[] {
    const events = [...this.pendingToolPayloads.keys()].map((toolCallId) =>
      this.makeEvent('tool_end', { isSuccess: false, toolCallId }),
    );
    this.pendingToolPayloads.clear();
    return events;
  }

  private makeEvent(type: HeterogeneousAgentEvent['type'], data: any): HeterogeneousAgentEvent {
    return { data, stepIndex: this.stepIndex, timestamp: Date.now(), type };
  }
}
