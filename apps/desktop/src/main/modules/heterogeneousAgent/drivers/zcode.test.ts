import { describe, expect, it, vi } from 'vitest';

import { getHeterogeneousAgentDriver } from '../index';
import type {
  HeterogeneousAgentBuildPlanHelpers,
  HeterogeneousAgentBuildPlanParams,
} from '../types';
import { zcodeDriver } from './zcode';

const buildAgentInput = vi.fn(async () => ({
  args: ['-p', 'raw prompt', '--attach', '/tmp/image.png'],
  stdin: '',
}));
const helpers: HeterogeneousAgentBuildPlanHelpers = { buildAgentInput };

const buildParams = (
  overrides: Partial<HeterogeneousAgentBuildPlanParams> = {},
): HeterogeneousAgentBuildPlanParams => ({
  args: [],
  helpers,
  promptInput: 'raw prompt',
  ...overrides,
});

describe('zcodeDriver', () => {
  it('is registered and composes stream-json, resume, custom, and attachment args', async () => {
    expect(getHeterogeneousAgentDriver('zcode')).toBe(zcodeDriver);

    const plan = await zcodeDriver.buildSpawnPlan(
      buildParams({
        args: ['--model', 'zai/GLM-5.3'],
        mcpConfigPath: '/tmp/lobe-zcode-mcp.json',
        resumeSessionId: 'sess_zcode_1',
      }),
    );

    expect(buildAgentInput).toHaveBeenCalledWith('zcode', 'raw prompt');
    // The shared `mcpConfigPath` is intentionally unused: ZCode wires MCP
    // servers through its settings file, not a `--mcp-config` flag.
    expect(plan).toEqual({
      args: [
        '--output-format',
        'stream-json',
        '--mode',
        'yolo',
        '--disallowed-tools',
        'AskUserQuestion',
        '--resume',
        'sess_zcode_1',
        '--model',
        'zai/GLM-5.3',
        '-p',
        'raw prompt',
        '--attach',
        '/tmp/image.png',
      ],
      stdinPayload: '',
    });
  });
});
