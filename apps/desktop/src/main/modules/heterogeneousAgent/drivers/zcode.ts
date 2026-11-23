import { buildZcodeArgs } from '@lobechat/heterogeneous-agents/spawn';

import type { HeterogeneousAgentBuildPlanParams, HeterogeneousAgentDriver } from '../types';

/**
 * ZCode configures MCP servers through its own settings file
 * (`ZCODE_CLI_SETTINGS_FILE`), not a `--mcp-config` argv flag, so the shared
 * `mcpConfigPath` plan input is intentionally unused here.
 */
export const zcodeDriver: HeterogeneousAgentDriver = {
  async buildSpawnPlan({
    args,
    helpers,
    promptInput,
    resumeSessionId,
  }: HeterogeneousAgentBuildPlanParams) {
    const inputPlan = await helpers.buildAgentInput('zcode', promptInput);

    return {
      args: buildZcodeArgs({
        extraArgs: [...args],
        inputArgs: inputPlan.args,
        resumeSessionId,
      }),
      stdinPayload: inputPlan.stdin,
    };
  },
};
