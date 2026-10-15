# 多智能体管理 preset（`team-mgmt`）

一个 `@deepseek-ai/dsh-agent-preset` 声明，给 Lead 会话装齐"角色→模型路由"和"多智能体协作流程契约"。
本目录是 bundle 包，patch 文件是 `cordis.patch.yml`，preset id 为 `team-mgmt`。

## 四层归属（为什么只有两层在这里声明）

| 层 | 谁提供 | 说明 |
|---|---|---|
| 协调层（Team 服务 / `spawn_teammate` / 任务板） | **宿主 profile bundle** `@deepseek-ai/dsh-experimental-agent-team-profile` | 已在本机 `web` profile 启用。**不能**写进 preset 的 `plugins`：它是 bundle patch，其 `lib/index.js` 是空模块（见其 `package.json` 的 `dsh.bundle.patch`）。同一进程里再挂一份会重复注册同名全局工具（`spawn_teammate`/`send_message`/`list_agents`/`interrupt_agent`/`team_task_*`）。 |
| 路由层 | **本 preset** | 4 个 `@deepseek-ai/dsh-tool-subagent` 实例，一个实例 = 一个 provider + 一个模型 + 一个工具名。 |
| 流程层 | **本 preset** | `@deepseek-ai/dsh-persona` 的作业契约（证据、只读审查、单 writer、验收、预算、升级链）+ `tool-todo`（禁止并行 in_progress）。 |
| 界面层 | **宿主 profile bundle** | `@deepseek-ai/dsh-experimental-client-ui-agent-team`（成员名册、任务板、打开成员会话），随协调层同一个 bundle 启用。 |

## 角色路由表

| 工具名 | provider / model | reasoningEffort | 角色 |
|---|---|---|---|
| `subagent_dev` | `workbuddy` / `glm-5.3-flashx` | 继承 | 唯一 writer |
| `subagent_review` | `openai-codex` / `gpt-6.1-sol` | `xhigh` | 只读审查（工具上摘掉 `write`/`edit`） |
| `subagent_design` | `openai-codex` / `gpt-6-astra` | `xhigh` | 方案设计 |
| `subagent_escalate` | `openai-codex` / `gpt-6.1-sol` | `high` | 降级 writer（dev 同类失败 2 次后接手） |

Lead 自己的模型不在这里：它由宿主 `@deepseek-ai/dsh-agent-default-model` 决定，preset 改不了。

## 安装

```sh
# 在 DSH 会话里（plugin_manager 工具，需要 Full access）
#   action: install_bundle
#   target: /mnt/data2/weizhiwei/AERO_WORLD/dsh-presets/team-mgmt
```

或 CLI：

```sh
dsh plugin --profile web add /mnt/data2/weizhiwei/AERO_WORLD/dsh-presets/team-mgmt
```

**本机安装记录（2026-10-14，已完成）**：首次安装被一个与 preset 无关的环境问题挡住 —— 该 profile 的 `node_modules` 记录的 store 与 pnpm 10 按盘选中的 store 不一致：

```
ERR_PNPM_UNEXPECTED_STORE
The dependencies at "/mnt/data2/weizhiwei/.dsh/profiles/web/node_modules" are currently linked from
the store at "/home/weizhiwei/.local/share/pnpm/store/v10".
pnpm now wants to use the store at "/mnt/data2/.pnpm-store/v10".
```

已用最小改动修好（新增一个 profile 局部文件，删掉即还原默认行为）：

```
/mnt/data2/weizhiwei/.dsh/profiles/web/.npmrc
  store-dir=/home/weizhiwei/.local/share/pnpm/store/v10
```

之后 `install_bundle` 返回 `application: "applied"`、`warnings: []`，`web` profile 的 bundles 变为
`[…, 'dsh-llm-workbuddy', '@deepseek-ai/dsh-experimental-agent-team-profile', '@deepseek-ai/dsh-experimental-schedule-bundle', '@local/dsh-team-mgmt-preset']`，
preset 名册从 4 个变 5 个（`preset-team-mgmt` 在列）。不用这个修法的话，替代方案是 `cd <profile> && pnpm install` 重新链接依赖。

删掉 `store-dir` 行后，该 profile 的任何插件安装会重新报 ERR_PNPM_UNEXPECTED_STORE。

不改 profile 的等价用法（一次性 overlay，本目录的 patch 可直接当 `--patch` 用）：

```sh
dsh --profile web --patch dsh-presets/team-mgmt/cordis.patch.yml --dump-config
```

## 选中

- GUI：新会话在 preset 选择器里选"多智能体管理"。
- 或者把 `cordis.patch.yml` 末尾注释掉的 `agent-preset-registry` 三行取消注释（把 `team-mgmt` 设为该 profile 新会话默认）。
- **剩余的唯一未完成验证**：preset 作用域要在真实 Web 会话里 join 才会生效。安装完成后新开一个会话、选中"多智能体管理"，
  确认工具表里有 `subagent_dev` / `subagent_review` / `subagent_design` / `subagent_escalate`（同时还有 Team 的 `spawn_teammate` / `team_task_*`）。
  在此之前，"组合正确 + 角色工具在真实会话里可跑"已由 `probe-mechanism.yml` 在 headless 里实测（见下表），差别只在 preset 作用域这一层。
- **只在会 join preset 的入口有效**：`dsh --profile <name> "<task>"`（headless runner）即使 registry 的 `default` 指向本 preset 也不会 join —— 实测该会话只有 `subagent`/`subagent_fork`/`spawn_teammate`，没有 `subagent_dev`/`subagent_review`。

## 还原

```sh
# 卸载 bundle（plugin_manager action: remove_bundle，或）
dsh plugin --profile web remove @local/dsh-team-mgmt-preset
```

已有会话保持它启动时的组合，只有新会话受影响。

## 实测记录（2026-10-14）

命令（不修改任何 profile，只用 overlay）：

```sh
cd /mnt/data2/weizhiwei/AERO_WORLD
dsh --profile ds41_flash \
  --patch dsh-presets/verify/probe-mechanism.yml \
  --json '调用 subagent_dev（prompt: Reply with exactly: DEV_OK）；再调用 subagent_review（prompt: Reply with exactly: REVIEW_OK）。把两次返回原文逐字贴出来。'
```

| 结论 | 证据 |
|---|---|
| 静态 `agentOptions` 真的钉住路由 | dev 子会话 `request/header.config = {"provider":"workbuddy","model":"glm-5.3-flashx","maxTokens":32000}`；父会话是 `workbuddy/deepseek-v4.1-flash`，**不是继承** |
| `reasoningEffort` 同样生效 | review 子会话 `{"provider":"openai-codex","model":"gpt-6-astra","reasoningEffort":"xhigh","maxTokens":128000}` |
| `toolFilter.deny` 真的摘工具 | review 子会话工具表无 `write`/`edit`；dev 子会话有 |
| `maxDepth` 语义（**踩坑**） | `maxDepth: 0` → `Error: subagent depth 1 exceeds maxDepth 0`，**连 Lead 自己的委派都拒**；改 `1` 后：一层委派成功，子会话再往下 → `Error: subagent depth 2 exceeds maxDepth 1` |
| 与 Team bundle 共存 | 同一会话里 `subagent_dev`/`subagent_review` 与 `spawn_teammate`/`team_task_*` 同时可用；`--dump-config` 在 `web` profile 上 exit 0、零告警 |
| 子会话能看到角色工具 | 子会话工具表含 `subagent_dev`/`subagent_review`，所以递归只由 `maxDepth: 1` 兜住 |

## 已知边界（不粉饰）

- 只读审查目前只有 `toolFilter` 摘掉 `write`/`edit`；子会话仍持有 `bash`，理论上仍可写文件。硬约束需要给该会话换 `read-only` 权限预设，而 preset 目前无法按实例指定 sandbox/approval。
- `modelSelectionSettings: true` 单独打开不给路由：它要宿主 `subagent-model-selection` 设置里有非空授权路由表才暴露 `provider`/`model`/`reasoning_effort` 字段。本机会话里的 `subagent` 工具就没有这些字段。静态 `agentOptions` 才是可靠路径。另外同一 scope 里最多一个实例能拥有 model selection（`list_subagent_models` 是全局名）。
- 所有成员共享同一工作目录，没有 worktree 隔离；"写作用域"是流程约定 + 任务看板登记，不是文件锁。
- `persona` 的 `suffix` 默认空且会遮掉宿主 suffix，所以这里显式还原成 `Your working directory is {{cwd}}.`。
