# 智能体规则

## startup/state-file-fresh
- category: startup
- check: state_file_fresh
智能体在任何工具调用之前必须读取 agent_state.json。

## forbidden/no-release-script-edits
- category: forbidden
- check: no_release_script_edits
严禁在经批准的发布任务之外编辑 scripts/release.sh。

## done/tests-pass
- category: definition_of_done
- check: tests_pass
只有当任务的验收命令以零退出码退出时，该任务才算完成。

## uncertainty/open-question-note
- category: uncertainty
- check: opened_question_when_unsure
当置信度低于阈值时，应记录一条问题说明，而非进行猜测。

## approval/new-dependency
- category: approval
- check: new_dependency_approved
添加运行时依赖需要明确的人工批准。
