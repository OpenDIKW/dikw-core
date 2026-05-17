## Summary

<!-- 1-3 句话回答 why。重点说动机，不是 what（diff 就是 what）。 -->

## Test plan

<!-- 勾选所有适用项。逐项打勾，不要整段照抄。 -->

- [ ] `uv run ruff check . && uv run mypy src && uv run pytest`
- [ ] 涉及 `storage/` / `contract` / engine 改动？本地跑过 `pgvector/pgvector:pg16` 全量 contract
- [ ] 涉及 K-layer (`domains/knowledge/`) 或 Retrieval (`domains/info/`) 改动？`evals/BASELINES.md` 已写**真实数据**（数字不是占位）
- [ ] 改了 on-disk wiki/wisdom 格式？`docs/design.md` 已更新
- [ ] 新增 `SourceBackend` / `Storage` 适配器？contract test 已扩展，不是只测自己的实现
- [ ] CLI 命令 / flag / JSON 输出有变？同步检查 `OpenDIKW/dikw-skills` 是否需要升级（不阻塞本 PR）
- [ ] 触动 `server/` HTTP 接口契约？同步检查 `OpenDIKW/dikw-web` 是否需要升级（不阻塞本 PR）

## Notes / breaking changes

<!-- 列出 backward-incompat 的点（on-disk format、CLI surface、HTTP route、storage Protocol、env var、prompt template、配置字段名）。pre-alpha 阶段不一定要保兼容，但要让审查者看到代价。 -->
