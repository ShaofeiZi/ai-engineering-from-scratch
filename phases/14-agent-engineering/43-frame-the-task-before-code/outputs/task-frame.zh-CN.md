# 任务框架：在注册时阻止重复的邮箱地址

状态：READY

## 仓库事实
- 账户写入使用 AccountStore（`app/accounts.py:18`）
- 重复错误使用状态码 409（`tests/test_accounts.py:44`）

## 允许的路径
- `app/accounts.py`
- `tests/test_accounts.py`

## 禁止的路径
- `migrations/**`
- `deploy/**`

## 验收证据
- `python3 -m unittest tests.test_accounts`

## 未知事项
- 邮箱比较是否区分大小写
