from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1]
    proc=subprocess.run([sys.executable,"-m","pytest","-q","tests/test_download_system.py"],cwd=root,text=True,capture_output=True)
    provider=json.loads((root/"reports/01_stage0_provider_report.json").read_text()) if (root/"reports/01_stage0_provider_report.json").exists() else []
    lines=["# Stage-0 数据下载测试报告", "", "## Mock/底层框架", "", "```text", proc.stdout+proc.stderr, "```", f"退出码：`{proc.returncode}`。", "", "## 真实 provider 小样本", "", "仅运行了配置中的 3 港股×2 年、3 美股×2 年；A 股因 TUSHARE_TOKEN 未设置而跳过。"]
    for r in provider: lines.append(f"- {r.get('provider')} {r.get('symbol','')} {r.get('year','')}: **{r.get('status')}**" + (f"（{r.get('error')}）" if r.get('error') else ""))
    lines += ["", "## 全量下载闸门", "", "未构建、未运行全量任务；`build_download_tasks.py` 的非 dry-run 路径明确返回拒绝。", "", "## 解释", "", "`skipped`/`needs_review` 是本机真实运行结果，不表示 provider 已验证通过。网络/DNS 不可用和 Tushare 凭据缺失必须在后续环境具备后重新探测。"]
    (root/"reports/01_stage0_test_report.md").write_text("\n".join(lines),encoding="utf-8")
    print(root/"reports/01_stage0_test_report.md")
    return proc.returncode
if __name__ == "__main__": raise SystemExit(main())
