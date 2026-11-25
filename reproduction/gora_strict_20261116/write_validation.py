"""Generate the user-facing record only from completed runtime evidence."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def main():
    summaries = {}
    lines = ['# 实际验证记录', '', '日期：2026-11-16。仅完成短训和推理验证，不代表完整 GSM8K 成绩已复现。', '',
             '正式训练未自动启动。使用的 GPU 均为0/1；2/3上的任务未参与本复现。', '']
    for profile in ('author','paper'):
        run = HERE / 'validation_runs' / f'{profile}-seed42-smoke'
        names = {'init':'initialization.rank0.json', 'train':'training_complete.json',
                 'merge':'load_merge_validation.json', 'engine':'engine_smoke/protocol.json'}
        reports = {k: json.loads((run/v).read_text()) for k,v in names.items() if (run/v).exists()}
        complete = len(reports) == len(names) and all(v.get('status')=='PASS' for v in reports.values())
        if complete:
            assert reports['engine']['hf_vllm_same_first_tokens_on_synthetic_prompts'] is True
        summary = {'status':'PASS' if complete else 'INCOMPLETE', 'evidence':reports}
        summaries[profile] = summary
        lines += [f'## {profile}', '', f'状态：{summary["status"]}', '']
        if 'init' in reports:
            r=reports['init']
            values=[x['gradient_update_inner_product'] for x in r['layers']]
            lines += [f'- 真实8B模型：{r["modules"]}个投影、{r["adapter_tensors"]}个A/B张量，共{r["adapter_parameters"]:,}个adapter参数。',
                      f'- 每层实际梯度累计次数范围：{r["gradient_count_range"]}；独立rank公式：{r.get("independent_rank_formula_check","未记录")}。',
                      f'- 两卡初始A/B哈希完全一致：{len(set(r["same_initial_adapter_hash_on_two_gpus"]))==1}。',
                      f'- 梯度与初始化更新内积范围：[{min(values):.8g}, {max(values):.8g}]。',
                      f'- 固定训练探针loss：base={r["probe_loss"]["base"]:.9f}，初始化后={r["probe_loss"]["initialized"]:.9f}，gamma缩小1000倍后={r["probe_loss"]["gamma_times_0_001"]:.9f}。仅作方向/数值诊断，不据此挑参数。']
        if 'train' in reports:
            r=reports['train']
            lines += [f'- 原作者Trainer真实训练：{r["micro_steps"]}个micro steps、{r["optimizer_steps"]}次optimizer更新、{r["changed_adapter_tensors"]}个张量全部产生有限非零变化。']
        if 'merge' in reports:
            r=reports['merge']
            lines += [f'- 严格加载{r["loaded_adapter_tensors"]}个adapter张量，值逐项相等；FP32逐层合并代数最大相对误差{r["fp32_layer_algebra_max_relative_error"]:.8g}。',
                      f'- 完整模型3个探针BF16 logits检查通过；最大RMSE={max(x["rmse"] for x in r["full_model_bf16_logit_checks"]):.8g}，3个next-token一致。',
                      f'- 导出后逐张量核对：{r.get("export_parameter_tensors_verified","未导出")}个参数张量。']
        if 'engine' in reports:
            r=reports['engine']
            lines += [f'- vLLM {r["vllm"]}真实TP=2加载并生成{r["examples"]}道合成题，使用0/1卡；同样应用prompt重复惩罚后，HF/vLLM首token逐题一致。']
        if not complete:
            lines += ['- 缺失阶段：'+', '.join(k for k in names if k not in reports)]
        lines += ['']
    lines += ['## 如何理解这些结果', '',
              'PASS表示本地适配版在上述检查中通过，不代表与论文硬件、原始数据抽样、原始自定义Llama/FA2路径完全一致，也不证明能达到72.91%。具体差异和运行命令见 README_zh.md。', '',
              '作者代码里的正号、取整和梯度hook保留在author组；paper组明确修改这些部分。不能把paper组说成未经修改的作者实现。', '',
              'setup/环境阶段失败的尝试保留在带failed后缀的目录，不计入上述通过结果。没有用失败尝试或合成题挑选最终GSM8K解码参数。', '']
    (HERE/'VALIDATION_zh.md').write_text('\n'.join(lines))
    (HERE/'validation_summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(json.dumps({k:v['status'] for k,v in summaries.items()},indent=2))

if __name__ == '__main__':
    main()
