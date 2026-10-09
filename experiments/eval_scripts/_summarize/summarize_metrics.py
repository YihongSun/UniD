import os
import os.path as osp
import re
import argparse

def try_float(s):
    try:
        return float(s)
    except ValueError:
        return s

def load_metrics_txt(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    metrics = {}
    topkey = None
    for l in lines:
        if ":" in l:
            key, value = l.split(":", 1)
            metrics[key.strip()] = try_float(value.strip())
            if topkey is None:
                topkey = key.strip()
    return metrics, topkey

def load_metrics_md(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    headers = []
    values = []
    for line in lines:
        if line.startswith("|") and not line.startswith("|---"):
            parts = [part.strip() for part in line.split("|")[1:-1]]
            if not headers:
                headers = parts
            else:
                values = [try_float(part) for part in parts]
                break
    return dict(zip(headers, values))

def load_metrics_csv(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    headers = []
    values = []
    for line in lines:
        if len(headers) == 0:
            headers = [part.strip() for part in line.split(",")]
        else:
            values = [float(part.strip()) if part.strip().replace('.', '', 1).isdigit() else part.strip() for part in line.split(",")]
            break
    return dict(zip(headers, values))

def parse_args():
    parser = argparse.ArgumentParser(description="Summarize evaluation metrics from multiple files")
    parser.add_argument("--eval_root", type=str, required=True, help="Root directory containing evaluation result folders")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    eval_root = args.eval_root

    task = eval_root.strip("/").split("/")[-2]

    if task == "albedo":
        hypersim_file = osp.join(eval_root, "hypersim_val", "summary.txt")
        hs = load_metrics_md(hypersim_file)

        iiw_file = osp.join(eval_root, "iiw_test", "whdr.txt")
        iiw = load_metrics_md(iiw_file)

        summary_file = osp.join(eval_root, "summary.txt")
        with open(summary_file, "w") as f:
            header = "HS(PSNR),HS(LPIPS),IIW(WHDR10),IIW(WHDR15)"
            values = "{},{},{},{}".format(hs['PSNR (dB)'], hs['LPIPS'], iiw['func10'], iiw['func15'])
            f.write(header + "\n")
            f.write(values + "\n")
    elif task == "illum":
        hypersim_file = osp.join(eval_root, "hypersim_val", "summary.txt")
        hs = load_metrics_md(hypersim_file)

        saw_file = osp.join(eval_root, "saw_test", "pr-test_metrics.txt")
        saw = load_metrics_csv(saw_file)

        summary_file = osp.join(eval_root, "summary.txt")
        with open(summary_file, "w") as f:
            header = "HS(PSNR),HS(LPIPS),SAW(Prec@0.7),SAW(AP)"
            values = "{},{},{},{}".format(hs['PSNR (dB)'], hs['LPIPS'], saw['p@0.7'], saw['AP'])
            f.write(header + "\n")
            f.write(values + "\n")
    elif task == "semseg":
        vip_file = osp.join(eval_root, "VIPSeg_val.csv")
        vip = load_metrics_csv(vip_file)

        ade20k_file = osp.join(eval_root, "ade20k_val", "performance_validation.csv")
        ade20k = load_metrics_csv(ade20k_file)

        cityscapes_file = osp.join(eval_root, "cityscapes_val", "performance_val.csv")
        cityscapes = load_metrics_csv(cityscapes_file)

        with open(osp.join(eval_root, "summary.txt"), "w") as f:
            header = "CS(mIoU),CS(fwIoU),ADE20K(mIoU),ADE20K(fwIoU),VIPSeg(mIoU),VIPSeg(fwIoU),VIPSeg(VC-4)"
            values = "{},{},{},{},{},{},{}".format(
                cityscapes['mIoU'], cityscapes['fwIoU'],
                ade20k['mIoU'], ade20k['fwIoU'],
                vip['mIoU'], vip['fwIoU'], vip['VC-4']
            )
            f.write(header + "\n")
            f.write(values + "\n")
    elif task.startswith("partseg"):
        coco_file = osp.join(eval_root, "coco_val", "partseg_metrics_summary.txt")
        coco = load_metrics_md(coco_file)

        pascal = osp.join(eval_root, "pascal_val", "partseg_metrics_summary.txt")
        pascal = load_metrics_md(pascal)
        
        with open(osp.join(eval_root, "summary.txt"), "w") as f:
            header = "COCO(mIoU),COCO(fwIoU),PASCAL(mIoU),PASCAL(fwIoU)"
            values = "{},{},{},{}".format(
                coco['mIoU'], coco['fwIoU'],
                pascal['mIoU'], pascal['fwIoU']
            )
            f.write(header + "\n")
            f.write(values + "\n")
    elif task.startswith("matseg"):
        dms_file = osp.join(eval_root, "dms_val", "matseg_metrics_summary.txt")
        dms = load_metrics_md(dms_file)

        ade20k = osp.join(eval_root, "ade20k_val", "matseg_metrics_summary.txt")
        ade20k = load_metrics_md(ade20k)
        
        with open(osp.join(eval_root, "summary.txt"), "w") as f:
            header = "DMS(mIoU),DMS(fwIoU),ADE20k(mIoU),ADE20k(fwIoU)"
            values = "{},{},{},{}".format(
                dms['mIoU'], dms['fwIoU'],
                ade20k['mIoU'], ade20k['fwIoU']
            )
            f.write(header + "\n")
            f.write(values + "\n")
    elif task.startswith("video_normal"):
        task_name = task.replace("video_", "")
        vc4_file = osp.join(eval_root, "interiornet_val", f"{task_name}_vc4.txt")
        vc4, vc4_key = load_metrics_txt(vc4_file)
        vc8_file = osp.join(eval_root, "interiornet_val", f"{task_name}_vc8.txt")
        vc8, vc8_key = load_metrics_txt(vc8_file)
        vc16_file = osp.join(eval_root, "interiornet_val", f"{task_name}_vc16.txt")
        vc16, vc16_key = load_metrics_txt(vc16_file)
        perf_file = osp.join(eval_root, "interiornet_val", f"{task_name}_perf.txt")
        perf = load_metrics_csv(perf_file)

        with open(osp.join(eval_root, "summary.txt"), "w") as f:
            header = "mean,median,rmse,a1,a2,a3,a4,a5,VC-4,VC-8,VC-16"
            values = "{},{},{},{},{},{},{},{},{:.2f},{:.2f},{:.2f}".format(perf['mean'], perf['median'], perf['rmse'], perf['a1'], perf['a2'], perf['a3'], perf['a4'], perf['a5'], vc4[vc4_key], vc8[vc8_key], vc16[vc16_key])
            f.write(header + "\n")
            f.write(values + "\n")
    elif task.startswith("video_"):
        task_name = task.replace("video_", "")
        vc4_file = osp.join(eval_root, "interiornet_val", f"{task_name}_vc4.txt")
        vc4, vc4_key = load_metrics_txt(vc4_file)
        vc8_file = osp.join(eval_root, "interiornet_val", f"{task_name}_vc8.txt")
        vc8, vc8_key = load_metrics_txt(vc8_file)
        vc16_file = osp.join(eval_root, "interiornet_val", f"{task_name}_vc16.txt")
        vc16, vc16_key = load_metrics_txt(vc16_file)

        with open(osp.join(eval_root, "summary.txt"), "w") as f:
            header = "VC-4,VC-8,VC-16"
            values = "{:.2f},{:.2f},{:.2f}".format(vc4[vc4_key], vc8[vc8_key], vc16[vc16_key])
            f.write(header + "\n")
            f.write(values + "\n")
    else:
        raise NotImplementedError(f"Task `{task}` not supported for summarization")
