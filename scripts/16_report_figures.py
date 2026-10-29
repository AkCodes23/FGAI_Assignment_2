#!/usr/bin/env python
"""Regenerate report figures and numeric summaries from saved assignment artifacts.

This script does not load models, train, infer, or change evaluation metric files.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import to_rgb
from sacrebleu.metrics import CHRF

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
DATA_OUT = RESULTS / "report_figure_data.json"
IDS = ["basic_transformer", "mt5_base", "llama_prompt3shot", "gemma_prompt3shot"]
LABELS = {
    "basic_transformer": "Basic Transformer (from scratch)",
    "mt5_base": "mT5-small (fine-tuned)",
    "llama_prompt3shot": "TinyLlama 1.1B (3-shot prompt)",
    "gemma_prompt3shot": "Gemma 3 1B (3-shot prompt)",
}
PLOT_LABELS = {
    "basic_transformer": "Basic Transformer",
    "mt5_base": "mT5-small",
    "llama_prompt3shot": "TinyLlama",
    "gemma_prompt3shot": "Gemma 3",
}
COLORS = {
    "basic_transformer": "#355C7D",
    "mt5_base": "#6C5B7B",
    "llama_prompt3shot": "#F08A5D",
    "gemma_prompt3shot": "#2A9D8F",
}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.facecolor": "#FBFAF7", "axes.facecolor": "#FBFAF7",
    "axes.edgecolor": "#555555", "axes.spines.top": False,
    "axes.spines.right": False, "grid.color": "#D8D5CE",
    "grid.alpha": 0.65, "savefig.facecolor": "#FBFAF7",
    "svg.hashsalt": "bhashabridge-report-figures",
})


def read_json(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def jsonl(path: str):
    return [json.loads(line) for line in (ROOT / path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, name: str, svg: bool = False):
    fig.savefig(FIG / f"{name}.png", dpi=190, bbox_inches="tight")
    if svg:
        svg_path = FIG / f"{name}.svg"
        fig.savefig(svg_path, bbox_inches="tight", metadata={"Date": None})
        svg_text = svg_path.read_text(encoding="utf-8")
        svg_path.write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


def wrap_label(s: str, width: int = 19) -> str:
    words, lines, line = s.split(), [], ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if len(candidate) > width and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return "\n".join(lines)


def read_system_rows():
    table = {}
    with (RESULTS / "summary_table.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            table[row["tag"]] = row
    return table


def figure_pipeline():
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3.5); ax.axis("off")
    nodes = [
        (0.25, 1.2, 1.75, 1.05, "IITB parallel\ncorpus"),
        (2.45, 1.2, 1.75, 1.05, "Clean, deduplicate,\npreserve split"),
        (4.65, 1.2, 2.05, 1.05, "Four systems\n2 encoder–decoder\n2 decoder-only"),
        (7.15, 1.2, 1.75, 1.05, "Shared 200-row\ntest sample\n(seed 42)"),
        (9.3, 1.2, 2.55, 1.05, "Metrics + review flags\nBLEU / chrF / TER\nscript, number, repetition"),
    ]
    for x, y, w, h, text in nodes:
        box = plt.Rectangle((x, y), w, h, facecolor="#E9EEF2", edgecolor="#355C7D", linewidth=1.6, joinstyle="round")
        ax.add_patch(box); ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=9.5, color="#202A32")
    for (x, y, w, h, _), (nx, ny, _, _, _) in zip(nodes[:-1], nodes[1:]):
        ax.annotate("", xy=(nx-0.08, y+h/2), xytext=(x+w+0.08, y+h/2), arrowprops={"arrowstyle": "-|>", "lw": 1.7, "color": "#555555"})
    ax.text(6, 3.0, "One fixed sample keeps the headline comparison paired", ha="center", fontsize=13, weight="bold", color="#203040")
    ax.text(6, 0.55, "The sample is drawn once from 2,507 official test rows; every system is scored on the same source–reference pairs.", ha="center", fontsize=9, color="#444444")
    save(fig, "pipeline_flow", svg=True)


def figure_architecture():
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.5); ax.axis("off")
    ax.text(3, 6.05, "Encoder-decoder", ha="center", fontsize=13, weight="bold", color="#203040")
    ax.text(9, 6.05, "Decoder-only, 3-shot", ha="center", fontsize=13, weight="bold", color="#203040")
    def box(x,y,w,h,label,color="#E9EEF2",edge="#355C7D",size=9):
        ax.add_patch(plt.Rectangle((x,y),w,h,facecolor=color,edgecolor=edge,lw=1.5))
        ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=size,color="#202A32",linespacing=1.15)
    box(0.35,4.2,1.7,0.8,"English source\n+ task prefix")
    box(2.25,4.2,1.55,0.8,"Encoder\nattention")
    box(4.1,4.2,1.55,0.8,"Decoder\ncross-attention")
    box(2.55,2.7,1.2,0.7,"Hindi output",color="#E3F1EC",edge="#2A9D8F")
    for start,end in [((2.05,4.6),(2.25,4.6)),((3.8,4.6),(4.1,4.6)),((4.9,4.2),(3.2,3.4))]:
        ax.annotate("",xy=end,xytext=start,arrowprops={"arrowstyle":"-|>","lw":1.5,"color":"#555"})
    box(6.3,4.45,5.35,1.25,"System\nYou are a careful professional translator.\nTranslate English sentences into natural Hindi.\nPreserve meaning, names and numbers.\nNever add explanations, transliteration, notes or quotes.",color="#F2EAF4",edge="#6C5B7B",size=7.0)
    box(6.3,2.75,5.35,1.4,"Repeat this user/assistant pair for 3 saved examples.\nUser: Translate the following English sentence into Hindi.\nReply with the Hindi translation only.\nEnglish: {src}\nHindi:\nAssistant: {saved Hindi target}",color="#F2EAF4",edge="#6C5B7B",size=7.0)
    box(6.3,1.45,5.35,1.05,"Then send the same template with the test source.\nThe model continues after 'Hindi:'. Greedy decoding\nis used for headline scores.",color="#E3F1EC",edge="#2A9D8F",size=8)
    box(6.3,0.55,5.35,0.62,"Decoder-only Transformer\n(causal self-attention) generates the Hindi continuation",color="#E9EEF2",edge="#355C7D",size=8.5)
    ax.annotate("",xy=(8.98,4.17),xytext=(8.98,4.40),arrowprops={"arrowstyle":"-|>","lw":1.5,"color":"#555"})
    ax.annotate("",xy=(8.98,2.55),xytext=(8.98,2.72),arrowprops={"arrowstyle":"-|>","lw":1.5,"color":"#555"})
    ax.annotate("",xy=(8.98,1.20),xytext=(8.98,1.43),arrowprops={"arrowstyle":"-|>","lw":1.5,"color":"#555"})
    ax.text(3,1.65,"Basic: 15.2M, no pretraining, 8k SentencePiece vocab\nmT5: 300.2M, mC4 pretraining, 250,112 vocab",ha="center",fontsize=8.7,color="#444")
    ax.text(9,0.27,"TinyLlama: 1.100B params  |  Gemma 3: 0.9999B params; no local weight updates",ha="center",fontsize=8.1,color="#444")
    ax.text(6,0.08,"Boxes show the evaluated data path and prompt structure; model internals are simplified.",ha="center",fontsize=8.0,color="#666")
    save(fig,"architecture_comparison",svg=True)

def figure_losses():
    basic = [r for r in jsonl("logs/train_basic_reduced.jsonl") if r.get("event") == "epoch"]
    hist = [r for r in jsonl("logs/train_mt5.jsonl") if r.get("event") == "epoch"]
    fig, axs = plt.subplots(1,2,figsize=(11,4.2))
    for ax, rows, label, key, color in [
        (axs[0], basic, "Basic Transformer", "train_loss", COLORS["basic_transformer"]),
        (axs[1], hist, "mT5-small", "train_loss", COLORS["mt5_base"]),
    ]:
        ep=[r["epoch"] for r in rows]; tr=[r["train_loss"] for r in rows]; va=[r["val_loss"] for r in rows]
        ax.plot(ep,tr,"o-",color=color,label="Train",lw=2,ms=4)
        ax.plot(ep,va,"s--",color="#E76F51",label="Validation",lw=2,ms=4)
        ax.set_title(f"{label}: training and validation loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss"); ax.grid(axis="y")
        if label == "mT5-small":
            ax.set_xticks([1, 2, 3])
        else:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        best_i=int(np.argmin(va))
        if label == "mT5-small":
            ax.annotate(f"best {va[best_i]:.3f}",(ep[best_i],va[best_i]),xytext=(-8,8),textcoords="offset points",ha="right",fontsize=8)
        else:
            ax.annotate(f"best {va[best_i]:.3f}",(ep[best_i],va[best_i]),xytext=(7,8),textcoords="offset points",fontsize=8)
        if label == "Basic Transformer":
            ax.axvline(ep[best_i], color="#2A9D8F", linestyle=":", linewidth=1.7, label="Best validation epoch (22)")
            ax.axvline(ep[-1], color="#6B7280", linestyle="--", linewidth=1.4, label="Early stop (26)")
            ax.legend(frameon=False, fontsize=7, loc="best")
        else:
            ax.legend(frameon=False)
    fig.tight_layout(); save(fig,"training_curves")


def chrf_sentence_bootstrap(values, n_boot=400, seed=42):
    rng=np.random.default_rng(seed); values=np.asarray(values,dtype=float)
    indexes=rng.integers(0,len(values),size=(n_boot,len(values)))
    samples=values[indexes].mean(axis=1)
    return float(np.percentile(samples,2.5)),float(np.percentile(samples,97.5)),samples

def sentence_chrf(rows):
    metric=CHRF(word_order=0)
    return np.array([metric.sentence_score(r["hyp"],[r["ref"]]).score for r in rows],dtype=float)


def figure_scores(table, translations, ci, stats):
    fig, axs=plt.subplots(1,2,figsize=(12,5.0))
    for ax,key,metric_name,lim in [(axs[0],"bleu","BLEU",None),(axs[1],"chrf","chrF",None)]:
        for i,tag in enumerate(IDS):
            value=float(table[tag][key])
            if key=="bleu":
                value=float(table[tag][key]); lo=float(ci[tag]["bleu"]["lo"]); hi=float(ci[tag]["bleu"]["hi"])
            else:
                value=stats[tag]["sentence_chrf_mean"]; lo,hi=stats[tag]["sentence_mean_chrf_ci95"]
            ax.errorbar(value,i,xerr=[[value-lo],[hi-value]],fmt="o",color=COLORS[tag],capsize=4,markersize=7,lw=2)
            ax.text(hi+max(0.3,(hi-lo)*0.08),i,f"{value:.2f} [{lo:.2f}, {hi:.2f}]",va="center",fontsize=8,color="#333")
        ax.set_yticks(range(len(IDS)),[wrap_label(LABELS[t]) for t in IDS]); ax.invert_yaxis()
        ax.set_xlabel(("Corpus BLEU" if key=="bleu" else "Mean sentence chrF")+" (higher is better)")
        ax.set_title("BLEU: corpus 95% bootstrap interval" if key=="bleu" else "chrF: sentence-mean 95% bootstrap interval")
        ax.grid(axis="x")
    fig.suptitle("Gemma leads corpus BLEU and mean sentence-level chrF on the fixed sample",fontsize=13,weight="bold")
    fig.tight_layout(rect=(0,0,1,.93)); save(fig,"score_intervals")


def figure_quality(table):
    fig,axs=plt.subplots(1,2,figsize=(11.5,4.8))
    for ax,xkey,xlabel,ticks in [(axs[0],"inference_s","Inference time for 200 rows (s)",[10,30,100,300,600]),(axs[1],"params_m","Parameters (millions; full model count)",[10,100,1000])]:
        xs=[float(table[t][xkey]) for t in IDS]
        for tag,x in zip(IDS,xs):
            row=table[tag]; y=float(row["chrf"])
            ax.scatter(x,y,s=95,color=COLORS[tag],edgecolor="white",linewidth=0.8,zorder=3)
            if tag=="gemma_prompt3shot":
                offset=(-8,-8); align="right"; valign="top"
            elif tag=="llama_prompt3shot":
                offset=(-8,8); align="right"; valign="bottom"
            else:
                offset=(7,7); align="left"; valign="bottom"
            ax.annotate(PLOT_LABELS[tag],(x,y),xytext=offset,textcoords="offset points",ha=align,va=valign,fontsize=8,color=COLORS[tag])
        ax.set_xscale("log"); ax.set_xticks(ticks); ax.set_xticklabels([str(x) for x in ticks]); ax.minorticks_off()
        ax.set_xlim(min(xs)*0.72,max(xs)*1.42)
        ax.set_ylabel("Corpus chrF (higher is better)"); ax.set_xlabel(xlabel); ax.grid(True,axis="both")
    axs[0].set_title("Inference time separates speed from match quality")
    axs[1].set_title("Parameter count alone does not predict score")
    fig.tight_layout(); save(fig,"quality_cost")

def figure_sentence_distribution(table, scores):
    fig,ax=plt.subplots(figsize=(9.5,4.8))
    values=[scores[t]["sentence_chrf_values"] for t in IDS]
    bp=ax.boxplot(values,patch_artist=True,showfliers=False,showmeans=True,meanline=False,medianprops={"color":"#222","linewidth":1.5})
    for patch,t in zip(bp["boxes"],IDS): patch.set_facecolor(COLORS[t]); patch.set_alpha(.75)
    for m,t in zip(bp["means"],IDS): m.set_markerfacecolor(COLORS[t]); m.set_markeredgecolor("white")
    ax.set_xticks(range(1,len(IDS)+1),[PLOT_LABELS[t] for t in IDS]); ax.set_ylabel("Sentence chrF (0–100)")
    ax.set_title("Gemma has the highest median sentence match; all systems have a wide tail")
    ax.grid(axis="y"); fig.tight_layout(); save(fig,"sentence_chrf_distribution")


def figure_length_bucket(scores):
    buckets=["1–10","11–20","21–30","31+"]
    fig,ax=plt.subplots(figsize=(9.5,4.8))
    for tag in IDS:
        vals=scores[tag]["source_length_buckets"]
        means=[vals[b]["mean_chrf"] for b in buckets]
        ax.plot(range(len(buckets)),means,"o-",lw=2,ms=6,color=COLORS[tag],label=PLOT_LABELS[tag])
    counts=[scores[IDS[0]]["source_length_buckets"][b]["n"] for b in buckets]
    ax.set_xticks(range(len(buckets)),[f"{b}\n(n={n})" for b,n in zip(buckets,counts)])
    ax.set_xlabel("English source length (whitespace words)"); ax.set_ylabel("Mean sentence chrF")
    ax.set_title("Gemma's chrF rises most across source-length buckets",fontsize=12,weight="bold")
    ax.grid(axis="y")
    fig.legend(frameon=False,fontsize=8,ncol=2,loc="lower center",bbox_to_anchor=(.5,.015))
    fig.subplots_adjust(top=.87,bottom=.27,left=.1,right=.98); save(fig,"chrf_by_source_length")


def figure_risk(risk):
    flags=[("repetition_loops","Repetition"),("dropped_number_sentences","Dropped number"),("added_number_sentences","Added number"),("untranslated_latin_sentences","Latin text"),("no_devanagari","No Devanagari"),("length_ratio_gt2","Output >2×"),("length_ratio_lt0.5","Output <0.5×")]
    arr=np.array([[100*risk[tag][key] for key,_ in flags] for tag in IDS])
    fig,ax=plt.subplots(figsize=(11,4.3)); im=ax.imshow(arr,cmap="YlOrRd",vmin=0,vmax=50,aspect="auto")
    ax.set_xticks(range(len(flags)),[lab for _,lab in flags],rotation=24,ha="right")
    ax.set_yticks(range(len(IDS)),[wrap_label(LABELS[t]) for t in IDS])
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            r,g,b=to_rgb(im.cmap(im.norm(arr[i,j])))
            luminance=0.2126*r+0.7152*g+0.0722*b
            ax.text(j,i,f"{arr[i,j]:.1f}%",ha="center",va="center",fontsize=8,color=("#202020" if luminance>0.55 else "#FFFFFF"),weight="bold")
    ax.set_title("Output review flags differ by system; they are not semantic error rates")
    cb=fig.colorbar(im,ax=ax,pad=.02); cb.set_label("Flagged outputs (%)")
    fig.tight_layout(); save(fig,"risk_flag_heatmap")


def figure_fertility(fertility):
    keys=["basic_transformer_sentencepiece","google/mt5-small","TinyLlama/TinyLlama-1.1B-Chat-v1.0","unsloth/gemma-3-1b-it"]
    names=["Basic Transformer","mT5-small","TinyLlama","Gemma 3"]
    ratios=[fertility[k]["hi_tokens_per_word"] for k in keys]
    colors=[COLORS[t] for t in IDS]
    fig,ax=plt.subplots(figsize=(10,4.8))
    bars=ax.bar(range(len(keys)),ratios,color=colors,width=.62)
    for b,k in zip(bars,keys):
        d=fertility[k]; ax.text(b.get_x()+b.get_width()/2,b.get_height()+.13,f"{d['hi_tokens_per_word']:.2f} pieces/word\n{d['hi_tokens_mean']:.2f} pieces/reference",ha="center",fontsize=8)
    ax.set_xticks(range(len(keys)),names)
    ax.set_ylabel("Hindi tokenizer pieces per whitespace word"); ax.set_ylim(0,max(ratios)*1.32)
    ax.set_title("Tokenizer fertility on the same 2,507 Hindi test references")
    ax.grid(axis="y"); fig.tight_layout(); save(fig,"tokenizer_fertility")


def figure_temperature(temp):
    ts=[r["temperature"] for r in temp["results"]["llama_prompt3shot"]]
    fig,axs=plt.subplots(1,3,figsize=(12,4.2))
    for ax,field,title in [(axs[0],"bleu","BLEU"),(axs[1],"chrf","chrF"),(axs[2],"distinct_2","Distinct-2")]:
        for tag in ["llama_prompt3shot","gemma_prompt3shot"]:
            vals=[r[field] for r in temp["results"][tag]]
            ax.plot(ts,vals,"o-",lw=2,ms=5,color=COLORS[tag],label=LABELS[tag])
        ax.set_title(title); ax.set_xlabel("Temperature"); ax.set_xticks(ts); ax.grid(axis="y")
    axs[0].set_ylabel("Corpus score"); axs[1].set_ylabel("Corpus score"); axs[2].set_ylabel("Unique bigram ratio")
    axs[0].legend(frameon=False,fontsize=7)
    fig.suptitle("Temperature raises TinyLlama diversity; Gemma's match scores fall",fontsize=13,weight="bold")
    fig.tight_layout(rect=(0,0,1,.91)); save(fig,"temperature_analysis")


def figure_paired_bootstrap(paired):
    fig,ax=plt.subplots(figsize=(11.5,6.2))
    items=paired
    short={"basic_transformer":"Basic Transformer","mt5_base":"mT5-small","llama_prompt3shot":"TinyLlama","gemma_prompt3shot":"Gemma 3"}
    y=np.arange(len(items))
    for yi,r in zip(y,items):
        lo,hi=r["ci_lo"],r["ci_hi"]; val=r["delta_bleu"]
        ax.errorbar(val,yi,xerr=[[val-lo],[hi-val]],fmt="o",capsize=4,lw=2,color=COLORS[r["a"]])
        ax.text(hi+.12,yi,f"Δ={val:.3f} [{lo:.3f}, {hi:.3f}]",va="center",fontsize=8)
    ax.axvline(0,color="#D1495B",lw=1.5,ls="--")
    ax.set_yticks(y,[f"{short[r['a']]} − {short[r['b']]}" for r in items]); ax.invert_yaxis()
    ax.set_xlim(-13.8,11.5)
    ax.set_xlabel("Paired corpus BLEU difference (A − B)"); ax.set_title("Five paired BLEU intervals exclude zero; Basic Transformer vs TinyLlama overlaps")
    ax.grid(axis="x"); fig.tight_layout(); save(fig,"paired_bleu_bootstrap")


def main():
    FIG.mkdir(parents=True,exist_ok=True)
    table=read_system_rows(); ci=read_json("results/confidence_intervals.json")
    risk=read_json("results/error_analysis/risk_flags.json")
    temp=read_json("results/temperature/temperature_results.json")
    fertility=read_json("results/tokenizer_fertility.json")
    paired=read_json("results/paired_bootstrap.json")
    trans={tag:jsonl(f"results/translations/{tag}.jsonl") for tag in IDS}
    indexes={tag:{r["i"]:r for r in trans[tag]} for tag in IDS}
    common_ids=sorted(set.intersection(*(set(indexes[t]) for t in IDS)))
    if len(common_ids)!=200: raise ValueError(f"Expected 200 aligned rows, got {len(common_ids)}")
    refs=[indexes[IDS[0]][i]["ref"] for i in common_ids]
    if any(indexes[t][i]["ref"] != refs[k] for t in IDS for k,i in enumerate(common_ids)):
        raise ValueError("Reference texts differ across system files")
    rng=np.random.default_rng(42); stats={}; all_scores={}
    for tag in IDS:
        rows=[indexes[tag][i] for i in common_ids]
        vals=sentence_chrf(rows)
        lo,hi,boot=chrf_sentence_bootstrap(vals)
        buckets={"1–10":[],"11–20":[],"21–30":[],"31+":[]}
        for row,score in zip(rows,vals):
            wc=len(row["src"].split())
            name="1–10" if wc<=10 else "11–20" if wc<=20 else "21–30" if wc<=30 else "31+"
            buckets[name].append(float(score))
        bucket_stats={name:{"n":len(v),"mean_chrf":round(float(np.mean(v)),3) if v else None} for name,v in buckets.items()}
        stats[tag]={"sentence_mean_chrf_ci95":[round(lo,3),round(hi,3)],"sentence_chrf_mean":round(float(np.mean(vals)),3),"sentence_chrf_median":round(float(np.median(vals)),3),"sentence_chrf_q25":round(float(np.percentile(vals,25)),3),"sentence_chrf_q75":round(float(np.percentile(vals,75)),3),"source_length_buckets":bucket_stats}
        all_scores[tag]={"sentence_chrf_values":[round(float(x),3) for x in vals],"source_length_buckets":bucket_stats,"chrf_bootstrap_sample_summary":{"mean":round(float(np.mean(boot)),3),"sd":round(float(np.std(boot,ddof=1)),3)}}
    figure_pipeline(); figure_architecture(); figure_losses(); figure_scores(table,trans,ci,stats)
    figure_quality(table); figure_sentence_distribution(table,all_scores); figure_length_bucket(all_scores)
    figure_risk(risk); figure_fertility(fertility); figure_temperature(temp); figure_paired_bootstrap(paired)
    input_files=["results/summary_table.csv","results/confidence_intervals.json","results/paired_bootstrap.json","results/temperature/temperature_results.json","results/tokenizer_fertility.json","results/error_analysis/risk_flags.json","logs/train_basic_reduced.jsonl","logs/train_mt5.jsonl"]
    input_files += [f"results/translations/{tag}.jsonl" for tag in IDS]
    payload={
        "schema_version":1,"generated_by":"scripts/16_report_figures.py","sample_rows":len(common_ids),"sample_ids_sha256":hashlib.sha256(json.dumps(common_ids,separators=(",",":")).encode()).hexdigest(),"bootstrap_resamples":400,"bootstrap_seed":42,
        "bootstrap_method":"Percentile interval from 400 with-replacement bootstrap means of sentence-level chrF on the same 200 paired rows; chrF word_order=0.",
        "systems":{tag:{"label":LABELS[tag],"sentence_mean_chrf_ci95":stats[tag]["sentence_mean_chrf_ci95"],"sentence_chrf_mean":stats[tag]["sentence_chrf_mean"],"sentence_chrf_median":stats[tag]["sentence_chrf_median"],"sentence_chrf_q25":stats[tag]["sentence_chrf_q25"],"sentence_chrf_q75":stats[tag]["sentence_chrf_q75"],"source_length_buckets":stats[tag]["source_length_buckets"],"sentence_chrf_values":all_scores[tag]["sentence_chrf_values"]} for tag in IDS},
        "paired_bleu_bootstrap_source":"results/paired_bootstrap.json","paired_bleu_bootstrap":paired,
        "tokenizer_fertility_source":"results/tokenizer_fertility.json","tokenizer_fertility":fertility,
        "input_sha256":{name:sha256(ROOT/name) for name in input_files},
        "figures":[f"results/figures/{name}.png" for name in ["pipeline_flow","architecture_comparison","training_curves","score_intervals","quality_cost","sentence_chrf_distribution","chrf_by_source_length","risk_flag_heatmap","tokenizer_fertility","temperature_analysis","paired_bleu_bootstrap"]],
    }
    DATA_OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"figures":payload["figures"],"derived_json":"results/report_figure_data.json","sample_rows":len(common_ids),"sentence_mean_chrf_ci95":{t:stats[t]["sentence_mean_chrf_ci95"] for t in IDS},"source_buckets":{t:stats[t]["source_length_buckets"] for t in IDS},"input_sha256":payload["input_sha256"]},ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()
