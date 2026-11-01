import pandas as pd,numpy as np,matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
OUT='reports/thesis_extras/figures/'
import os;os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':'#e6e6e6','axes.axisbelow':True,'axes.titlesize':10.5,'axes.titlelocation':'left','figure.dpi':200,'savefig.bbox':'tight'})
C={'lexical':'#2a78d6','dense':'#eb6834','hybrid':'#1baf7a'};L={'lexical':'Lexical','dense':'Dense','hybrid':'Hybrid'}
M=['lexical','dense','hybrid']
s=pd.read_csv('reports/results_summary.csv');p=pd.read_csv('reports/results_per_query.csv')
def g(k,m,x):
    r=s[(s.kb_variant==k)&(s.retrieval_method==m)&(s.metric==x)].iloc[0];return r['mean'],r.ci95_low,r.ci95_high
# 5.1
fig,ax=plt.subplots(1,4,figsize=(7.4,2.5),sharey=True)
for a,(x,t) in zip(ax,[('precision_at_k_all','Precision@5'),('recall_at_k_all','Recall@5'),('mrr_all','MRR'),('ndcg_at_k_all','NDCG@5')]):
    for i,mm in enumerate(M):
        v,l,h=g('curated',mm,x);a.bar(i,v,.62,color=C[mm]);a.errorbar(i,v,[[v-l],[h-v]],color='#222',capsize=3,lw=1)
        a.text(i,h+.02,f'{v:.2f}',ha='center',fontsize=9)
    a.set_title(t);a.set_xticks(range(3),[L[q] for q in M],fontsize=8.5);a.grid(axis='x',visible=False);a.set_ylim(0,1.05)
ax[0].set_ylabel('Mean over 150 queries');fig.tight_layout();fig.savefig(OUT+'f51.png');plt.close()
# 5.2 outcome stacked + reference agreement
c=p[p.kb_variant=='curated']
cols=['Lexical','Dense','Hybrid','Oracle','Random']; keys=['lexical','dense','hybrid','oracle','random']
fig,ax=plt.subplots(1,3,figsize=(7.4,2.6))
for a,(x,t) in zip(ax,[('ndcg_at_k_all','NDCG@5'),('faithfulness','Faithfulness'),('reference_agreement','Reference agreement')]):
    d=[c[c.retrieval_method==k][x].dropna() for k in M]
    b=a.boxplot(d,widths=.55,patch_artist=True,showmeans=True,medianprops=dict(color='#222'),meanprops=dict(marker='D',markerfacecolor='white',markeredgecolor='#222',markersize=5),flierprops=dict(marker='o',markersize=2.5,alpha=.5))
    for pa,k in zip(b['boxes'],M):pa.set_facecolor(C[k])
    a.set_xticks([1,2,3],[f'{L[k]}\n(n={len(q)})' for k,q in zip(M,d)],fontsize=8.5);a.set_title(t);a.grid(axis='x',visible=False);a.set_ylim(-.04,1.06)
fig.tight_layout();fig.savefig(OUT+'f52.png');plt.close()
# 6.1
fig,ax=plt.subplots(1,3,figsize=(7.4,2.6))
K=['curated','mixed','reduced']
for a,(x,t) in zip(ax,[('ndcg_at_k_all','NDCG@5'),('reference_agreement','Reference agreement'),('abstained','Abstention rate')]):
    for j,mm in enumerate(M):
        for i,k in enumerate(K):
            v,l,h=g(k,mm,x);a.bar(i+(j-1)*.26,v,.24,color=C[mm],label=L[mm] if i==0 else None);a.errorbar(i+(j-1)*.26,v,[[v-l],[h-v]],color='#222',capsize=2,lw=.8)
    a.set_title(t);a.set_xticks(range(3),['Curated','Mixed','Reduced']);a.grid(axis='x',visible=False)
ax[0].set_ylim(0,1);ax[1].set_ylim(0,1);ax[2].set_ylim(0,.4)
h,l=ax[0].get_legend_handles_labels();fig.legend(h,l,ncol=3,loc='lower center',frameon=False,bbox_to_anchor=(.5,-.07));fig.tight_layout();fig.savefig(OUT+'f61.png');plt.close()
# 6.2 robustness
r=pd.read_csv('reports/robustness_summary.csv')
fig,ax=plt.subplots(1,4,figsize=(7.4,2.7),sharey=True)
PC={'control_rerun':('#9a9a94','Control re-run'),'inject_noise':('#eb6834','Noise replaces a chunk'),'drop_top_relevant':('#2a78d6','Top relevant chunk dropped')}
for a,(x,t) in zip(ax,[('abstained_delta','Abstention'),('reference_agreement_delta','Reference agreement'),('context_relevance_delta','Context relevance'),('faithfulness_delta','Faithfulness')]):
    for j,(pp,(col,lab)) in enumerate(PC.items()):
        for i,k in enumerate(K):
            q=r[(r.kb_variant==k)&(r.perturbation==pp)&(r.metric==x)].iloc[0]
            yy=2-i+(1-j)*.24
            a.errorbar(q['mean'],yy,xerr=[[q['mean']-q.ci95_low],[q.ci95_high-q['mean']]],fmt='o',color=col,ms=4,capsize=2,lw=1,label=lab if i==0 else None)
    a.axvline(0,color='#333',lw=.8);a.set_title(t);a.grid(axis='y',visible=False);a.set_xlim(-.5,.5);a.set_xticks([-.4,0,.4])
    a.set_xlabel('Change from baseline',fontsize=8.5)
ax[0].set_yticks([2,1,0],['Curated','Mixed','Reduced'])
h,l=ax[0].get_legend_handles_labels();fig.legend(h,l,ncol=3,loc='lower center',frameon=False,bbox_to_anchor=(.5,-.08));fig.tight_layout();fig.savefig(OUT+'f62.png');plt.close()
# 7.1 heatmap
cc=pd.read_csv('reports/retrieval_generation_correlations.csv');cc=cc[cc.scope=='pooled']
R=[('precision_at_k_all','Precision@5'),('recall_at_k_all','Recall@5'),('mrr_all','MRR'),('ndcg_at_k_all','NDCG@5')]
G=[('faithfulness','Faithfulness'),('answer_relevance','Answer\nrelevance'),('context_relevance','Context\nrelevance'),('reference_agreement','Reference\nagreement'),('abstained','Abstained')]
A=np.array([[cc[(cc.retrieval_metric==a)&(cc.generation_metric==b)].spearman_rho.iloc[0] for b,_ in G] for a,_ in R])
from matplotlib.colors import LinearSegmentedColormap as LS
cm=LS.from_list('d',['#eb6834','#f2efe9','#2a78d6'])
fig,a=plt.subplots(figsize=(6.2,2.6));a.grid(False);im=a.imshow(A,cmap=cm,vmin=-1,vmax=1,aspect='auto')
for i in range(4):
    for j in range(5):a.text(j,i,f'{A[i,j]:.2f}'.replace('-','\u2212'),ha='center',va='center',fontsize=10,color='#111')
a.set_xticks(range(5),[b for _,b in G]);a.set_yticks(range(4),[b for _,b in R]);a.tick_params(length=0)
for sp in a.spines.values():sp.set_visible(False)
fig.colorbar(im,ax=a,label="Spearman's \u03c1",shrink=.9);fig.savefig(OUT+'f71.png');plt.close()
# 7.2 hit vs miss
real=p[p.retrieval_method.isin(M)].copy();real['hit']=real.mrr_all>0
vals=[('faithfulness','Faithfulness'),('answer_relevance','Answer\nrelevance'),('reference_agreement','Reference\nagreement'),('abstained','Abstention\nrate')]
fig,ax=plt.subplots(figsize=(6.2,2.5));w=.36
for i,(x,t) in enumerate(vals):
    for j,(hh,col,lab) in enumerate([(True,'#2a78d6','At least one relevant chunk in the top 5'),(False,'#eb6834','No relevant chunk in the top 5')]):
        v=real[real.hit==hh][x].mean();ax.bar(i+(j-.5)*w,v,w*.92,color=col,label=lab if i==0 else None);ax.text(i+(j-.5)*w,v+.02,f'{v:.2f}',ha='center',fontsize=9)
ax.set_xticks(range(4),[t for _,t in vals]);ax.set_ylim(0,1.15);ax.grid(axis='x',visible=False);ax.set_ylabel('Mean');ax.legend(frameon=False,loc='upper center',ncol=2,bbox_to_anchor=(.5,1.2));fig.savefig(OUT+'f72.png');plt.close()
h=real[real.hit]
for gm in ['faithfulness','reference_agreement','context_relevance','answer_relevance']:
    d=h[['ndcg_at_k_all','mrr_all','recall_at_k_all',gm]].dropna();print(gm,len(d),[round(spearmanr(d[x],d[gm])[0],3) for x in ['ndcg_at_k_all','mrr_all','recall_at_k_all']])
a=pd.read_csv('reports/results_aggregated.csv');print(a[(a.retrieval_method=='dense')][['kb_variant','ndcg_at_k','mrr','recall_at_k','precision_at_k','partial']])
