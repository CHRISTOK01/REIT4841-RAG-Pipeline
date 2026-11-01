import json,glob,numpy as np,pandas as pd
pd.set_option('display.width',220,'display.max_columns',40,'display.max_rows',200,'display.precision',3)
rng=np.random.default_rng(1)
def ci(x,B=2000):
    x=np.asarray(x,float);x=x[~np.isnan(x)]
    m=rng.choice(x,(B,len(x))).mean(1);return x.mean(),np.percentile(m,2.5),np.percentile(m,97.5),len(x)
p=pd.read_csv('reports/results_per_query.csv')
real=p[p.retrieval_method.isin(['lexical','dense','hybrid'])].copy()
R=[json.loads(l) for l in open('results/full_sweep.jsonl',encoding='utf8')]
st=[r for r in R if r['record_type']=='standard']
cur={c['doc_id'] for r in st if r['kb_variant']=='curated' for c in r['retrieved']}
# noise: use KB chunk files
import os
def docs(v):
    f=glob.glob(f'artifacts/b1bc54c28398/{v}/*chunks*')+glob.glob(f'artifacts/b1bc54c28398/{v}/*.jsonl')
    return f
print(docs('curated'),docs('mixed'))
f=docs('curated')[0]; g=docs('mixed')[0]
cd={json.loads(l)['chunk_id'] for l in open(f,encoding='utf8')}
md=[json.loads(l) for l in open(g,encoding='utf8')]
print('chunks',len(cd),len(md)); noise={c['chunk_id'] for c in md}-cd; print('noise share',len(noise)/len(md))
rows=[]
for r in st:
    if r['kb_variant']=='mixed' and r['retrieval_method'] in('lexical','dense','hybrid','random'):
        ids=[c['chunk_id'] for c in r['retrieved']]; n=sum(i in noise for i in ids)
        rows.append((r['retrieval_method'],n/len(ids) if ids else 0,n>0,ids[0] in noise, sum(c in noise for c in ids[:1])))
d=pd.DataFrame(rows,columns='m share any top1 t'.split());print(d.groupby('m')[['share','any','top1']].mean())
print('--hit/miss');real['hit']=real.mrr_all>0
real['ans']=real.coverage>0
print(real.groupby('hit')[['faithfulness','answer_relevance','context_relevance','reference_agreement','abstained']].agg(['mean','count']).T)
print(real.groupby(['ans','hit'])[['reference_agreement','abstained','faithfulness']].mean(), real.groupby(['ans','hit']).size())
print('--curated by method hit');c=real[real.kb_variant=='curated']
print(c.groupby(['retrieval_method','hit'])[['reference_agreement','abstained','faithfulness']].agg('mean'),c.groupby(['retrieval_method','hit']).size())
print('--ceiling');f=real.faithfulness.dropna();print((f==1).mean(),f.std(),real.answer_relevance.std(),real.answer_relevance.quantile([.05,.5,.95]).tolist(),(real.reference_agreement.dropna()==1).mean())
print('--wins NDCG curated');w=c.pivot(index='query_id',columns='retrieval_method',values='ndcg_at_k_all')
print(((w.dense>w.lexical).sum(),(w.dense<w.lexical).sum(),(w.dense==w.lexical).sum()),((w.dense>w.hybrid).sum(),(w.dense<w.hybrid).sum(),(w.dense==w.hybrid).sum()))
print('allzero',(w==0).sum().to_dict(),'lex+dense both miss',((w.dense==0)&(w.lexical==0)).sum())
print('--qtype');print(c.groupby(['question_type','retrieval_method']).ndcg_at_k_all.mean().unstack().assign(n=c[c.retrieval_method=='dense'].groupby('question_type').size()))
print('--reduced split dense curated vs reduced same queries')
rd=real[real.kb_variant=='reduced'];print(rd.groupby(['retrieval_method','ans'])[['abstained','partial','faithfulness','reference_agreement','context_relevance']].mean())
# same 111 queries on curated
a=set(rd[rd.ans].query_id); 
print(c[c.query_id.isin(a)].groupby('retrieval_method')[['ndcg_at_k_all','abstained','faithfulness','reference_agreement']].mean())
print(c[~c.query_id.isin(a)].groupby('retrieval_method')[['ndcg_at_k_all','abstained','faithfulness','reference_agreement']].mean())
print('--robust')
b=pd.read_csv('reports/robustness_per_query.csv');print(b.columns.tolist());print(b.groupby(['kb_variant','perturbation']).size())
rs=pd.read_csv('reports/robustness_summary.csv');x=rs[rs.metric.str.endswith('_delta')];x=x.assign(s=x['mean'].map('{:+.3f}'.format)+' ['+x.ci95_low.map('{:+.3f}'.format)+', '+x.ci95_high.map('{:+.3f}'.format)+'] n='+x.n.astype(str))
print(x.pivot_table(index=['kb_variant','perturbation'],columns='metric',values='s',aggfunc='first').to_string())
# stratify drop
rob=[r for r in R if r['record_type']=='robustness' and r['perturbation']=='drop_top_relevant']
rows=[]
for r in rob:
    rel=set(r['relevant_chunk_ids']);left=sum(c['chunk_id'] in rel for c in r['perturbed_retrieved'])
    bg,pg=r['baseline_generation_metrics'],r['perturbed_generation_metrics']
    rows.append((r['kb_variant'],left>0,bg['abstained'],pg['abstained'],bg.get('reference_agreement'),pg.get('reference_agreement'),pg['abstention_label']))
e=pd.DataFrame(rows,columns='kb left ba pa bref pref lab'.split());print(e.groupby(['left'])[['ba','pa']].mean(),e.groupby('left').size());print(e.groupby(['kb','left']).pa.agg(['mean','size']))
print(pd.crosstab(e.left,e.lab,normalize='index'))
print('--variance')
V=[]
for i in range(5):
    for l in open(f'results/variance/repeat_{i}.jsonl',encoding='utf8'):
        r=json.loads(l)
        if r.get('record_type')=='standard':
            g=r['generation_metrics'];ab=g.get('abstained')
            V.append((i,r['retrieval_method'],r['kb_variant'],r['query_id'],None if ab else g['faithfulness'],None if ab else g.get('answer_relevance'),g['context_relevance'],None if ab else g.get('reference_agreement'),ab))
v=pd.DataFrame(V,columns='rep m kb q f ar cr ref ab'.split()).astype({'f':float,'ar':float,'ref':float,'ab':float,'cr':float});print(len(v))
print(v.groupby(['rep','m'])[['f','ar','cr','ref','ab']].mean().groupby('m').std())
print(v.groupby(['m','q'])[['f','ar','cr','ref','ab']].std().mean())
