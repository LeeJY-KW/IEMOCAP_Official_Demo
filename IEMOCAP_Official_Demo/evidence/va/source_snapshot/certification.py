"""Fixed five-fold certification-related experiments, with explicit test gates."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from . import train as shared, matched
from .data import DIMS, load_manifest, save_npz
from .model import InductiveGraph


def read(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now().astimezone().isoformat()


def prepare_va(cache='artifacts/iemocap', output='artifacts/certification_iemocap'):
    """Use source header means as released, never synthesize V/A from categories."""
    original = load_manifest(cache)
    pattern = re.compile(r'^\[([\d.]+)\s*-\s*([\d.]+)\]\s+(\S+)\s+(\w+)\s+\[([\d.]+),\s*([\d.]+),\s*([\d.]+)\]')
    dialogues, hashes, excluded = [], {}, []
    for d in original['dialogues']:
        path = Path(original['dataset_root'])/f"Session{d['session']}"/'dialog/EmoEvaluation'/f"{d['id']}.txt"
        hashes[str(path.resolve())] = file_hash(path)
        values = {}
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if not line.startswith('['):
                continue
            m = pattern.match(line)
            if not m or m[3] in values:
                raise ValueError(f'Malformed/duplicate VAD header: {path}: {line}')
            values[m[3]] = m
        utterances = []
        for u in d['utterances']:
            m = values.pop(u['id'])
            if (float(m[1]), float(m[2]), m[4]) != (u['start'], u['end'], u['original_emotion']):
                raise ValueError('Source and feature manifest disagree')
            va = [float(m[5]), float(m[6])]
            if not all(1 <= x <= 5 for x in va):
                excluded.append(dict(id=u['id'],target=va,reason='source header outside declared1..5; retained as context only; excluded from both V/A losses and metrics'))
            utterances.append(dict(id=u['id'], target=va))
        if values:
            raise ValueError('Unmatched source annotations')
        dialogues.append(dict(id=d['id'], group=d['session'], utterances=utterances))
    assert sum(len(d['utterances']) for d in dialogues) == 10039
    folds = [dict(fold=i, **shared.split_dialogues(original, i)) for i in range(1, 6)]
    descriptor = dict(task='va', name='IEMOCAP utterance V/A', cache=str(Path(cache).resolve()),
                      labels=['valence','arousal'], dialogues=dialogues, folds=folds,
                      source_hashes=hashes, original_manifest_sha256=file_hash(Path(cache)/'manifest.json'),
                      resolution='utterance; no dense frame labels, no interpolation',
                      population='all10039 context utterances;10035 valid V/A pairs scored;4 out-of-range headers excluded without relabeling',excluded=excluded)
    target = Path(output)/'va_dataset.json'
    if target.exists() and read(target) != descriptor:
        raise ValueError('Changed VA descriptor')
    shared.write_json(target, descriptor)
    return target


def prepare_categorical(cache='artifacts/iemocap',output='artifacts/certification_iemocap'):
    original=load_manifest(cache)
    labels=['hap','sad','ang','fea','sur','dis','neu']
    dialogues=[]
    for d in original['dialogues']:
        utterances=[dict(id=u['id'],target=labels.index(u['original_emotion']) if u['original_emotion'] in labels else -100) for u in d['utterances']]
        dialogues.append(dict(id=d['id'],group=d['session'],utterances=utterances))
    folds=[dict(fold=i,**shared.split_dialogues(original,i)) for i in range(1,6)]
    descriptor=dict(task='classification',name='IEMOCAP literal Ekman7',cache=str(Path(cache).resolve()),labels=labels,
                    dialogues=dialogues,folds=folds,source_hashes={str((Path(cache)/'manifest.json').resolve()):file_hash(Path(cache)/'manifest.json')},
                    original_manifest_sha256=file_hash(Path(cache)/'manifest.json'),
                    population='original hap/sad/ang/fea/sur/dis/neu only; exc/fru/xxx/oth ignored as targets, retained as unlabeled within-dialogue context',
                    limitation='controlled corpus, not in-the-wild; only2 disgust samples; absent classes disclosed per fold; no relabeling')
    target=Path(output)/'classification_dataset.json'
    if target.exists() and read(target)!=descriptor:raise ValueError('Changed categorical descriptor')
    shared.write_json(target,descriptor)
    return target


def audit_splits(dataset):
    records = {d['id']: d for d in dataset['dialogues']}
    if len(records) != len(dataset['dialogues']):
        raise ValueError('Duplicate dialogue IDs')
    all_ids = [u['id'] for d in records.values() for u in d['utterances']]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError('Duplicate utterance IDs')
    coverage, counts = [], []
    for fold in dataset['folds']:
        groups = []
        row = dict(fold=fold['fold'])
        flattened = sum([fold[role] for role in ('train','dev','test')], [])
        if len(flattened) != len(set(flattened)) or set(flattened) != set(records):
            raise ValueError('Fold not a disjoint exhaustive partition')
        for role in ('train','dev','test'):
            groups.append({records[d]['group'] for d in fold[role]})
            targets = [u['target'] for d in fold[role] for u in records[d]['utterances']]
            row[role] = dict(dialogues=len(fold[role]), utterances=len(targets))
            if dataset['task'] == 'classification':
                support = np.bincount([t for t in targets if t!=-100], minlength=7)
                if len(support)!=7 or not support.sum():
                    raise ValueError(f'Invalid class support: {role}:{support}')
                row[role]['class_support'] = support.tolist()
                row[role]['labeled_utterances']=int(support.sum())
                row[role]['absent_classes']=[dataset['labels'][i] for i in np.flatnonzero(support==0)]
            else:
                row[role]['labeled_utterances']=int(valid_targets('va',np.asarray(targets)).sum())
        if any(groups[i]&groups[j] for i,j in ((0,1),(0,2),(1,2))):
            raise ValueError('Source group leakage')
        coverage.extend(fold['test'])
        counts.append(row)
    if len(coverage) != len(set(coverage)) or set(coverage) != set(records) or len(counts)!=5:
        raise ValueError('Each dialogue must be tested exactly once across five folds')
    return counts


def freeze(dataset_path, output, epochs=50):
    if not 1<=epochs<=50:raise ValueError('Fixed epoch budget must be 1..50')
    dataset, output = read(dataset_path), Path(output)
    counts = audit_splits(dataset)
    folder = Path(__file__).parent
    names = ['certification.py','model.py','train.py','matched.py','data.py']
    sources = {name:file_hash(folder/name) for name in names}
    cache_hashes = {d['id']:file_hash(Path(dataset['cache'])/'dialogues'/f"{d['id']}.npz") for d in dataset['dialogues']}
    for path, expected in dataset['source_hashes'].items():
        if file_hash(path) != expected:
            raise ValueError('Changed raw annotation source')
    declaration = dict(protocol='certification-fivefold-v1', task=dataset['task'], dataset_sha256=file_hash(dataset_path),
                       dataset_path=str(Path(dataset_path).resolve()), cache_hashes=cache_hashes, sources=sources,
                       environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__),
                       budget=dict(seed=42,epochs=epochs,batch_dialogues=12,lr=1e-4,weight_decay=1e-5),
                       graph=dict(hidden=256,layers=2,dropout=.2,paths=[1,2,3],window=8,max_neighbors=8,context_gate='observed'),
                       loss='half full / half masked; CE or 1-meanCCC+.1MSE on [-1,1]',
                       selection='earliest maximum full-ATV development weightedF1 or meanCCC',
                       folds=dataset['folds'],counts=counts,initialization='fresh random per fold; no previous checkpoint',
                       test_policy='all five selection locks before any heldout prediction; no test retuning')
    path = output/'protocol.json'
    if path.exists():
        if read(path) != declaration:
            raise ValueError('Frozen protocol/source/environment/cache drift')
        for name, expected in sources.items():
            if file_hash(output/'source_snapshot'/name) != expected:
                raise ValueError('Archived source drift')
    else:
        if output.exists() and list(output.iterdir()):
            raise ValueError('Unexpected files in fresh output')
        (output/'source_snapshot').mkdir(parents=True)
        for name in names:
            shutil.copy2(folder/name, output/'source_snapshot'/name)
        shared.write_json(path, declaration)
    return dataset, declaration


def load_data(dataset, protocol, ids):
    indexed={d['id']:d for d in dataset['dialogues']}
    rows=[]
    for identifier in ids:
        path=Path(dataset['cache'])/'dialogues'/f'{identifier}.npz'
        if file_hash(path)!=protocol['cache_hashes'][identifier]:
            raise ValueError('Cache drift')
        with np.load(path,allow_pickle=False) as saved:
            d={k:saved[k].copy() for k in saved.files}
        expected=indexed[identifier]['utterances']
        if d['ids'].tolist()!=[u['id'] for u in expected]:
            raise ValueError('Target/feature order mismatch')
        for name,dimension in DIMS.items():
            if d[name].shape != (len(expected),dimension) or not np.isfinite(d[name]).all():
                raise ValueError('Invalid feature cache')
        d['targets']=np.asarray([u['target'] for u in expected],dtype=np.float32 if dataset['task']=='va' else np.int64)
        d['labels']=d['targets'] if dataset['task']=='classification' else np.full(len(expected),-100,dtype=np.int64)
        d['dialogue_id']=identifier
        rows.append(d)
    return rows


def normalize(rows,stats,device):
    batches=shared.normalize_dialogues(rows,stats,device)
    for b,r in zip(batches,rows):
        b['targets']=torch.as_tensor(r['targets'],device=device)
    return batches


def ccc_numpy(target,prediction):
    y,p=np.asarray(target,dtype=np.float64),np.asarray(prediction,dtype=np.float64)
    if y.shape!=p.shape or y.ndim!=2 or y.shape[1]!=2 or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Invalid CCC inputs')
    ym,pm=y.mean(0),p.mean(0)
    numerator=2*((y-ym)*(p-pm)).mean(0)
    denominator=y.var(0)+p.var(0)+(ym-pm)**2
    return np.divide(numerator,denominator,out=np.ones(2),where=denominator>0)


def regression_loss(raw,target):
    valid=((target>=1)&(target<=5)).all(1)
    if not valid.any():return raw.sum()*0
    raw,target=raw[valid],target[valid]
    p=torch.tanh(raw); y=(target-3)/2
    pm,ym=p.mean(0),y.mean(0)
    cov=((p-pm)*(y-ym)).mean(0)
    denominator=(p-pm).square().mean(0)+(y-ym).square().mean(0)+(pm-ym).square()
    ccc=2*cov/denominator.clamp_min(1e-8)
    return 1-ccc.mean()+.1*F.mse_loss(p,y)


def valid_targets(task,y):
    return np.asarray(y)!=-100 if task=='classification' else ((np.asarray(y)>=1)&(np.asarray(y)<=5)).all(1)


def score_predictions(task,y,p):
    if task=='classification':
        result=shared.classification_metrics(y,np.argmax(p,axis=1),7)
        cm=np.asarray(result['confusion']);true,pred=cm.sum(1),cm.sum(0)
        result.update(support=true.tolist(),per_class_precision=np.divide(cm.diagonal(),pred,out=np.zeros(7),where=pred>0).tolist(),
                      per_class_recall=np.divide(cm.diagonal(),true,out=np.zeros(7),where=true>0).tolist())
        return result
    keep=valid_targets(task,y);y,p=np.asarray(y)[keep],np.asarray(p)[keep]
    scores=ccc_numpy(y,p)
    return dict(ccc_valence=float(scores[0]),ccc_arousal=float(scores[1]),mean_ccc=float(scores.mean()),
                mse=np.mean((np.asarray(y)-p)**2,axis=0).tolist(),n=len(y))


@torch.no_grad()
def evaluate(model,batches,task,path=None):
    model.eval(); targets=[]; predictions=[]; ids=[]; dialogues=[]; times=[]; per_dialogue={}
    for b in batches:
        raw=model(b)
        out=(3+2*torch.tanh(raw)) if task=='va' else raw.softmax(-1)
        y,p=b['targets'].cpu().numpy(),out.cpu().numpy()
        if not np.isfinite(p).all():raise FloatingPointError('Nonfinite predictions')
        targets.extend(y);predictions.extend(p);ids.extend(b['ids']);dialogues.extend([b['dialogue_id']]*len(y));times.extend(b['times'].cpu().numpy())
        per_dialogue[b['dialogue_id']]=score_predictions(task,y,p) if valid_targets(task,y).any() else dict(n=0,note='no labeled targets')
    metrics=score_predictions(task,np.asarray(targets),np.asarray(predictions))
    metrics['per_dialogue']=per_dialogue
    if path:
        save_npz(path,ids=np.asarray(ids),dialogues=np.asarray(dialogues),times=np.asarray(times),targets=np.asarray(targets),target_valid=valid_targets(task,np.asarray(targets)),predictions=np.asarray(predictions))
    return metrics


def make_model(task,graph,device):
    return InductiveGraph(DIMS,2 if task=='va' else 7,variant='multi_graph',**graph).to(device)


def train_fold(dataset,protocol,output,fold,device):
    path=Path(output)/f"fold{fold['fold']}";path.mkdir(exist_ok=True)
    identity=shared.digest(dict(protocol=protocol,fold=fold))
    if (path/'selection.json').exists():
        lock=read(path/'selection.json')
        if lock['identity']!=identity or file_hash(path/'best.pt')!=lock['checkpoint_sha256']:
            raise ValueError('Stale selected checkpoint')
        return lock
    cfg=protocol['budget']; task=dataset['task']; seed=cfg['seed']
    shared.set_seed(seed);rng=np.random.default_rng(seed)
    train_raw=load_data(dataset,protocol,fold['train']);dev_raw=load_data(dataset,protocol,fold['dev'])
    stats=shared.fit_normalizer(train_raw)
    batches=normalize(train_raw,stats,device);dev=normalize(dev_raw,stats,device)
    model=make_model(task,protocol['graph'],device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    first,best,best_epoch,best_model,history=1,-float('inf'),0,None,[]
    if (path/'last.pt').exists():
        state=torch.load(path/'last.pt',map_location=device,weights_only=False)
        if state['identity']!=identity:raise ValueError('Stale resume checkpoint')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);shared.restore_random_state(state['rng'],rng)
        first,best,best_epoch=state['epoch']+1,state['best'],state['best_epoch'];best_model=state['best_model'];history=state['history']
        del state
    for epoch in range(first,cfg['epochs']+1):
        start=time.time();model.train();order,masks,regimes=matched.epoch_plan(batches,seed,epoch);total,n=0.,0
        for offset in range(0,len(order),cfg['batch_dialogues']):
            group=[batches[i] for i in order[offset:offset+cfg['batch_dialogues']]]
            y=torch.cat([b['targets'] for b in group]);losses=[]
            for missing in (False,True):
                outputs=[model(b,torch.as_tensor(masks[b['dialogue_id']],device=device) if missing else None) for b in group]
                raw=torch.cat(outputs)
                losses.append(regression_loss(raw,y) if task=='va' else (F.cross_entropy(raw,y) if (y!=-100).any() else raw.sum()*0))
            loss=.5*sum(losses)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            finite=[torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
            if not finite or not torch.stack(finite).all():raise FloatingPointError('Nonfinite gradient')
            optimizer.step();total+=float(loss.detach())*len(y);n+=len(y)
        metrics=evaluate(model,dev,task);score=metrics['mean_ccc' if task=='va' else 'weighted_f1']
        if score>best:best,best_epoch,best_model=score,epoch,shared.snapshot(model)
        row=dict(epoch=epoch,loss=total/n,dev_score=score,best=best,best_epoch=best_epoch,seconds=time.time()-start,
                 order_hash=shared.digest([batches[i]['dialogue_id'] for i in order]),mask_hash=shared.digest({k:v.tolist() for k,v in masks.items()}),regimes=regimes)
        history.append(row)
        shared.save_torch(path/'last.pt',dict(identity=identity,model=shared.snapshot(model),optimizer=optimizer.state_dict(),rng=shared.random_state(rng),epoch=epoch,best=best,best_epoch=best_epoch,best_model=best_model,history=history))
        matched.save_history(path/'history.jsonl',history)
        shared.write_json(Path(output)/'campaign_status.json',dict(stage='training',task=task,fold=fold['fold'],epoch=epoch,epochs=cfg['epochs'],time=now()))
        print(f'certification {task} fold={fold["fold"]} epoch={epoch}/{cfg["epochs"]} loss={total/n:.5f} dev={score:.5f} best={best:.5f}@{best_epoch} seconds={row["seconds"]:.1f}',flush=True)
    model.load_state_dict(best_model)
    shared.save_torch(path/'best.pt',dict(identity=identity,model=shared.snapshot(model),normalizer=stats,epoch=best_epoch,dev_score=best,task=task,protocol_hash=shared.digest(protocol)))
    shared.write_json(path/'dev.json',evaluate(model,dev,task,path/'dev_predictions.npz'))
    lock=dict(identity=identity,fold=fold['fold'],best_epoch=best_epoch,dev_score=best,epochs_completed=cfg['epochs'],
              checkpoint_sha256=file_hash(path/'best.pt'),normalizer_hash=shared.digest({k:{n:v.tolist() for n,v in fields.items()} for k,fields in stats.items()}),
              test_loaded_during_training=False,locked_at=now(),parameters=sum(p.numel() for p in model.parameters()),training_seconds=sum(h['seconds'] for h in history))
    shared.write_json(path/'selection.json',lock)
    return lock


def summarize(dataset,protocol,output,results,locks):
    task=dataset['task'];keys=['ccc_valence','ccc_arousal','mean_ccc'] if task=='va' else ['weighted_f1','macro_f1','uar','accuracy']
    summary={k:dict(mean=float(np.mean([r[k] for r in results])),sample_sd=float(np.std([r[k] for r in results],ddof=1))) for k in keys}
    y,p,ids=[],[],[]
    for fold in range(1,6):
        with np.load(Path(output)/f'fold{fold}'/'test_predictions.npz',allow_pickle=False) as f:
            y.extend(f['targets']);p.extend(f['predictions']);ids.extend(f['ids'].tolist())
            recalculated=score_predictions(task,f['targets'],f['predictions'])
            for k in keys:
                if not np.isclose(recalculated[k],results[fold-1][k],rtol=0,atol=1e-10):raise ValueError('Prediction/metric mismatch')
    expected={u['id'] for d in dataset['dialogues'] for u in d['utterances']}
    if len(ids)!=len(set(ids)) or set(ids)!=expected:raise ValueError('OOF coverage failure')
    summary['pooled_oof_secondary']=score_predictions(task,np.asarray(y),np.asarray(p))
    summary['training_seconds']=sum(x['training_seconds'] for x in locks)
    summary['test_utterances']=len(ids)
    summary['scored_utterances']=sum(r['n'] for r in results)
    shared.write_json(Path(output)/'test_summary.json',summary)
    lines=[f'# 공인인증 관련 {"4번 V/A 회귀" if task=="va" else "3번 7분류"} 실제 실험 결과','',f'완료: {now()}. Base observed gate, 5-fold, fold당50epoch. 분할별 개발셋으로만 checkpoint를 선택했다.','', '| 지표 | 5개 Fold 평균 ± 표본SD |','|---|---|']
    lines += [f'| {k} | {summary[k]["mean"]:.4f} ± {summary[k]["sample_sd"]:.4f} |' for k in keys]
    lines+=['','| Fold | 개발 선택 epoch | '+ ' | '.join(keys)+' |','|---|---|'+ '|'.join(['---']*len(keys))+'|']
    lines += [f'| {i+1} | {locks[i]["best_epoch"]} | '+' | '.join(f'{r[k]:.4f}' for k in keys)+' |' for i,r in enumerate(results)]
    if task=='classification':
        names=['행복','슬픔','분노','공포','놀람','혐오','중립']
        lines += ['', '| 감정 | 전체 채점 표본 | Fold F1 평균 ± 표본SD |','|---|---|---|']
        for c,name in enumerate(names):
            vals=[r['per_class_f1'][c] for r in results]
            lines.append(f'| {name} | {sum(r["support"][c] for r in results)} | {np.mean(vals):.4f} ± {np.std(vals,ddof=1):.4f} |')
        lines += ['','| Fold | 행복 | 슬픔 | 분노 | 공포 | 놀람 | 혐오 | 중립 |','|---|---|---|---|---|---|---|---|']
        lines += [f'| {i+1} | '+' | '.join(map(str,r['support']))+' |' for i,r in enumerate(results)]
    lines += ['',f'추론 문맥 발화 {len(ids):,}개, 지표 채점 발화 {summary["scored_utterances"]:,}개. 누적 학습 {summary["training_seconds"]/3600:.2f}시간. 모든 OOF ID가 한 번만 시험되며 저장 예측에서 지표를 재검산했다.',
              '발화 단위 실제 V/A 주석의 회귀 결과다. 프레임 단위 연속 주석 평가와 동일하지 않다.' if task=='va' else 'IEMOCAP 원본의 해당 7개 감정만 표적으로 썼다. 흥분·좌절·모호 라벨은 다른 감정으로 치환하지 않고 분류 손실/채점에서 제외했다. 혐오는 전체2개로, 일부 Fold에는 정답 표본이 없다. 통제 환경 자료이므로 In-the-wild 인증 요건을 충족한 결과가 아니다.',
              '공인기관 인증 완료를 의미하지 않는다. 기존 6분류10회 및 논문73.13과 직접 비교하지 않는다. 테스트 재튜닝은 수행하지 않았다.','']
    (Path(output)/'results.ko.md').write_text('\n'.join(lines))
    shared.write_json(Path(output)/'final_audit.json',dict(passed=True,folds=5,utterances=len(ids),prediction_metrics_verified=True,protocol_sha256=file_hash(Path(output)/'protocol.json'),time=now()))
    return summary


def campaign(dataset_path,output,device='cuda',epochs=50):
    torch.set_num_threads(4)
    dataset,protocol=freeze(dataset_path,output,epochs)
    output=Path(output)
    locks=[train_fold(dataset,protocol,output,fold,device) for fold in protocol['folds']]
    gate=dict(protocol_hash=shared.digest(protocol),selection_hashes={str(i):file_hash(output/f'fold{i}'/'selection.json') for i in range(1,6)},all_folds_locked=True)
    if (output/'evaluation_gate.json').exists() and read(output/'evaluation_gate.json')!=gate:raise ValueError('Changed test gate')
    shared.write_json(output/'evaluation_gate.json',gate)
    results=[]
    for fold,lock in zip(protocol['folds'],locks):
        path=output/f"fold{fold['fold']}"
        if file_hash(path/'best.pt')!=lock['checkpoint_sha256']:raise ValueError('Changed best checkpoint')
        if (path/'test.json').exists():
            result=read(path/'test.json')
        else:
            saved=torch.load(path/'best.pt',map_location=device,weights_only=False)
            model=make_model(dataset['task'],protocol['graph'],device);model.load_state_dict(saved['model'])
            batches=normalize(load_data(dataset,protocol,fold['test']),saved['normalizer'],device)
            result=evaluate(model,batches,dataset['task'],path/'test_predictions.npz')
            shared.write_json(path/'test.json',result)
        results.append(result)
    shared.write_json(output/'test_results.json',results)
    summary=summarize(dataset,protocol,output,results,locks)
    shared.write_json(output/'campaign_status.json',dict(stage='complete',task=dataset['task'],time=now()))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare-va','prepare-classification','campaign'])
    parser.add_argument('--dataset');parser.add_argument('--output');parser.add_argument('--device',default='cuda');parser.add_argument('--epochs',type=int,default=50)
    args=parser.parse_args()
    if args.command=='prepare-va':print(prepare_va())
    elif args.command=='prepare-classification':print(prepare_categorical())
    else:campaign(args.dataset,args.output,args.device,args.epochs)
