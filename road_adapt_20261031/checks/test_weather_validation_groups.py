"""CPU regressions for single-pass per-weather validation statistics."""
from pathlib import Path
import importlib
import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

CANDIDATE=Path(__file__).resolve().parents[1]
PROJECT=CANDIDATE
sys.path[:0]=[str(CANDIDATE),str(PROJECT)]
from ultralytics.utils.metrics import DetMetrics
from weather_hfc_training import WeatherHFCValidator

torch.set_num_threads(2)
NAMES={0:'crack'}

def subject():
    assert importlib.util.find_spec('weather_validation_groups') is not None, 'Grouped validation feature is not implemented'
    return importlib.import_module('weather_validation_groups')

def image_stat(correct,scores,targets):
    return dict(tp=torch.tensor([[value]*10 for value in correct],dtype=torch.bool).reshape(-1,10),
                conf=torch.tensor(scores),pred_cls=torch.zeros(len(scores)),target_cls=torch.tensor(targets).float())

def column_stats(images):
    return {key:[row[key] for row in images] for key in ('tp','conf','pred_cls','target_cls')}

def reference(images):
    metrics=DetMetrics(names=NAMES,plot=False)
    if images:
        merged={key:torch.cat([row[key] for row in images],0).numpy() for key in images[0]}
        if merged['tp'].any():metrics.process(**merged)
    return {key:float(value) for key,value in metrics.results_dict.items()}

def test_degraded_recomputes_pooled_ap_instead_of_mean_weather_ap():
    module=subject()
    fog=image_stat([True],[.7],[0])
    rain=image_stat([False,True],[.99,.6],[0,0])
    lowlight=image_stat([False],[.8],[0,0,0])
    result=module.group_detection_statistics(column_stats([fog,rain,lowlight]),
        ['fog','rain','lowlight'],{'clear':0,'fog':1,'rain':1,'lowlight':1},NAMES)
    assert result['degraded']['metrics']==pytest.approx(reference([fog,rain,lowlight]))
    assert result['fog']['metrics']==pytest.approx(reference([fog]))
    assert result['rain']['metrics']==pytest.approx(reference([rain]))
    mean=np.mean([result[name]['metrics']['metrics/mAP50-95(B)'] for name in ('fog','rain','lowlight')])
    assert result['degraded']['metrics']['metrics/mAP50-95(B)']!=pytest.approx(mean)
    assert result['degraded']['images']==3
    assert result['degraded']['instances']==6

def test_empty_images_count_without_fabricated_statistics():
    module=subject()
    result=module.group_detection_statistics(column_stats([]),[],{'clear':2,'fog':1,'rain':0,'lowlight':0},NAMES)
    assert set(result)=={'clear','fog','rain','lowlight','degraded'}
    assert result['clear']['images']==2 and result['degraded']['images']==1
    assert all(row['instances']==0 and all(value==0 for value in row['metrics'].values()) for row in result.values())

@pytest.mark.parametrize('change',('labels','columns','unknown','count'))
def test_inconsistent_condition_metadata_is_rejected(change):
    module=subject()
    stats=column_stats([image_stat([True],[.9],[0])])
    labels=['fog'];counts={'clear':0,'fog':1,'rain':0,'lowlight':0}
    if change=='labels':labels=[]
    if change=='columns':stats['conf']=[]
    if change=='unknown':labels=['snow']
    if change=='count':counts['fog']=0
    with pytest.raises(ValueError):module.group_detection_statistics(stats,labels,counts,NAMES)

def make_validator(kind,tmp_path):
    validator=kind(manifest=tmp_path/'manifest.jsonl',save_dir=tmp_path/kind.__name__,
                   args=dict(device='cpu',imgsz=32,split='val',plots=False,save_txt=False,save_json=False,
                             conf=.001,workers=0))
    validator.device=torch.device('cpu');validator.data={'val':'valid.txt'};validator.training=False
    validator.init_metrics(SimpleNamespace(names=NAMES))
    return validator

def batch(conditions,gt_image_indices):
    size=len(conditions)
    return dict(img=torch.zeros(size,3,32,32),condition=conditions,
                cls=torch.zeros(len(gt_image_indices),1),bboxes=torch.tensor([[.5,.5,.5,.5]]*len(gt_image_indices)).reshape(-1,4),
                batch_idx=torch.tensor(gt_image_indices),ori_shape=[(32,32)]*size,
                ratio_pad=[((1.,1.),(0.,0.))]*size,im_file=[f'valid_{i}.jpg' for i in range(size)])

def detection(score=.9):return torch.tensor([[8.,8.,24.,24.,score,0.]])
def empty():return torch.empty((0,6))

def test_validator_empty_skip_alignment_and_native_overall_unchanged(tmp_path):
    module=subject()
    wrapped=make_validator(module.ConditionWeatherValidator,tmp_path)
    native=make_validator(WeatherHFCValidator,tmp_path)
    inputs=[([empty(),detection()],batch(['clear','fog'],[1])),
            ([empty(),detection(.7)],batch(['rain','lowlight'],[0]))]
    for predictions,metadata in inputs:
        wrapped.update_metrics([p.clone() for p in predictions],metadata)
        native.update_metrics([p.clone() for p in predictions],metadata)
    overall=wrapped.get_stats()
    assert overall==native.get_stats()
    groups=wrapped.condition_metrics
    assert wrapped.seen==4
    assert len(wrapped.stats['tp'])==3
    assert groups['clear']['images']==1 and groups['clear']['instances']==0
    assert groups['fog']['instances']==groups['rain']['instances']==1
    assert groups['lowlight']['instances']==0
    assert groups['degraded']['images']==3 and groups['degraded']['instances']==2

def test_wholly_empty_validator_returns_zero_and_counts_images(tmp_path):
    module=subject()
    validator=make_validator(module.ConditionWeatherValidator,tmp_path)
    validator.update_metrics([empty(),empty()],batch(['clear','fog'],[]))
    assert all(value==0 for value in validator.get_stats().values())
    assert validator.condition_metrics['degraded']['images']==1
    assert validator.condition_metrics['clear']['images']==1

def test_new_epoch_with_zero_true_positives_does_not_retain_old_score(tmp_path):
    module=subject()
    validator=make_validator(module.ConditionWeatherValidator,tmp_path)
    validator.update_metrics([detection()],batch(['fog'],[0]))
    assert validator.get_stats()['metrics/mAP50(B)']>0
    validator.init_metrics(SimpleNamespace(names=NAMES))
    validator.update_metrics([empty()],batch(['rain'],[0]))
    assert validator.get_stats()['metrics/mAP50(B)']==0
    assert validator.condition_metrics['rain']['images']==1
    assert validator.condition_metrics['fog']['images']==0

def test_validator_rejects_missing_or_wrong_length_conditions(tmp_path):
    module=subject()
    validator=make_validator(module.ConditionWeatherValidator,tmp_path)
    metadata=batch(['fog'],[0]);metadata['condition']=[]
    with pytest.raises(ValueError):validator.update_metrics([detection()],metadata)
    assert validator.seen==0

def test_helper_does_not_mutate_native_statistics():
    module=subject()
    stats=column_stats([image_stat([True],[.9],[0])])
    copied={key:[tensor.clone() for tensor in values] for key,values in stats.items()}
    module.group_detection_statistics(stats,['clear'],{'clear':1,'fog':0,'rain':0,'lowlight':0},NAMES)
    assert all(torch.equal(a,b) for key in stats for a,b in zip(stats[key],copied[key]))
