"""Single-pass weather diagnostics for the opt-in fine-tuning profile.

Ordinary overall metrics stay on the native implementation. Each weather group
and the pooled degraded group use the SAME per-image detection statistics, not
extra inference or an average of AP values. Only this opt-in subclass resets
stale cross-epoch metrics and handles a completely empty validation result.
"""
from collections import Counter
from numbers import Integral

import numpy as np
import torch

from ultralytics.utils.metrics import DetMetrics
from weather_hfc_training import WeatherHFCValidator

CONDITIONS=('clear','fog','rain','lowlight')
STAT_KEYS=('tp','conf','pred_cls','target_cls')


def group_detection_statistics(stats,labels,counts,names):
    """Compute per-condition and pooled-weather AP from aligned native stats.

    ``labels[i]`` names the condition of the image that appended stats row i.
    ``counts`` gives ALL image counts, including GT-free, prediction-free images
    omitted by native update_metrics. Inputs are read only.
    """
    if set(stats)!=set(STAT_KEYS):
        raise ValueError('Expected the four native detection statistics columns')
    lengths={len(stats[key]) for key in STAT_KEYS}
    if len(lengths)!=1 or len(labels)!=next(iter(lengths)):
        raise ValueError('Condition labels and all statistics columns must have matching lengths')
    if any(label not in CONDITIONS for label in labels):
        raise ValueError('Unknown condition label in detection statistics')
    if set(counts)-set(CONDITIONS):
        raise ValueError('Image counts contain an unknown condition')
    image_counts={condition:counts.get(condition,0) for condition in CONDITIONS}
    if any(not isinstance(value,Integral) or isinstance(value,bool) or value<0 for value in image_counts.values()):
        raise ValueError('Condition image counts must be nonnegative integers')
    appended=Counter(labels)
    if any(appended[condition]>image_counts[condition] for condition in CONDITIONS):
        raise ValueError('Statistics rows exceed actual image counts for a condition')
    for index in range(len(labels)):
        tp,conf,pred_cls,target_cls=(stats[key][index] for key in STAT_KEYS)
        if (not all(torch.is_tensor(value) for value in (tp,conf,pred_cls,target_cls))
                or tp.ndim!=2 or tp.shape[1]!=10
                or any(value.ndim!=1 for value in (conf,pred_cls,target_cls))
                or len(tp)!=len(conf) or len(tp)!=len(pred_cls)):
            raise ValueError('Malformed per-image native detection statistics')
    output={}
    groups={condition:(condition,) for condition in CONDITIONS}
    groups['degraded']=CONDITIONS[1:]
    for group,conditions in groups.items():
        indices=[index for index,label in enumerate(labels) if label in conditions]
        metric=DetMetrics(names=names,plot=False)
        instances=0
        if indices:
            joined={key:torch.cat([stats[key][index].detach() for index in indices],0).cpu().numpy()
                    for key in STAT_KEYS}
            instances=len(joined['target_cls'])
            if joined['tp'].any():
                metric.process(**joined)
        output[group]=dict(images=int(sum(image_counts[condition] for condition in conditions)),
                           instances=int(instances),
                           metrics={key:float(value) for key,value in metric.results_dict.items()})
    return output


class ConditionWeatherValidator(WeatherHFCValidator):
    """Keep native overall AP; expose extra ``condition_metrics`` after get_stats."""

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.condition_metrics={}
        self._condition_stats_labels=[]
        self._condition_image_counts=Counter({condition:0 for condition in CONDITIONS})

    def build_dataset(self,img_path,mode='val',batch=None):
        if mode!='val' or self.args.split!='val':
            raise ValueError('ConditionWeatherValidator only evaluates the validation split')
        dataset=super().build_dataset(img_path,mode,batch)
        if dataset.source_split!='valid':
            raise ValueError('Condition validation requires the valid manifest split')
        return dataset

    def init_metrics(self,model):
        # Native get_stats skips processing if TP is entirely false. A fresh
        # metric object prevents a previous epoch's AP from surviving that case.
        self.metrics=DetMetrics(save_dir=self.save_dir,on_plot=self.on_plot)
        super().init_metrics(model)
        self.condition_metrics={}
        self._condition_stats_labels=[]
        self._condition_image_counts=Counter({condition:0 for condition in CONDITIONS})

    def update_metrics(self,preds,batch):
        conditions=batch.get('condition')
        if (not isinstance(conditions,(list,tuple)) or len(conditions)!=len(preds)
                or len(batch['img'])!=len(preds)
                or any(condition not in CONDITIONS for condition in conditions)):
            raise ValueError('Every validation image needs one known condition label')
        batch_idx=batch['batch_idx'].reshape(-1)
        expected=[]
        for index,(condition,prediction) in enumerate(zip(conditions,preds)):
            # Native skips appending exactly when BOTH predictions and GT are
            # empty; such images still contribute to the actual image counts.
            if len(prediction) or bool((batch_idx==index).any()):
                expected.append(condition)
        before={key:len(values) for key,values in self.stats.items()}
        super().update_metrics(preds,batch)
        if any(len(self.stats[key])-before[key]!=len(expected) for key in STAT_KEYS):
            raise RuntimeError('Native statistics append behavior changed; condition alignment is unsafe')
        self._condition_stats_labels.extend(expected)
        self._condition_image_counts.update(conditions)

    def get_stats(self):
        if all(len(self.stats[key])==0 for key in STAT_KEYS):
            # Native torch.cat([]) fails for an entirely background-only,
            # prediction-free validation. Report the well-defined zero case.
            self.nt_per_class=np.zeros(self.nc,dtype=np.int64)
            overall=self.metrics.results_dict
        else:
            overall=super().get_stats()
        if sum(self._condition_image_counts.values())!=self.seen:
            raise RuntimeError('Condition image counts do not match native seen images')
        self.condition_metrics=group_detection_statistics(self.stats,self._condition_stats_labels,
                                                         self._condition_image_counts,self.names)
        return overall
