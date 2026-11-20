"""Local non-plotting Weather setup must not download fonts or datasets."""
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

@pytest.fixture
def local_data(tmp_path):
    for split in ('train','valid'):
        (tmp_path/split/'images').mkdir(parents=True)
    data=tmp_path/'data.yaml'
    data.write_text(yaml.safe_dump(dict(path=str(tmp_path),train='train/images',val='valid/images',
                                      nc=7,names={i:'目标'+str(i) for i in range(7)}),allow_unicode=True),encoding='utf-8')
    return data

def test_shared_data_check_retains_default_font_behavior(local_data,monkeypatch):
    import ultralytics.data.utils as utils
    calls=[]
    monkeypatch.setattr(utils,'check_font',lambda name:calls.append(name))
    assert utils.check_det_dataset(str(local_data))['nc']==7
    assert calls==['Arial.Unicode.ttf']

def test_data_check_explicitly_skips_fonts(local_data,monkeypatch):
    import ultralytics.data.utils as utils
    def forbidden(*args,**kwargs):raise AssertionError('unexpected font/network check')
    monkeypatch.setattr(utils,'check_font',forbidden)
    assert utils.check_det_dataset(str(local_data),autodownload=False,check_fonts=False)['nc']==7

def test_weather_trainer_real_constructor_without_font_patch(local_data,monkeypatch,tmp_path):
    import ultralytics.data.utils as utils
    from weather_hfc_training import WeatherHFCTrainer
    def forbidden(*args,**kwargs):raise AssertionError('unexpected font/network check')
    monkeypatch.setattr(utils,'check_font',forbidden)
    manifest=tmp_path/'manifest.jsonl';manifest.write_text('',encoding='utf-8')
    overrides=dict(model=str(ROOT/'ultralytics/cfg/models/rt-detr/weather-hfc/baseline.yaml'),
                   data=str(local_data),device='cpu',project=str(tmp_path/'runs'),name='setup',
                   plots=False,batch=2,workers=0,multi_scale=False)
    for key in ('mosaic','mixup','copy_paste','degrees','translate','scale','shear','perspective','fliplr','flipud'):
        overrides[key]=0
    trainer=WeatherHFCTrainer(overrides=overrides,manifest=manifest)
    assert trainer.data['nc']==7 and trainer.model.endswith('baseline.yaml')

def test_weather_validator_data_check_is_offline(local_data,monkeypatch):
    import ultralytics.data.utils as utils
    from weather_hfc_training import WeatherHFCValidator
    def forbidden(*args,**kwargs):raise AssertionError('unexpected font/network check')
    monkeypatch.setattr(utils,'check_font',forbidden)
    validator=WeatherHFCValidator.__new__(WeatherHFCValidator)
    validator.args=SimpleNamespace(plots=False)
    assert validator.check_detection_data(str(local_data))['nc']==7
