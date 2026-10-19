from v19_config import Settings
from v1983_stages import STAGES, apply_stage, get_stage, next_stage_key, stage_from_settings
from v198_universe import progressive_balanced_sample


def fake_catalog(n=900):
    rows=[]
    for i in range(n):
        code=1000+i
        suffix='.TW' if i%2==0 else '.TWO'
        rows.append({'ticker':f'{code:04d}{suffix}','name':str(code),'sector':f'S{i%9}'})
    return rows


def test_stage_sizes_are_progressive_then_hold_universe_for_time_confirmation():
    stocks=[x['stocks'] for x in STAGES]
    dates=[x['dates'] for x in STAGES]
    assert stocks==[40,60,80,100,120,150,150,150]
    assert dates==[4,6,8,12,14,18,24,24]
    assert stocks==sorted(stocks) and dates==sorted(dates)
    assert all(dates[i] <= dates[i+1] for i in range(len(dates)-1))
    s=apply_stage(Settings(),'S1')
    assert s.stocks==40 and s.dates==4 and s.download_batch_size==15
    assert s.progressive_universe and not s.export_full_artifacts and not s.export_history_database
    assert s.factor_audit is False and s.optional_data is False


def test_new_bridge_stages_are_lightweight_cloud_first():
    s5=apply_stage(Settings(optional_data=True,financial_history=True),'S5',optional_data=True,financial_history=True)
    s6=apply_stage(Settings(optional_data=True,financial_history=True),'S6',optional_data=True,financial_history=True)
    assert (s5.stocks,s5.dates,s5.download_batch_size)==(120,14,20)
    assert (s6.stocks,s6.dates,s6.download_batch_size)==(150,18,20)
    assert not s5.optional_data and not s5.financial_history and not s5.factor_audit
    assert not s6.optional_data and not s6.financial_history and not s6.factor_audit
    assert not s5.export_full_artifacts and not s6.export_full_artifacts
    s7=apply_stage(Settings(seed=1920,sampling_method='uniform'),'S7')
    s8=apply_stage(Settings(seed=1920,universe_seed=1900,sampling_method='uniform'),'S8')
    assert all(get_stage(k)['cloud_enabled'] for k in ('S5','S6','S7','S8'))
    assert s5.min_download_coverage==.90 and s6.min_download_coverage==.90 and s7.min_download_coverage==.90 and s8.min_download_coverage==.90
    assert (s7.stocks,s7.dates)==(150,24) and s7.seed==271828 and s7.sampling_method=='stratified'
    assert (s8.stocks,s8.dates)==(150,24) and s8.seed==314159 and s8.universe_seed==161803 and s8.sampling_method=='stratified'


def test_optional_data_opens_only_at_s4_during_cloud_scaling():
    s2=apply_stage(Settings(optional_data=True,financial_history=True),'S2',optional_data=True,financial_history=True)
    assert not s2.optional_data and not s2.financial_history
    s4=apply_stage(Settings(optional_data=True,financial_history=True),'S4',optional_data=True,financial_history=True)
    assert s4.optional_data and not s4.financial_history
    s5=apply_stage(Settings(optional_data=True,financial_history=True),'S5',optional_data=True,financial_history=True)
    assert not s5.optional_data and not s5.financial_history


def test_progressive_balanced_sample_is_nested_and_order_invariant():
    src=fake_catalog()
    pools=[]
    for n in (40,60,80,100,120,150):
        a=progressive_balanced_sample(src,n,1900,'balanced')
        b=progressive_balanced_sample(src[::-1],n,1900,'balanced')
        assert [x['ticker'] for x in a]==[x['ticker'] for x in b]
        pools.append({x['ticker'] for x in a})
    assert all(pools[i] < pools[i+1] for i in range(len(pools)-1))


def test_stage_navigation():
    assert next_stage_key(None)=='S1'
    assert next_stage_key('S1')=='S2'
    assert next_stage_key('S4')=='S5'
    assert next_stage_key('S5')=='S6'
    assert next_stage_key('S6')=='S7'
    assert next_stage_key('S7')=='S8'
    assert next_stage_key('S8') is None


def test_stage_from_settings_handles_new_and_legacy_s5_dimensions():
    assert stage_from_settings({'research_stage':'S5','stocks':120,'dates':14})=='S5'
    assert stage_from_settings({'research_stage':'S6','stocks':150,'dates':18})=='S6'
    assert stage_from_settings({'research_stage':'S7','stocks':150,'dates':24})=='S7'
    assert stage_from_settings({'research_stage':'S8','stocks':150,'dates':24})=='S8'
    # V19.8.3 used S5 for 150x18; dimensions safely map it to the new S6.
    assert stage_from_settings({'research_stage':'S5','stocks':150,'dates':18})=='S6'
    assert stage_from_settings({'research_stage':'S5','stocks':999,'dates':18}) is None


def _stage_report(stage='S2', complete_cases=None, coverage=.95, dates=None, attempts=None, status='RESEARCH_PARTIAL'):
    from v1983_stages import get_stage
    st=get_stage(stage)
    n_dates=st['dates'] if dates is None else dates
    expected=st['dates']*3
    attempts=expected if attempts is None else attempts
    complete_cases=expected-1 if complete_cases is None else complete_cases
    return {
        'status':status,
        'settings':{'research_stage':stage,'stocks':st['stocks'],'dates':st['dates'],'horizons':[5,20,60]},
        'cutoffs':[f'2026-01-{i+1:02d}' for i in range(n_dates)],
        'fits':[{'status':'TEST'} for _ in range(attempts)],
        'expected_adaptive_cases':expected,
        'complete_adaptive_cases':complete_cases,
        'summary':[],
        'case_diagnostics':[],
        'all_factor_coverage':{},
        'universe_diagnostics':{'universe_quality':{'download_coverage':coverage}},
        'symbols':int(st['stocks']*coverage)+1,
    }


def test_partial_research_is_not_an_execution_lock_anymore():
    from v1983_stages import evaluate_stage_gate
    gate=evaluate_stage_gate(_stage_report('S2', complete_cases=17))
    assert gate['execution_status']=='STAGE_COMPLETE'
    assert gate['research_status']=='RESEARCH_PARTIAL'
    assert gate['complete_adaptive_cases']==17 and gate['expected_cases']==18
    assert 'SOME_ADAPTIVE_OUTCOMES_INCOMPLETE' in gate['warnings']


def test_known_s3_23_of_24_pattern_can_advance_but_stays_partial():
    from v1983_stages import evaluate_stage_gate
    gate=evaluate_stage_gate(_stage_report('S3', complete_cases=23))
    assert gate['execution_status']=='STAGE_COMPLETE'
    assert gate['research_case_completion']==23/24


def test_stage_gate_still_blocks_real_execution_gaps():
    from v1983_stages import evaluate_stage_gate
    missing_date=evaluate_stage_gate(_stage_report('S3', dates=7, attempts=24))
    assert missing_date['execution_status']=='STAGE_INCOMPLETE'
    assert 'cutoff_dates' in missing_date['blocking_reasons']
    low_coverage=evaluate_stage_gate(_stage_report('S2', coverage=.89))
    assert low_coverage['execution_status']=='STAGE_INCOMPLETE'
    assert 'history_coverage' in low_coverage['blocking_reasons']
    short_attempt=evaluate_stage_gate(_stage_report('S2', attempts=17))
    assert short_attempt['execution_status']=='STAGE_INCOMPLETE'
    assert 'case_attempts' in short_attempt['blocking_reasons']


def test_service_recovers_old_partial_stage_pointer_without_rerun(tmp_path):
    from v19_data import atomic_json
    from v19_service import read_last_report
    root=tmp_path/'v19';runs=root/'runs';runs.mkdir(parents=True)
    s1=runs/'old_s1';s2=runs/'new_s2';s1.mkdir();s2.mkdir()
    r1=_stage_report('S1', complete_cases=12, status='RESEARCH_COMPLETE')
    r2=_stage_report('S2', complete_cases=17, status='RESEARCH_PARTIAL')
    atomic_json(s1/'v19_results.json',r1);atomic_json(s2/'v19_results.json',r2)
    atomic_json(root/'last_success.json',{'directory':str(s1.resolve()),'status':'RESEARCH_COMPLETE'})
    recovered=read_last_report(root)
    assert recovered['settings']['research_stage']=='S2'
    pointer=(__import__('json').loads((root/'last_success.json').read_text()))
    assert pointer['stage']=='S2' and pointer['execution_status']=='STAGE_COMPLETE'
