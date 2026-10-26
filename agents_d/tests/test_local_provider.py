"""Offline contracts for self-hosted text/VLM inference; no GPU or network."""
import base64
import json
from pathlib import Path

import httpx2
import pytest
from agents import Agent, ModelTracing, RunConfig, Runner, function_tool
from agents.tool import ToolOutputImage, ToolOutputText
from pydantic import BaseModel

from realclip_d import agent_runtime as runtime
from realclip_d.config import Settings
from realclip_d.local_provider import LocalChatModel, prepare_local_input


def local_settings(**updates):
    return Settings(provider='openai_compatible', model='local-test-vlm',
        base_url='http://local.invalid/v1', web_search=False, budget_usd=1,
        input_usd_per_million=0, output_usd_per_million=0,
        max_input_tokens=10000, local_context_margin_tokens=10, **updates)


def image_url():
    p=Path(__file__).parents[1]/'examples/images/treatment_dna_f1.png'
    return 'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()


def response(message, usage=True):
    d={'id':'local-completion','object':'chat.completion','created':1,'model':'local-test-vlm',
       'choices':[{'index':0,'finish_reason':'tool_calls' if message.get('tool_calls') else 'stop','message':message}]}
    if usage:d['usage']={'prompt_tokens':100,'completion_tokens':10,'total_tokens':110}
    return d


def test_local_configuration_is_explicit_and_does_not_enable_hosted_search():
    with pytest.raises(ValueError,match='web search'):
        Settings(provider='openai_compatible',model='vlm',base_url='http://localhost:8000/v1')
    for url in ['http://name:password@localhost/v1','file:///model','http://localhost/v1?key=secret']:
        with pytest.raises(ValueError):
            Settings(provider='openai_compatible',model='vlm',base_url=url,web_search=False)
    assert local_settings().model=='local-test-vlm'
    assert runtime._model_settings(local_settings(),False).reasoning is None


@pytest.mark.asyncio
async def test_cloud_key_is_not_used_by_local_client(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','cloud-only-test-key')
    monkeypatch.delenv('LOCAL_MODEL_API_KEY',raising=False)
    model,client=runtime._live_model(local_settings())
    assert isinstance(model,LocalChatModel)
    assert client.api_key=='not-required'
    assert str(client.base_url)=='http://local.invalid/v1/'
    await client.close()


def test_tool_images_are_not_silently_dropped_and_do_not_interrupt_tool_results():
    url=image_url()
    original=[{'type':'function_call_output','call_id':'a','output':[
        {'type':'input_text','text':'native crop metadata'}, {'type':'input_image','image_url':url,'detail':'high'}]},
        {'type':'function_call_output','call_id':'b','output':'other tool result'}]
    mapped=prepare_local_input(original)
    assert [r.get('type',r.get('role')) for r in mapped]==['function_call_output','function_call_output','user']
    assert mapped[0]['output']=='native crop metadata'
    assert mapped[-1]['content'][-1]['image_url']==url
    assert isinstance(original[0]['output'],list)
    with pytest.raises(ValueError,match='Unsupported'):
        prepare_local_input([{'type':'function_call_output','call_id':'x','output':[{'type':'input_file','file_id':'x'}]}])


class Final(BaseModel):
    answer:str


@pytest.mark.asyncio
async def test_real_sdk_local_tool_image_loop_schema_counting_and_usage(tmp_path):
    events=[];generations=[];counts=[];url=image_url()
    def handle(req):
        body=json.loads(req.content);events.append(req.url.path)
        if req.url.path=='/tokenize':
            counts.append(body);return httpx2.Response(200,json={'count':100})
        assert req.url.path=='/v1/chat/completions'
        generations.append(body)
        assert 'reasoning_effort' not in body and 'service_tier' not in body
        assert body['response_format']['type']=='json_schema'
        if len(generations)==1:
            msg={'role':'assistant','content':None,'tool_calls':[{'id':'crop_call','type':'function',
                 'function':{'name':'inspect_tile','arguments':'{}'}}]}
        else:msg={'role':'assistant','content':'{"answer":"done"}'}
        return httpx2.Response(200,json=response(msg))
    settings=local_settings(observation_max_turns=3,local_extra_body={'chat_template_kwargs':{'enable_thinking':False}})
    client=runtime._DAsyncOpenAI(api_key='local-test',base_url=settings.base_url,max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle)))
    inner=LocalChatModel(model=settings.model,openai_client=client,settings=settings)
    ledger=runtime.BudgetLedger(tmp_path/'budget.json',1,settings)
    model=runtime.RecordedModel(inner,settings,tmp_path/'stage','observation',ledger,True,False)
    @function_tool
    async def inspect_tile()->list[ToolOutputText|ToolOutputImage]:
        return [ToolOutputText(text='native crop; software test only'),ToolOutputImage(image_url=url,detail='high')]
    agent=Agent(name='local-probe',instructions='Software plumbing only.',model=model,
        model_settings=runtime._model_settings(settings,False),tools=[inspect_tile],output_type=Final)
    try:
        result=await Runner.run(agent,[{'role':'user','content':[{'type':'input_text','text':'Inspect the tile.'},
            {'type':'input_image','image_url':url,'detail':'high'}]}],max_turns=3,
            run_config=RunConfig(tracing_disabled=True))
    finally:await client.close()
    assert result.final_output.answer=='done'
    assert events==['/tokenize','/v1/chat/completions']*2
    for count,generated in zip(counts,generations,strict=True):
        assert count['messages']==generated['messages']
        assert count['tools']==generated['tools']
        assert count['chat_template_kwargs']==generated['chat_template_kwargs']
    assert any(m['role']=='tool' and 'native crop' in m['content'] for m in generations[-1]['messages'])
    images=[part for m in generations[-1]['messages'] if m['role']=='user'
            for part in (m['content'] if isinstance(m['content'],list) else []) if part['type']=='image_url']
    assert len(images)==2 and all(p['image_url']['url']==url for p in images)
    assert all(r['status']=='complete' and r['usage']['input_tokens']==100 for r in ledger.state['requests'])
    assert 'data:image/' not in (tmp_path/'stage/requests/002.json').read_text()


@pytest.mark.asyncio
@pytest.mark.parametrize('bad_count',[None,'100',True,20000])
async def test_failed_or_excess_count_sends_no_generation_and_makes_no_reservation(tmp_path,bad_count):
    paths=[]
    def handle(req):
        paths.append(req.url.path)
        return httpx2.Response(200,json={'count':bad_count})
    settings=local_settings();client=runtime._DAsyncOpenAI(api_key='local',base_url=settings.base_url,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle)))
    inner=LocalChatModel(model=settings.model,openai_client=client,settings=settings)
    ledger=runtime.BudgetLedger(tmp_path/'budget.json',1,settings)
    recorded=runtime.RecordedModel(inner,settings,tmp_path/'stage','research',ledger,True,False)
    try:
        with pytest.raises((ValueError,runtime.BudgetError)):
            await recorded.get_response(system_instructions='Test.',input='Test.',
                model_settings=runtime._model_settings(settings,False),tools=[],output_schema=None,
                handoffs=[],tracing=ModelTracing.DISABLED)
    finally:await client.close()
    assert paths==['/tokenize'] and not ledger.state['requests']
    assert not list((tmp_path/'stage/requests').glob('*.json'))


@pytest.mark.asyncio
async def test_missing_provider_usage_stays_unknown_without_retry(tmp_path):
    calls=[]
    def handle(req):
        calls.append(req.url.path)
        return httpx2.Response(200,json={'count':100} if req.url.path=='/tokenize' else
            response({'role':'assistant','content':'done'},usage=False))
    settings=local_settings();client=runtime._DAsyncOpenAI(api_key='local',base_url=settings.base_url,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle)),max_retries=0)
    inner=LocalChatModel(model=settings.model,openai_client=client,settings=settings)
    ledger=runtime.BudgetLedger(tmp_path/'budget.json',1,settings)
    recorded=runtime.RecordedModel(inner,settings,tmp_path/'stage','research',ledger,True,False)
    try:
        with pytest.raises(runtime.UnknownUsageError):
            await recorded.get_response(system_instructions='Test.',input='Test.',
                model_settings=runtime._model_settings(settings,False),tools=[],output_schema=None,
                handoffs=[],tracing=ModelTracing.DISABLED)
    finally:await client.close()
    assert calls==['/tokenize','/v1/chat/completions']
    assert ledger.state['requests'][0]['status']=='unknown'


@pytest.mark.asyncio
@pytest.mark.parametrize('generate',[False,True])
async def test_server_probe_checks_models_and_opt_in_tool_image_generation(tmp_path,monkeypatch,generate):
    from realclip_d.local_provider_check import check
    settings=local_settings(observation_max_turns=2)
    config=tmp_path/'config.json';config.write_text(settings.model_dump_json())
    calls=[];generations=[]
    def handle(req):
        calls.append(req.url.path)
        if req.url.path=='/v1/models':
            return httpx2.Response(200,json={'object':'list','data':[{'id':settings.model,'object':'model','created':1,'owned_by':'local'}]})
        if req.url.path=='/tokenize':return httpx2.Response(200,json={'count':100})
        generations.append(json.loads(req.content))
        message=({'role':'assistant','content':None,'tool_calls':[{'id':'inspect','type':'function',
                  'function':{'name':'inspect_tile','arguments':'{}'}}]} if len(generations)==1 else
                 {'role':'assistant','content':'{"tool_image_color":"blue"}'})
        return httpx2.Response(200,json=response(message))
    client=runtime._DAsyncOpenAI(api_key='local',base_url=settings.base_url,max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle)))
    inner=LocalChatModel(model=settings.model,openai_client=client,settings=settings)
    monkeypatch.setattr(runtime,'_live_model',lambda _: (inner,client))
    result=await check(config,tmp_path/'check',generate)
    assert result['status']=='passed'
    assert len(generations)==(2 if generate else 0)
    assert all(path in ['/v1/models','/tokenize','/v1/chat/completions'] for path in calls)
    if generate:assert result['tool_image_followup_passed']
