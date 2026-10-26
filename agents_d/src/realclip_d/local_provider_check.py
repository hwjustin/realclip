"""Probe a user's running local VLM; no biological data or cloud fallback."""
import argparse
import asyncio
import base64
import io
import json
from pathlib import Path

from agents import Agent, ModelTracing, RunConfig, Runner, function_tool
from agents.tool import ToolOutputImage, ToolOutputText
from PIL import Image
from pydantic import BaseModel

from . import agent_runtime as runtime
from .config import load_settings
from .storage import write_json


class TileAnswer(BaseModel):
    tool_image_color: str


def tile(color):
    buffer=io.BytesIO();Image.new('RGB',(32,32),color).save(buffer,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()


async def check(config:Path,output:Path,generate:bool):
    settings=load_settings(config)
    if settings.provider!='openai_compatible':raise ValueError('This check only accepts openai_compatible; no cloud requests are made')
    if output.exists() and any(output.iterdir()):raise ValueError('Use a fresh check output directory')
    settings=settings.model_copy(update={'budget_usd':1.0,'observation_max_turns':2,'max_output_tokens':256})
    inner,client=runtime._live_model(settings)
    result={'provider':settings.provider,'model':settings.model,'generation_requested':generate,
            'biological_validation':False,'gpu_hardware_identity_checked':False}
    try:
        models=await client.models.list()
        result['served_models']=[m.id for m in models.data]
        if settings.model not in result['served_models']:raise ValueError('Configured model is not listed by /v1/models')
        @function_tool
        async def inspect_tile()->list[ToolOutputText|ToolOutputImage]:
            """Return a software-test tile, distinct from the original image."""
            return [ToolOutputText(text='Software-test inspection tile; inspect the returned pixels.'),
                    ToolOutputImage(image_url=tile('blue'),detail='high')]
        messages=[{'role':'user','content':[{'type':'input_text','text':'Call inspect_tile, then return the returned tool image color as one lowercase color word. The tool image, not this initial image, is the target.'},
                   {'type':'input_image','image_url':tile('red'),'detail':'high'}]}]
        kwargs=dict(system_instructions='This is a geometric software capability check, not a cell experiment.',input=messages,
            model_settings=runtime._model_settings(settings,False),tools=[inspect_tile],
            output_schema=None,handoffs=[],tracing=ModelTracing.DISABLED)
        result['input_token_bound']=await inner.count_input((),kwargs,output/'input_count.json')
        if result['input_token_bound']>settings.max_input_tokens:raise ValueError('Probe exceeds configured input token budget')
        if generate:
            ledger=runtime.BudgetLedger(output/'budget.json',1.0,settings)
            recorded=runtime.RecordedModel(inner,settings,output/'probe','observation',ledger,True,False)
            from dataclasses import replace
            agent=Agent(name='Local VLM capability check',instructions=kwargs['system_instructions'],model=recorded,
                model_settings=replace(runtime._model_settings(settings,False),tool_choice='required'),
                tools=[inspect_tile],output_type=TileAnswer)
            value=await Runner.run(agent,messages,max_turns=2,run_config=RunConfig(tracing_disabled=True))
            result['tool_image_color']=value.final_output.tool_image_color
            result['tool_image_followup_passed']=value.final_output.tool_image_color.lower().strip()=='blue'
            if not result['tool_image_followup_passed']:raise ValueError('Tool-image color check failed; review probe requests/responses')
        result['status']='passed'
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        write_json(output/'server_check.json',result)
        await client.close()
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--generate',action='store_true',help='Opt in to two local inference turns to check vision, tool images and structured output')
    args=parser.parse_args()
    print(json.dumps(asyncio.run(check(args.config,args.output,args.generate)),indent=2))


if __name__=='__main__':main()
