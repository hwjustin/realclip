"""Paired measurement exposure and visual evidence integration; no network."""
import csv
from copy import deepcopy
import json
import socket

from agents import AgentOutputSchema
from PIL import Image
import pytest

from realclip_d import agent_runtime as runtime
from realclip_d.config import Settings
from realclip_d.contracts import CPMetadata, SampleInput
from realclip_d.inputs import image_manifest
from realclip_d.observation_contracts import (AppearanceCheck, ObservationOutput,
    assemble_result, validate_observation_references)
from realclip_d.observation_protocol import prepare_observation
from realclip_d.research_protocol import mock_research_v3
from realclip_d.storage import read_json, verify_stage
from realclip_d.tools.cp import CPTable
from realclip_d.workflow import _describe


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def blocked(*a, **kw):
        raise AssertionError("CP integration tests must remain offline")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(runtime, "AsyncOpenAI", blocked)


@pytest.fixture
def material(tmp_path):
    images = []
    for role in ("treatment", "control"):
        path = tmp_path / f"{role}.png"
        Image.new("L", (16, 16), 0).save(path)
        images.append(dict(image_id=role, path=path, role=role, channel="AGP", field="1"))
    # All treated rows precede all controls: this is the actual pagination failure.
    rows = [dict(feature_id=f"{role}_{name}", feature_name=name, compartment="Cells",
                 channel="AGP", family="AreaShape", role=role, value=value,
                 unit="standardized", scope="well")
            for role, value in (("treatment", 2), ("control", 1)) for name in ("Area", "Shape")]
    path = tmp_path / "cp.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    metadata = CPMetadata(value_kind="preprocessed", normalization_reference="Synthetic common reference.",
        comparison_context="Same fictional acquisition and processing groups.",
        aggregation_method="Synthetic mean per well.", within_feature_order_preserved=True,
        features=[dict(feature_name=name, definition=f"Synthetic {name} measurement.",
                       higher_value_meaning="Larger fictional feature value; no physical effect size.")
                  for name in ("Area", "Shape")])
    sample = SampleInput(sample_id="INTEGRATION", compound={"name": "PRIVATE"}, images=images,
                         cp_file=path, cp_metadata=metadata,
                         cp_preprocessing="Synthetic order-preserving standardized values.")
    research = mock_research_v3(sample, [])
    research.morphology_hypotheses = []; research.observation_guidance = []
    return sample, research, rows


def table_for(sample):
    return CPTable(sample.cp_file, preprocessing=sample.cp_preprocessing, metadata=sample.cp_metadata)


def output(status="contradicts"):
    return dict(sample_id="INTEGRATION", observations=[dict(id="O1",
        text="SYNTHETIC EXAMPLE: compact profiles form a subset, while other profiles remain spread.",
        scope="Fictional supplied fields; the well-level feature does not identify individual cells.",
        image_ids=["treatment", "control"], cp_feature_ids=["treatment_Area", "control_Area"],
        visually_confirmed=True, cp_assessment=dict(status=status,
            reason="The fictional area feature is higher; it does not support an overall smaller-cell claim.",
            effect_on_description="Restrict the sentence to the visually described subset; do not claim a population-wide size decrease."))],
        hypothesis_assessments=[], description_observation_ids=["O1"],
        limitations=["SYNTHETIC FORMAT TEST: no biological interpretation was performed."])


def test_pair_pages_do_not_split_roles_and_support_exact_discovery(material):
    sample, _, _ = material; table = table_for(sample)
    first, second = table.query(limit=1).data, table.query(offset=1, limit=1).data
    assert {r['role'] for r in first['rows']} == {"treatment", "control"}
    assert first['total'] == 2 and first['total_rows'] == 4 and first['next_offset'] == 1
    assert second['next_offset'] is None
    assert first['comparisons'][0]['comparison_ready'] is True
    assert first['comparisons'][0]['numeric_order'] == "treatment_higher"
    assert table.query(query="are").data['total'] == 1
    assert table.query(feature_names=["Shape"]).data['comparisons'][0]['feature_name'] == "Shape"
    assert table.query(feature_names=[]).status == "not_found"
    assert "treatment_Area" not in json.dumps(table.catalogue())


@pytest.mark.parametrize("gap", ["metadata", "normalization", "normalization_unknown", "order", "aggregation", "definition", "missing", "nonfinite", "units", "scope", "duplicate"])
def test_unknown_or_mismatched_measurements_are_never_silently_comparable(material, gap):
    sample, _, rows = material
    if gap == "metadata": sample.cp_metadata = None
    elif gap == "normalization": sample.cp_metadata.normalization_reference = None
    elif gap == "normalization_unknown": sample.cp_metadata.normalization_reference = ' unknown '
    elif gap == "order": sample.cp_metadata.within_feature_order_preserved = None
    elif gap == "aggregation": sample.cp_metadata.aggregation_method = None
    elif gap == "definition": sample.cp_metadata.features = []
    elif gap == "missing": rows = rows[:2]
    elif gap == "nonfinite": rows[2]['value'] = 'NaN'
    elif gap == "units": rows[2]['unit'] = 'different-unit'
    elif gap == "scope": rows[2]['scope'] = 'different-well-scope'
    elif gap == "duplicate": rows.append({**rows[0], "feature_id": "duplicate_identity"})
    with sample.cp_file.open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if gap == "duplicate":
        with pytest.raises(ValueError, match="Duplicate CP measurement identity"): table_for(sample)
        return
    pair=table_for(sample).query(feature_names=["Area"]).data['comparisons'][0]
    assert pair['comparison_ready'] is False and pair['interpretation_gaps']


@pytest.mark.parametrize("status", ["supports", "contradicts", "mixed"])
def test_decisive_cp_requires_exposed_complete_comparable_pairs(material, status):
    sample, research, _ = material; view,_=prepare_observation(sample,research)
    table=table_for(sample); images=image_manifest(sample,include_metadata=True)
    value=ObservationOutput.model_validate(output(status));ids={"treatment_Area","control_Area"}
    validate_observation_references(value,view,images,ids,table)
    with pytest.raises(ValueError,match="not returned"):
        validate_observation_references(value,view,images,{"treatment_Area"},table)
    bad=deepcopy(output(status));bad['observations'][0]['cp_feature_ids']=["treatment_Area","control_Shape"]
    with pytest.raises(ValueError,match="complete treatment/control pairs"):
        validate_observation_references(ObservationOutput.model_validate(bad),view,images,
            {"treatment_Area","control_Shape"},table)
    sample.cp_metadata.within_feature_order_preserved=None
    with pytest.raises(ValueError,match="interpretation gaps"):
        validate_observation_references(value,view,images,ids,table_for(sample))
    bad=output('not_informative')
    validate_observation_references(ObservationOutput.model_validate(bad),view,images,ids,table_for(sample))


def test_presence_in_both_conditions_cannot_support_a_predicted_change():
    check=dict(check="Increased bright-profile fraction",assessment="supported",observation_ids=["O1"],
        reason="Bright objects are present.",contrast=dict(treatment="Bright objects occur.",
            control="Bright objects also occur.",difference="not_observed"))
    with pytest.raises(ValueError,match="not presence alone"):AppearanceCheck.model_validate(check)
    check['assessment']='not_supported';check['contrast']['difference']='unassessable'
    with pytest.raises(ValueError,match="uncertain"):AppearanceCheck.model_validate(check)


def test_schema_requires_explicit_integration_and_prevents_disconnected_cp(material):
    sample,research,_=material;data=output()
    del data['observations'][0]['cp_assessment']
    with pytest.raises(ValueError,match='cp_assessment'):ObservationOutput.model_validate(data)
    data=output();data['observations'][0]['cp_assessment']['status']='not_checked'
    with pytest.raises(ValueError,match='cannot cite'):ObservationOutput.model_validate(data)
    data=output();data['observations'][0]['visually_confirmed']=False
    data['observations'][0]['cp_assessment']['status']='measurement_only'
    with pytest.raises(ValueError,match='Measurement-only'):ObservationOutput.model_validate(data)
    data=output();cp=deepcopy(data['observations'][0]);cp.update(id='CP1',image_ids=[],visually_confirmed=False)
    cp['cp_assessment']['status']='measurement_only';data['observations'].append(cp)
    data['observations'][0]['cp_feature_ids']=[];data['observations'][0]['cp_assessment']['status']='not_checked'
    data['hypothesis_assessments']=[dict(hypothesis_id='H1',assessment='uncertain',reason='Synthetic check.',
        checks=[dict(check='Synthetic size comparison.',assessment='uncertain',observation_ids=['O1','CP1'],
            reason='The two evidence types require integration.',contrast=dict(treatment='Synthetic.',
                control='Synthetic.',difference='unassessable'))])]
    with pytest.raises(ValueError,match='must be integrated'):ObservationOutput.model_validate(data)
    schema=AgentOutputSchema(ObservationOutput).json_schema()
    assert 'cp_assessment' in schema['$defs']['AppearanceObservation']['required']


@pytest.mark.asyncio
async def test_sdk_loop_keeps_conflicting_cp_on_selected_visual_observation(material,tmp_path,monkeypatch):
    from realclip_d import observation_pipeline as pipeline
    sample,research,_=material;original=runtime.ScriptedModel
    payload=output();observation=payload['observations'][0]
    for row in payload['observations']:
        row['comparison_kind']='treatment_control' if row['visually_confirmed'] else 'measurement_only'
    def model(phase,steps,final):
        if phase=='guided':
            return original([[('query_cp',{'feature_names':['Area'],'limit':1})]],
                            {k:v for k,v in payload.items() if k!='description_observation_ids'})
        if phase=='writer':
            return original([],dict(sample_id=sample.sample_id,sentences=[dict(
                id='S1',text=observation['text'],observation_ids=[observation['id']])],limitations=[]))
        if phase=='reviewer':
            return original([],dict(sample_id=sample.sample_id,reviews=[dict(sentence_id='S1',action='keep',
                text=observation['text'],observation_ids=[observation['id']],reason='Synthetic review fixture.')],limitations=[]))
        return original(steps,final)
    monkeypatch.setattr(pipeline,'mock_model_for_phase',model)
    stage=tmp_path/'stage'
    result=await _describe(sample,research,Settings(),stage,'mock',None)
    verify_stage(stage)
    evidence=read_json(stage/'description_evidence.json')['observations'][0]
    assert evidence['cp_assessment']['status']=='contradicts'
    assert set(evidence['cp_feature_ids'])=={'treatment_Area','control_Area'}
    assert 'subset' in evidence['cp_assessment']['effect_on_description']
    assert result.description==evidence['text']
    assert result.schema_version=='4'
    request=read_json(stage/'phases/guided/requests/001.json')
    tool=next(t for t in request['tools'] if t['name']=='query_cp')['parameters']['properties']
    assert {'query','feature_names'}<=set(tool)


def test_private_feature_interpretations_are_rejected_before_model_creation(material):
    sample,research,_=material
    sample.cp_metadata.features[0].definition='PRIVATE identity must never enter the observer.'
    with pytest.raises(ValueError,match='private compound'):prepare_observation(sample,research)
