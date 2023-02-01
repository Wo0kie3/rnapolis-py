from hypothesis import given
from hypothesis import strategies as st

from rnapolis.common import (
    BasePair,
    BasePhosphate,
    BaseRibose,
    Interaction,
    OtherInteraction,
    Residue,
    ResidueAuth,
    ResidueLabel,
    Stacking,
    Structure2D,
)


@given(st.from_type(ResidueLabel))
def test_rnapdbee_adapters_api_compliance_residue_label(obj):
    assert obj.__dict__.keys() == {"chain", "number", "name"}


@given(st.from_type(ResidueAuth))
def test_rnapdbee_adapters_api_compliance_residue_auth(obj):
    assert obj.__dict__.keys() == {"chain", "number", "icode", "name"}


@given(st.from_type(Residue))
def test_rnapdbee_adapters_api_compliance_residue(obj):
    # explicitly use all properties to make sure they are not added to __dict__ as @cached_property
    obj.chain
    obj.number
    obj.icode
    obj.name
    obj.molecule_type
    obj.full_name
    assert obj.__dict__.keys() == {"label", "auth"}


@given(st.from_type(Interaction))
def test_rnapdbee_adapters_api_compliance_interaction(obj):
    assert obj.__dict__.keys() == {"nt1", "nt2"}


@given(st.from_type(BasePair))
def test_rnapdbee_adapters_api_compliance_base_pair(obj):
    # explicitly use all properties to make sure they are not added to __dict__ as @cached_property
    obj.lw.reverse
    if obj.saenger is not None:
        obj.saenger.is_canonical
    assert obj.__dict__.keys() == {"nt1", "nt2", "lw", "saenger"}


@given(st.from_type(Stacking))
def test_rnapdbee_adapters_api_compliance_stacking(obj):
    # explicitly use all properties to make sure they are not added to __dict__ as @cached_property
    if obj.topology is not None:
        obj.topology.reverse
    assert obj.__dict__.keys() == {"nt1", "nt2", "topology"}


@given(st.from_type(BaseRibose))
def test_rnapdbee_adapters_api_compliance_base_ribose(obj):
    assert obj.__dict__.keys() == {"nt1", "nt2", "br"}


@given(st.from_type(BasePhosphate))
def test_rnapdbee_adapters_api_compliance_base_phosphate(obj):
    assert obj.__dict__.keys() == {"nt1", "nt2", "bph"}


@given(st.from_type(OtherInteraction))
def test_rnapdbee_adapters_api_compliance_other(obj):
    assert obj.__dict__.keys() == {"nt1", "nt2"}


@given(st.from_type(Structure2D))
def test_rnapdbee_adapters_api_compliance_structure2d(obj):
    assert obj.__dict__.keys() == {
        "basePairs",
        "stackings",
        "baseRiboseInteractions",
        "basePhosphateInteractions",
        "otherInteractions",
    }