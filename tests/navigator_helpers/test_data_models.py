import pytest

from pydantic import ValidationError

from navigator_helpers.data_models import ContextualTag, ContextualTags, WeightedValue


def test_contextual_tag():
    with pytest.raises(ValidationError):
        ContextualTag(name="foo", values=[])

    tag = ContextualTag(name="foo", values=["bar"])
    assert not tag.add_values("bar")
    assert not tag.add_values("Bar")
    assert not tag.add_values("bar", weight=0.4)
    assert tag.values == ["bar"]

    assert tag.add_values("baz", weight=0.6) == 1
    assert tag.values == ["bar", WeightedValue(value="baz", weight=0.6)]

    assert not tag.add_values(["baz", "bar"])
    assert tag.values == ["bar", WeightedValue(value="baz", weight=0.6)]

    assert tag.add_values(["one", "two"], weight=0.5) == 2
    assert tag.values == [
        "bar",
        WeightedValue(value="baz", weight=0.6),
        WeightedValue(value="one", weight=0.5),
        WeightedValue(value="two", weight=0.5),
    ]


def test_contextual_tags():
    tags = ContextualTags()
    new_tag = ContextualTag(name="foo", values=["bar"])
    assert tags.add_tag(new_tag) == 1
    assert tags.get_tag_by_name("foo") == new_tag

    # Test adding a tag with the same name
    assert not tags.add_tag(new_tag)
    assert len(tags.tags) == 1

    # New tag for real this time
    assert tags.add_tag(ContextualTag(name="baz", values=["qux"])) == 1
    assert len(tags.tags) == 2
