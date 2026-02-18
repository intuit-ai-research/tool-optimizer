import pytest
from datasets import DatasetDict

from agent_tool_optimizer.inference.application.prompts_builder import PromptsBuilder


@pytest.mark.unit
def test_get_dataset_uses_local_data_when_dataset_id_is_none(mocker):
    """When dataset_id is None, get_dataset() should use local data, not HuggingFace."""

    mock_from_hf = mocker.patch.object(
        PromptsBuilder, "build_dataset_from_huggingface", return_value=None
    )
    
    builder = PromptsBuilder()
    result = builder.build_dataset(dataset_id=None)

    mock_from_hf.assert_not_called()
    assert isinstance(result, DatasetDict)
    assert "test" in result
    assert len(result["test"]) == 5
