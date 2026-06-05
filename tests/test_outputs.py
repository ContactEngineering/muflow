"""Tests for the output-schema infrastructure (``muflow/outputs.py``).

``get_outputs_schema`` turns a task's inner ``Outputs`` class into a
JSON-serialisable list of file descriptors (used to document/validate task
outputs). These tests cover the None / no-files / with-schema branches.
"""

import pydantic

from muflow.outputs import OutputFile, get_outputs_schema


class _ResultSchema(pydantic.BaseModel):
    accuracy: float
    loss: float


def test_none_outputs_class_returns_empty_list():
    assert get_outputs_schema(None) == []


def test_outputs_class_without_files_returns_empty_list():
    class NoFiles:
        pass

    assert get_outputs_schema(NoFiles) == []

    class EmptyFiles:
        files = {}

    assert get_outputs_schema(EmptyFiles) == []


def test_file_descriptor_without_schema():
    class Outputs:
        files = {
            "result.txt": OutputFile(file_type="text", description="a log"),
        }

    schema = get_outputs_schema(Outputs)
    assert schema == [
        {
            "filename": "result.txt",
            "file_type": "text",
            "description": "a log",
            "optional": False,
            "schema": None,
        }
    ]


def test_file_descriptor_with_pydantic_schema_is_expanded():
    class Outputs:
        files = {
            "result.json": OutputFile(
                file_type="json",
                description="training results",
                schema=_ResultSchema,
                optional=True,
            ),
        }

    schema = get_outputs_schema(Outputs)
    assert len(schema) == 1
    entry = schema[0]
    assert entry["filename"] == "result.json"
    assert entry["optional"] is True
    # The pydantic model is expanded to its JSON schema.
    assert entry["schema"] == _ResultSchema.model_json_schema()
    assert set(entry["schema"]["properties"]) == {"accuracy", "loss"}


def test_multiple_files_preserve_descriptor_fields():
    class Outputs:
        files = {
            "a.json": OutputFile(file_type="json", schema=_ResultSchema),
            "b.nc": OutputFile(file_type="netcdf", description="grid"),
        }

    schema = {e["filename"]: e for e in get_outputs_schema(Outputs)}
    assert schema["a.json"]["schema"] is not None
    assert schema["b.nc"]["file_type"] == "netcdf"
    assert schema["b.nc"]["schema"] is None
